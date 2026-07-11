import os
from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import mujoco  # type: ignore
import numpy as np
from gymnasium import spaces

import config
from utils import MotorFilter, get_device


class KlaskEnv(gym.Env[np.ndarray, np.ndarray]):
    """Klask 環境: カメラ30Hz、CoreXYモータ制限、過去履歴スタック対応版"""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        # config からパラメータを取得
        self.num_history = config.NUM_HISTORY_FRAMES
        self.inference_hz = config.INFERENCE_HZ
        self.num_substeps = config.MUJOCO_SUBSTEPS

        self.metadata: dict[str, Any] = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(self.inference_hz),
        }

        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # 30Hzなどの推論周期に合わせるため、物理演算タイムステップを自動調整
        self.model.opt.timestep = (1.0 / self.inference_hz) / self.num_substeps
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer: Any = None
        self.opponent_policy: Any = None

        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        # 修正: XMLの実際の座標 (pos="0 +/-0.174 0.001") に合わせる
        self.hole_p1 = np.array([0, -0.174])
        self.hole_p2 = np.array([0, 0.174])
        self.hole_radius = 0.018
        self.striker_radius = 0.0074
        self.puck_radius = 0.0075

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 4つの観測要素 [puck_x, puck_y, striker_x, striker_y] * 履歴数
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(4 * self.num_history,), dtype=np.float32
        )

        self.prev_action = np.zeros(2)

        # 履歴バッファ
        self.obs_queue_p1 = deque(maxlen=self.num_history)
        self.obs_queue_p2 = deque(maxlen=self.num_history)

        # 遅延バッファ (アクションの履歴を保持)
        self.action_buffer_p1 = deque(maxlen=10)
        self.action_buffer_p2 = deque(maxlen=10)
        self.latency_steps_p1 = 0
        self.latency_steps_p2 = 0

        # モータフィルタ (物理ステップごとに適用)
        dt = self.model.opt.timestep
        self.motor_filter_p1_x = MotorFilter(0.0, config.MOTOR_MAX_VELOCITY, config.MOTOR_MAX_ACCEL, dt)
        self.motor_filter_p1_y = MotorFilter(0.0, config.MOTOR_MAX_VELOCITY, config.MOTOR_MAX_ACCEL, dt)
        self.motor_filter_p2_x = MotorFilter(0.0, config.MOTOR_MAX_VELOCITY, config.MOTOR_MAX_ACCEL, dt)
        self.motor_filter_p2_y = MotorFilter(0.0, config.MOTOR_MAX_VELOCITY, config.MOTOR_MAX_ACCEL, dt)

        # XML内の各種 ID 検索と初期物理パラメータのバックアップ
        self.puck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "puck_geom")
        self.s1_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "striker_geom")
        self.s2_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "striker_opp_geom")

        self.puck_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.s1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "striker")
        self.s2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "striker_opp")

        self.init_puck_mass = float(self.model.body_mass[self.puck_body_id])
        self.init_s1_mass = float(self.model.body_mass[self.s1_body_id])
        self.init_s2_mass = float(self.model.body_mass[self.s2_body_id])

        self.init_puck_inertia = self.model.body_inertia[self.puck_body_id].copy()
        self.init_s1_inertia = self.model.body_inertia[self.s1_body_id].copy()
        self.init_s2_inertia = self.model.body_inertia[self.s2_body_id].copy()

        # 報酬設計用の状態変数
        self.was_contact = False
        self.prev_puck_y = 0.0

    def _get_obs_p1_single(self) -> np.ndarray:
        p_pos = self.data.qpos[4:6]
        s1_pos = self.data.qpos[0:2] + np.array([0, -0.105])
        return np.array(
            [p_pos[0] / 0.16, p_pos[1] / 0.21, s1_pos[0] / 0.16, s1_pos[1] / 0.21],
            dtype=np.float32,
        )

    def _get_obs_p2_single(self) -> np.ndarray:
        p_pos = self.data.qpos[4:6]
        s2_pos = self.data.qpos[2:4] + np.array([0, 0.105])
        return np.array(
            [-p_pos[0] / 0.16, -p_pos[1] / 0.21, -s2_pos[0] / 0.16, -s2_pos[1] / 0.21],
            dtype=np.float32,
        )

    def _get_obs_p1(self) -> np.ndarray:
        return np.concatenate(self.obs_queue_p1)

    def _get_obs_p2(self) -> np.ndarray:
        return np.concatenate(self.obs_queue_p2)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 学習初期はボールをプレイヤーのすぐ近くに置く (50%の確率)
        if self.np_random.random() < 0.5:
            # プレイヤーの初期位置付近
            self.data.qpos[4] = self.np_random.uniform(-0.05, 0.05)
            self.data.qpos[5] = self.np_random.uniform(-0.15, -0.05)
        else:
            self.data.qpos[4] = self.np_random.uniform(-0.1, 0.1)
            self.data.qpos[5] = self.np_random.uniform(-0.1, 0.15)

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(2)

        # モータフィルタのリセット (初期位置でリセット)
        self.motor_filter_p1_x.reset(self.data.qpos[0])
        self.motor_filter_p1_y.reset(self.data.qpos[1])
        self.motor_filter_p2_x.reset(self.data.qpos[2])
        self.motor_filter_p2_y.reset(self.data.qpos[3])

        # 指令位置も初期物理位置に揃えておく
        self.data.ctrl[0] = self.data.qpos[0]
        self.data.ctrl[1] = self.data.qpos[1]
        self.data.ctrl[2] = self.data.qpos[2]
        self.data.ctrl[3] = self.data.qpos[3]

        # 履歴バッファの初期化
        self.obs_queue_p1.clear()
        self.obs_queue_p2.clear()

        obs_p1_single = self._get_obs_p1_single()
        obs_p2_single = self._get_obs_p2_single()

        for _ in range(self.num_history):
            self.obs_queue_p1.append(obs_p1_single)
            self.obs_queue_p2.append(obs_p2_single)

        # 報酬設計用状態変数のリセット
        self.was_contact = False
        self.prev_puck_y = float(self.data.qpos[5])

        # 遅延バッファの初期化
        self.action_buffer_p1.clear()
        self.action_buffer_p2.clear()
        for _ in range(10):
            self.action_buffer_p1.append(np.zeros(2))
            self.action_buffer_p2.append(np.zeros(2))

        # ドメインランダマイゼーションの適用
        if config.ENABLE_DR:
            # 1. 摩擦のランダマイズ
            puck_fric = self.np_random.uniform(*config.DR_PUCK_FRICTION)
            striker_fric = self.np_random.uniform(*config.DR_STRIKER_FRICTION)
            self.model.geom_friction[self.puck_geom_id, 0] = puck_fric
            self.model.geom_friction[self.s1_geom_id, 0] = striker_fric
            self.model.geom_friction[self.s2_geom_id, 0] = striker_fric

            # 2. 質量と慣性モーメントのランダマイズ
            puck_mass = self.np_random.uniform(*config.DR_PUCK_MASS)
            striker_mass = self.np_random.uniform(*config.DR_STRIKER_MASS)
            
            self.model.body_mass[self.puck_body_id] = puck_mass
            self.model.body_mass[self.s1_body_id] = striker_mass
            self.model.body_mass[self.s2_body_id] = striker_mass
            
            self.model.body_inertia[self.puck_body_id] = self.init_puck_inertia * (puck_mass / self.init_puck_mass)
            self.model.body_inertia[self.s1_body_id] = self.init_s1_inertia * (striker_mass / self.init_s1_mass)
            self.model.body_inertia[self.s2_body_id] = self.init_s2_inertia * (striker_mass / self.init_s2_mass)

            # 3. モータ限界のランダマイズ
            motor_vel = self.np_random.uniform(*config.DR_MOTOR_MAX_VELOCITY)
            motor_acc = self.np_random.uniform(*config.DR_MOTOR_MAX_ACCEL)
            
            self.motor_filter_p1_x.max_vel = motor_vel
            self.motor_filter_p1_x.max_acc = motor_acc
            self.motor_filter_p1_y.max_vel = motor_vel
            self.motor_filter_p1_y.max_acc = motor_acc
            self.motor_filter_p2_x.max_vel = motor_vel
            self.motor_filter_p2_x.max_acc = motor_acc
            self.motor_filter_p2_y.max_vel = motor_vel
            self.motor_filter_p2_y.max_acc = motor_acc

            # 4. 遅延ステップの決定
            self.latency_steps_p1 = int(self.np_random.integers(config.DR_CONTROL_LATENCY[0], config.DR_CONTROL_LATENCY[1] + 1))
            self.latency_steps_p2 = int(self.np_random.integers(config.DR_CONTROL_LATENCY[0], config.DR_CONTROL_LATENCY[1] + 1))
        else:
            # デフォルト値にリセット
            self.model.geom_friction[self.puck_geom_id, 0] = 0.05
            self.model.geom_friction[self.s1_geom_id, 0] = 0.1
            self.model.geom_friction[self.s2_geom_id, 0] = 0.1

            self.model.body_mass[self.puck_body_id] = self.init_puck_mass
            self.model.body_mass[self.s1_body_id] = self.init_s1_mass
            self.model.body_mass[self.s2_body_id] = self.init_s2_mass
            
            self.model.body_inertia[self.puck_body_id] = self.init_puck_inertia
            self.model.body_inertia[self.s1_body_id] = self.init_s1_inertia
            self.model.body_inertia[self.s2_body_id] = self.init_s2_inertia

            self.motor_filter_p1_x.max_vel = config.MOTOR_MAX_VELOCITY
            self.motor_filter_p1_x.max_acc = config.MOTOR_MAX_ACCEL
            self.motor_filter_p1_y.max_vel = config.MOTOR_MAX_VELOCITY
            self.motor_filter_p1_y.max_acc = config.MOTOR_MAX_ACCEL
            self.motor_filter_p2_x.max_vel = config.MOTOR_MAX_VELOCITY
            self.motor_filter_p2_x.max_acc = config.MOTOR_MAX_ACCEL
            self.motor_filter_p2_y.max_vel = config.MOTOR_MAX_VELOCITY
            self.motor_filter_p2_y.max_acc = config.MOTOR_MAX_ACCEL

            self.latency_steps_p1 = 0
            self.latency_steps_p2 = 0

        return self._get_obs_p1(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        # アクションを遅延バッファに保存
        self.action_buffer_p1.append(action.copy())
        
        # 遅延されたアクションの取得
        delayed_idx_p1 = -1 - self.latency_steps_p1
        eff_action_p1 = self.action_buffer_p1[delayed_idx_p1]

        # エージェントアクション（P1目標位置）
        ref_p1_x = eff_action_p1[0] * (self.range_x[1] - self.range_x[0]) / 2
        ref_p1_y = eff_action_p1[1] * (self.range_y[1] - self.range_y[0]) / 2

        # 相手アクション（P2目標位置）
        ref_p2_x = 0.0
        ref_p2_y = 0.0
        if self.opponent_policy is not None:
            # 相手の観測履歴（連結済み）を渡す
            obs_p2 = self._get_obs_p2()
            act_p2, _ = self.opponent_policy.predict(obs_p2, deterministic=True)
            
            # P2アクションを遅延バッファに保存
            self.action_buffer_p2.append(act_p2.copy())
            delayed_idx_p2 = -1 - self.latency_steps_p2
            eff_action_p2 = self.action_buffer_p2[delayed_idx_p2]
            
            # P2は座標反転されて入力され、出力も反転して制御されるため `-` を掛ける
            ref_p2_x = -eff_action_p2[0] * (self.range_x[1] - self.range_x[0]) / 2
            ref_p2_y = -eff_action_p2[1] * (self.range_y[1] - self.range_y[0]) / 2
        else:
            self.action_buffer_p2.append(np.zeros(2))

        reward: float = 0.0
        terminated = False
        p1_hit = False
        max_p_v = 0.0

        # 指定されたサブステップ数だけシミュレーションを進める
        for _ in range(self.num_substeps):
            # モータの速度・加速度制限を適用して指令値を更新
            self.data.ctrl[0] = self.motor_filter_p1_x.update(ref_p1_x)
            self.data.ctrl[1] = self.motor_filter_p1_y.update(ref_p1_y)
            self.data.ctrl[2] = self.motor_filter_p2_x.update(ref_p2_x)
            self.data.ctrl[3] = self.motor_filter_p2_y.update(ref_p2_y)

            mujoco.mj_step(self.model, self.data)
            p_pos = self.data.qpos[4:6]
            p_vel = self.data.qvel[4:6]
            s1_pos = self.data.qpos[0:2] + np.array([0, -0.105])
            s2_pos = self.data.qpos[2:4] + np.array([0, 0.105])

            # --- 厳格な中心座標判定 ---
            if np.linalg.norm(p_pos - self.hole_p2) < self.hole_radius:
                reward += 5000.0
                terminated = True
                break
            if np.linalg.norm(p_pos - self.hole_p1) < self.hole_radius:
                reward -= 2000.0
                terminated = True
                break
            if np.linalg.norm(s1_pos - self.hole_p1) < self.hole_radius:
                reward -= 5000.0
                terminated = True
                break
            if np.linalg.norm(s2_pos - self.hole_p2) < self.hole_radius:
                reward += 2000.0
                terminated = True
                break

            if np.linalg.norm(s1_pos - p_pos) < 0.016:
                p1_hit = True
                p_v_norm = float(np.linalg.norm(p_vel))
                max_p_v = max(max_p_v, p_v_norm)

        if not terminated:
            p_pos_after = self.data.qpos[4:6]
            s1_pos_after = self.data.qpos[0:2] + np.array([0, -0.105])
            dist_after = float(np.linalg.norm(s1_pos_after - p_pos_after))

            # 1. 追いかけ報酬 (磁力報酬)
            reward -= dist_after * 5.0

            # 2. 接触報酬 (「接触し続ける」のではなく「新しく接触した瞬間」のみに限定)
            if p1_hit:
                # 前のステップで未接触だった場合のみ、接触開始ボーナスを与える
                if not self.was_contact:
                    reward += 200.0
                # ボールに十分なエネルギー（速度）を与えた場合のみ、速度ボーナスを与える
                if max_p_v > 0.05:
                    reward += max_p_v * 50.0

            # 3. ボールを奥へやる押し込み報酬 (位置の絶対値ではなく、前ステップより奥に進んだ変化量に基づく)
            # ボールが相手陣地（Y正方向）にあり、かつ前進した（差分がプラス）場合のみ報酬を与える
            puck_y_diff = p_pos_after[1] - self.prev_puck_y
            if p_pos_after[1] > 0 and puck_y_diff > 0:
                reward += puck_y_diff * 100.0

            # 時間・滑らかさ
            reward -= 0.1
            # 振動を抑えるための微小なアクション抑制
            reward -= float(np.linalg.norm(action - self.prev_action) * 0.05)

        self.prev_action = action.copy()

        # 状態変数の更新
        self.was_contact = p1_hit
        self.prev_puck_y = float(self.data.qpos[5]) if not terminated else 0.0

        # 履歴バッファの更新
        obs_p1_single = self._get_obs_p1_single()
        obs_p2_single = self._get_obs_p2_single()
        self.obs_queue_p1.append(obs_p1_single)
        self.obs_queue_p2.append(obs_p2_single)

        obs = self._get_obs_p1()  # 連結された履歴観測
        truncated = self.data.time > 10.0
        return obs, float(reward), terminated, truncated, {}

    def render(self) -> None:
        pass

    def set_opponent_policy(self, policy: Any) -> None:
        self.opponent_policy = policy

