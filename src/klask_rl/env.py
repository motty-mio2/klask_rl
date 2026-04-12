import gymnasium as gym
from gymnasium import spaces
import mujoco  # type: ignore
import numpy as np
import os
from typing import Any, SupportsFloat


class KlaskEnv(gym.Env[np.ndarray, np.ndarray]):
    """Klask 環境: 壁張り付き解消・ボール追従強化版"""

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer: Any = None
        self.opponent_policy: Any = None

        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        self.hole_p1 = np.array([0, -0.192])
        self.hole_p2 = np.array([0, 0.192])
        self.hole_radius = 0.018
        self.striker_radius = 0.0074
        self.puck_radius = 0.0075

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(4,), dtype=np.float32
        )

        self.prev_action = np.zeros(2)

    def _get_obs_p1(self) -> np.ndarray:
        p_pos = self.data.qpos[4:6]
        s1_pos = self.data.qpos[0:2] + np.array([0, -0.105])
        return np.array(
            [p_pos[0] / 0.16, p_pos[1] / 0.21, s1_pos[0] / 0.16, s1_pos[1] / 0.21],
            dtype=np.float32,
        )

    def _get_obs_p2(self) -> np.ndarray:
        p_pos = self.data.qpos[4:6]
        s2_pos = self.data.qpos[2:4] + np.array([0, 0.105])
        return np.array(
            [-p_pos[0] / 0.16, -p_pos[1] / 0.21, -s2_pos[0] / 0.16, -s2_pos[1] / 0.21],
            dtype=np.float32,
        )

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
        return self._get_obs_p1(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        # アクション適用
        self.data.ctrl[0] = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        self.data.ctrl[1] = action[1] * (self.range_y[1] - self.range_y[0]) / 2

        if self.opponent_policy is not None:
            obs_p2 = self._get_obs_p2()
            act_p2, _ = self.opponent_policy.predict(obs_p2, deterministic=True)
            self.data.ctrl[2] = -act_p2[0] * (self.range_x[1] - self.range_x[0]) / 2
            self.data.ctrl[3] = -act_p2[1] * (self.range_y[1] - self.range_y[0]) / 2

        reward: float = 0.0
        terminated = False
        p1_hit = False
        max_p_v = 0.0

        for _ in range(5):
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

            # 1. 追いかけ報酬 (差分報酬ではなく、距離が近いこと自体への報酬)
            # 距離が離れるほどペナルティを与える (磁力のように引き寄せる)
            reward -= dist_after * 5.0

            # 2. 接触報酬 (大幅増量)
            if p1_hit:
                reward += 200.0 + max_p_v * 50.0

            # 3. ボールを奥へやる報酬
            reward += float(p_pos_after[1] * 10.0)

            # 時間・滑らかさ
            reward -= 0.1
            # 振動を抑えるための微小なアクション抑制
            reward -= float(np.linalg.norm(action - self.prev_action) * 0.05)

        self.prev_action = action.copy()
        obs = self._get_obs_p1()
        truncated = self.data.time > 10.0
        return obs, float(reward), terminated, truncated, {}

    def render(self) -> None:
        pass

    def set_opponent_policy(self, policy: Any) -> None:
        self.opponent_policy = policy
