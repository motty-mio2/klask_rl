import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class KlaskEnv(gym.Env):
    """Klask 強化学習環境: 中心座標による最短判定版"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 実機寸法
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        self.hole_player_pos = np.array([0, -0.192])
        self.hole_opponent_pos = np.array([0, 0.192])
        self.hole_radius = 0.018 # 18mm
        self.striker_radius = 0.0074
        self.puck_radius = 0.0075
        
        # [目標X, 目標Y] (-1 ~ 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # [s_x, s_y, s_vx, s_vy, p_x, p_y, p_vx, p_vy]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        s_pos = self.data.qpos[0:2]
        s_vel = self.data.qvel[0:2]
        p_pos = self.data.qpos[2:4]
        p_vel = self.data.qvel[2:4]
        
        # 状態の正規化
        obs = np.concatenate([
            (s_pos - np.array([0, -0.105])) / 0.2,
            s_vel,
            p_pos / 0.3,
            p_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ボールをランダム配置
        self.data.qpos[2] = self.np_random.uniform(-0.12, 0.12)
        self.data.qpos[3] = self.np_random.uniform(0.0, 0.18)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 1. 行動の適用
        target_x = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        target_y = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        self.data.ctrl[0] = target_x
        self.data.ctrl[1] = target_y
        
        reward = 0.0
        terminated = False
        contact_event = False
        
        # 2. 物理演算と判定のスキャン (5サブステップ)
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
            
            p_pos = self.data.qpos[2:4]
            actual_s_pos = self.data.qpos[0:2] + np.array([0, -0.105])
            
            # --- 中心座標ベースの判定 ---
            
            # 相手ゴール判定 (パッキの中心が穴に入った)
            if np.linalg.norm(p_pos - self.hole_opponent_pos) < self.hole_radius:
                reward = 500.0 # Goal!
                terminated = True
                break
            
            # 自陣落下判定 (ストライカーの中心が穴に入った)
            if np.linalg.norm(actual_s_pos - self.hole_player_pos) < self.hole_radius:
                reward = -500.0 # Klask!
                terminated = True
                break

            # 接触チェック (ボーナス用)
            if np.linalg.norm(actual_s_pos - p_pos) < (self.striker_radius + self.puck_radius + 0.002):
                contact_event = True

        # 3. 報酬の計算
        if not terminated:
            # 時間ペナルティ
            reward -= 0.1
            
            # 接触ボーナス
            if contact_event:
                reward += 10.0
            
            # 補助的な誘導 (ボールを遠ざける)
            reward += (actual_s_pos[1] - p_pos[1]) * 0.1

        obs = self._get_obs()
        truncated = self.data.time > 15.0
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
