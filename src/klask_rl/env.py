import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class KlaskEnv(gym.Env):
    """Klask を MuJoCo で学習するための環境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        
        # 穴の定義 (中心座標と半径)
        self.hole_player_pos = np.array([0, -0.192])
        self.hole_opponent_pos = np.array([0, 0.192])
        self.hole_radius = 0.018
        self.striker_radius = 0.0074
        self.puck_radius = 0.015 # 仮の半径
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        s_pos = self.data.qpos[0:2]
        s_vel = self.data.qvel[0:2]
        p_pos = self.data.qpos[2:4]
        p_vel = self.data.qvel[2:4]
        
        obs = np.concatenate([
            (s_pos - np.array([0, -0.105])) / 0.2, # 正規化を微調整
            s_vel,
            p_pos / 0.3,
            p_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ボールをランダムな位置に配置
        self.data.qpos[2] = self.np_random.uniform(-0.15, 0.15)
        self.data.qpos[3] = self.np_random.uniform(-0.1, 0.18)
        
        # ボールに初速
        self.data.qvel[2] = self.np_random.uniform(-0.5, 0.5)
        self.data.qvel[3] = self.np_random.uniform(-1.0, 0.0)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        target_x = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        target_y = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        
        self.data.ctrl[0] = target_x
        self.data.ctrl[1] = target_y
        
        prev_puck_dist_to_goal = np.linalg.norm(self.data.qpos[2:4] - self.hole_opponent_pos)
        
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        s_pos = self.data.qpos[0:2]
        p_pos = self.data.qpos[2:4]
        
        reward = 0
        terminated = False
        
        # 1. パッキが相手の穴（ゴール）に入ったか
        dist_puck_to_goal = np.linalg.norm(p_pos - self.hole_opponent_pos)
        if dist_puck_to_goal < self.hole_radius:
            reward += 200.0 # Goal!
            terminated = True
        
        # 2. ストライカーが自分の穴（Klask!）に入ったか
        # ストライカーの中心座標 pos="0 -0.105" に対する相対座標 qpos[0:2] から計算
        actual_s_pos = s_pos + np.array([0, -0.105])
        dist_striker_to_hole = np.linalg.norm(actual_s_pos - self.hole_player_pos)
        if dist_striker_to_hole < (self.hole_radius - self.striker_radius * 0.5):
            reward -= 150.0 # Klask!
            terminated = True
            
        # 3. 誘導報酬: パッキを相手の穴に近づける
        reward += (prev_puck_dist_to_goal - dist_puck_to_goal) * 20.0
        
        # 4. 誘導報酬: パッキに近づく
        dist_s_to_p = np.linalg.norm(actual_s_pos - p_pos)
        reward -= dist_s_to_p * 0.1
        
        # 自陣ゴール（穴以外）
        if p_pos[1] < -0.21:
            reward -= 50.0
            terminated = True
            
        truncated = self.data.time > 15.0
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
