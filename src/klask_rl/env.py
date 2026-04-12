import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class KlaskEnv(gym.Env):
    """Klask 環境: 極小入出力版 (Input:4, Output:2)"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 定数
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        self.hole_player_pos = np.array([0, -0.192])
        self.hole_opponent_pos = np.array([0, 0.192])
        self.hole_radius = 0.018
        
        # 行動空間: [目標X, 目標Y] (-1.0 ~ 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 観測空間: [ボールX, ボールY, 自身のX, 自身のY]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)

    def _get_obs(self):
        # ボールの位置 (全体座標系)
        p_pos = self.data.qpos[4:6]
        # 自身の位置 (初期位置 0, -0.105 からの相対座標)
        s_pos = self.data.qpos[0:2]
        actual_s_pos = s_pos + np.array([0, -0.105])
        
        # おおよそ -1 ~ 1 に正規化して返す
        obs = np.array([
            p_pos[0] / 0.16,
            p_pos[1] / 0.21,
            actual_s_pos[0] / 0.16,
            actual_s_pos[1] / 0.21
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ボールを中央付近に配置
        self.data.qpos[4] = self.np_random.uniform(-0.1, 0.1)
        self.data.qpos[5] = self.np_random.uniform(-0.05, 0.15)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # P1のアクション適用
        self.data.ctrl[0] = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        self.data.ctrl[1] = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        
        # 相手側(P2)はボールを軽く追いかけるだけの簡易的な動きをさせる
        p_pos = self.data.qpos[4:6]
        self.data.ctrl[2] = np.clip(p_pos[0], self.range_x[0], self.range_x[1])
        self.data.ctrl[3] = np.clip(p_pos[1] - 0.105, self.range_y[0], self.range_y[1])

        reward = 0.0
        terminated = False
        contact_event = False
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
            p_pos = self.data.qpos[4:6]
            actual_s_pos = self.data.qpos[0:2] + np.array([0, -0.105])
            
            # ゴール判定 (中心座標ベース)
            if np.linalg.norm(p_pos - self.hole_opponent_pos) < self.hole_radius:
                reward = 500.0 # P1 Goal!
                terminated = True
                break
            if np.linalg.norm(p_pos - self.hole_player_pos) < self.hole_radius:
                reward = -200.0 # P1 Goaled (P2 hit)
                terminated = True
                break
            
            # 自爆判定 (Klask!)
            if np.linalg.norm(actual_s_pos - self.hole_player_pos) < self.hole_radius:
                reward = -500.0
                terminated = True
                break
                
            # 接触チェック
            if np.linalg.norm(actual_s_pos - p_pos) < 0.015:
                contact_event = True

        if not terminated:
            reward -= 0.1 # 時間ペナルティ
            if contact_event:
                reward += 10.0 # 接触ボーナス
            # ボールを奥に送る動機づけ
            reward += (0.2 - p_pos[1]) * 0.2

        obs = self._get_obs()
        truncated = self.data.time > 15.0
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
