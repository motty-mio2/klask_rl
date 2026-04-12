import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class KlaskEnv(gym.Env):
    """Klask 自己対戦環境: 1つの脳で2つの腕を操る"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # 物理・判定定数
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        self.hole_player_pos = np.array([0, -0.192])
        self.hole_opponent_pos = np.array([0, 0.192])
        self.hole_radius = 0.018
        self.striker_radius = 0.0074
        self.puck_radius = 0.0075
        
        # 行動空間: 4次元 [P1_X, P1_Y, P2_X, P2_Y]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 観測空間: 12次元 [P1_pos(2), P1_vel(2), P2_pos(2), P2_vel(2), Puck_pos(2), Puck_vel(2)]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)

    def _get_obs(self):
        # プレイヤー1 (手前)
        s1_pos = self.data.qpos[0:2]
        s1_vel = self.data.qvel[0:2]
        # プレイヤー2 (相手)
        s2_pos = self.data.qpos[2:4]
        s2_vel = self.data.qvel[2:4]
        # ボール
        p_pos = self.data.qpos[4:6]
        p_vel = self.data.qvel[4:6]
        
        obs = np.concatenate([
            (s1_pos - np.array([0, -0.105])) / 0.2, s1_vel,
            (s2_pos - np.array([0,  0.105])) / 0.2, s2_vel,
            p_pos / 0.3, p_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ボールを中央付近にランダム配置
        self.data.qpos[4] = self.np_random.uniform(-0.1, 0.1)
        self.data.qpos[5] = self.np_random.uniform(-0.1, 0.1)
        self.data.qvel[4:6] = self.np_random.uniform(-0.5, 0.5, size=2)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # アクション適用
        # P1 (ctrl 0,1)
        self.data.ctrl[0] = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        self.data.ctrl[1] = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        # P2 (ctrl 2,3) 相手側なので座標系を考慮 (XML上は同じrange)
        self.data.ctrl[2] = action[2] * (self.range_x[1] - self.range_x[0]) / 2
        self.data.ctrl[3] = action[3] * (self.range_y[1] - self.range_y[0]) / 2
        
        reward = 0.0
        terminated = False
        p1_contact = False
        p2_contact = False
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
            p_pos = self.data.qpos[4:6]
            s1_actual = self.data.qpos[0:2] + np.array([0, -0.105])
            s2_actual = self.data.qpos[2:4] + np.array([0,  0.105])
            
            # 判定チェック
            # P1ゴール (P2の勝ち)
            if np.linalg.norm(p_pos - self.hole_player_pos) < self.hole_radius:
                reward -= 500.0
                terminated = True
                break
            # P2ゴール (P1の勝ち)
            if np.linalg.norm(p_pos - self.hole_opponent_pos) < self.hole_radius:
                reward += 500.0
                terminated = True
                break
            # P1 Klask (自爆)
            if np.linalg.norm(s1_actual - self.hole_player_pos) < self.hole_radius:
                reward -= 500.0
                terminated = True
                break
            # P2 Klask (自爆)
            if np.linalg.norm(s2_actual - self.hole_opponent_pos) < self.hole_radius:
                reward += 500.0
                terminated = True
                break
                
            if np.linalg.norm(s1_actual - p_pos) < (self.striker_radius + self.puck_radius + 0.002):
                p1_contact = True
            if np.linalg.norm(s2_actual - p_pos) < (self.striker_radius + self.puck_radius + 0.002):
                p2_contact = True

        if not terminated:
            # 時間ペナルティ
            reward -= 0.01
            # 接触ボーナス (P1は加点、P2は減点 = P1にとって不利)
            if p1_contact: reward += 5.0
            if p2_contact: reward -= 5.0
            # ボールの位置報酬
            reward -= p_pos[1] * 0.5 # P1はプラス方向(奥)へ、P2はマイナス方向(手前)へ

        obs = self._get_obs()
        truncated = self.data.time > 20.0
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
