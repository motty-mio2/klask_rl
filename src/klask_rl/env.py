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
        
        # XMLの読み込み
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # レンダリング設定
        self.render_mode = render_mode
        self.viewer = None
        
        # 動作範囲 (正規化用)
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        
        # 行動空間: ストライカーの目標位置 (dx, dy) -1.0 ~ 1.0 に正規化
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 観測空間: 
        # [striker_x, striker_y, striker_vx, striker_vy, puck_x, puck_y, puck_vx, puck_vy]
        # すべて正規化した値とする
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        # 内部状態から正規化した観測値を取得
        s_pos = self.data.qpos[0:2] # [x, y] slider joints
        s_vel = self.data.qvel[0:2]
        p_pos = self.data.qpos[2:4] # puck x, y (puck_free joint)
        p_vel = self.data.qvel[2:4]
        
        obs = np.concatenate([
            (s_pos - np.mean(self.range_x)) / (self.range_x[1] - self.range_x[0]),
            s_vel,
            p_pos / 0.3, # 盤面サイズでおおよそ正規化
            p_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ボールを少しランダムな位置に配置
        self.data.qpos[2] = self.np_random.uniform(-0.1, 0.1) # puck x
        self.data.qpos[3] = self.np_random.uniform(0.05, 0.15) # puck y
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # アクションを実際の座標範囲に変換して適用
        target_x = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        target_y = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        
        self.data.ctrl[0] = target_x
        self.data.ctrl[1] = target_y
        
        # 以前のパッキ位置を保存
        prev_puck_pos = self.data.qpos[2:4].copy()
        
        # 物理シミュレーションを複数回進める
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # 状態の取得
        s_pos = self.data.qpos[0:2]
        p_pos = self.data.qpos[2:4]
        p_vel = self.data.qvel[2:4]
        
        # --- 報酬設計 (Reward Engineering) ---
        reward = 0
        
        # 1. パッキに近づく報酬 (学習初期の誘導)
        dist_to_puck = np.linalg.norm(s_pos - p_pos)
        reward -= dist_to_puck * 0.1
        
        # 2. パッキが相手側（奥）に動いた時の報酬
        puck_vel_y = p_vel[1] # Y軸正の方向が相手側
        if p_pos[1] > -0.1: # 自陣の半分より奥にパッキがある時
            reward += max(0, -puck_vel_y) * 2.0 # 相手側に強く弾けば報酬
        
        # 3. パッキの位置による報酬 (奥にあるほど良い)
        reward += (0.1 - p_pos[1]) * 0.5
        
        # 4. 生存報酬
        reward += 0.01
        
        # 5. 終了判定と大きな報酬/ペナルティ
        terminated = False
        # 相手ゴール (Y=0.21)
        if p_pos[1] > 0.2:
            reward += 50.0
            terminated = True
        # 自陣ゴール (Y=-0.21)
        elif p_pos[1] < -0.2:
            reward -= 50.0
            terminated = True
        
        # タイムアウト
        truncated = self.data.time > 10.0 # 10秒でリセット
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
