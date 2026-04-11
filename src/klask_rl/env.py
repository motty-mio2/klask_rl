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
        
        # ボールをランダムな位置に配置 (自陣・相手陣含む広範囲)
        self.data.qpos[2] = self.np_random.uniform(-0.15, 0.15) # puck x
        self.data.qpos[3] = self.np_random.uniform(-0.1, 0.2)   # puck y
        
        # ボールに初速を与える (静止したままだと学習が遅いため)
        self.data.qvel[2] = self.np_random.uniform(-0.5, 0.5) # vx
        self.data.qvel[3] = self.np_random.uniform(-1.0, 0.0) # vy (手前に向かってくる)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # アクションを実際の座標範囲に変換して適用
        target_x = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        target_y = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        
        self.data.ctrl[0] = target_x
        self.data.ctrl[1] = target_y
        
        # 以前の状態
        prev_dist = np.linalg.norm(self.data.qpos[0:2] - self.data.qpos[2:4])
        
        # 物理シミュレーションを複数回進める
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        s_pos = self.data.qpos[0:2]
        p_pos = self.data.qpos[2:4]
        p_vel = self.data.qvel[2:4]
        
        reward = 0
        
        # 1. パッキに近づく報酬 (改善: 前のステップとの比較)
        curr_dist = np.linalg.norm(s_pos - p_pos)
        reward += (prev_dist - curr_dist) * 10.0 # 近づいたら報酬、離れたら罰
        
        # 2. パッキが相手側（奥）に動いた時の報酬 (強化)
        if p_pos[1] > -0.1: # 中央付近か奥にある時
            # 相手側への速度に大きな報酬
            reward += max(0, p_vel[1]) * 5.0
        
        # 3. ボールを触ったボーナス (衝突判定)
        if curr_dist < 0.023: # 半径の合計に近い
             reward += 1.0
        
        # 4. 生存/タイムペナルティ (早くゴールさせる)
        reward -= 0.01
        
        # 5. 終了判定
        terminated = False
        if p_pos[1] > 0.2: # 相手ゴール
            reward += 100.0
            terminated = True
        elif p_pos[1] < -0.2: # 自陣ゴール/はみ出し
            reward -= 50.0
            terminated = True
        
        truncated = self.data.time > 15.0 # 15秒
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
