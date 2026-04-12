import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class KlaskEnv(gym.Env):
    """Klask 環境: 攻撃的 Self-Play 版 (速攻 & クリア重視)"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        xml_path = os.path.join(os.path.dirname(__file__), "../../model/klask.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # ボールを極限まで滑りやすく
        puck_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'puck')
        self.model.geom_friction[puck_id] = [0.003, 0.003, 0.0001]
        
        self.render_mode = render_mode
        self.viewer = None
        self.opponent_policy = None
        
        self.range_x = np.array([-0.1511, 0.1511])
        self.range_y = np.array([-0.0976, 0.0976])
        self.hole_p1 = np.array([0, -0.192])
        self.hole_p2 = np.array([0, 0.192])
        self.hole_radius = 0.018
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        
        self.prev_action = np.zeros(2)

    def _get_obs_p1(self):
        p_pos = self.data.qpos[4:6]
        s1_pos = self.data.qpos[0:2] + np.array([0, -0.105])
        return np.array([p_pos[0]/0.16, p_pos[1]/0.21, s1_pos[0]/0.16, s1_pos[1]/0.21], dtype=np.float32)

    def _get_obs_p2(self):
        p_pos = self.data.qpos[4:6]
        s2_pos = self.data.qpos[2:4] + np.array([0, 0.105])
        return np.array([-p_pos[0]/0.16, -p_pos[1]/0.21, -s2_pos[0]/0.16, -s2_pos[1]/0.21], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # センター付近から開始
        self.data.qpos[4] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[5] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[4:6] = self.np_random.uniform(-0.5, 0.5, size=2)
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(2)
        return self._get_obs_p1(), {}

    def step(self, action):
        p_pos_before = self.data.qpos[4:6].copy()
        s1_pos_before = self.data.qpos[0:2] + np.array([0, -0.105])
        dist_before = np.linalg.norm(s1_pos_before - p_pos_before)

        # アクション適用
        self.data.ctrl[0] = action[0] * (self.range_x[1] - self.range_x[0]) / 2
        self.data.ctrl[1] = action[1] * (self.range_y[1] - self.range_y[0]) / 2
        
        if self.opponent_policy is not None:
            obs_p2 = self._get_obs_p2()
            act_p2, _ = self.opponent_policy.predict(obs_p2, deterministic=True)
            self.data.ctrl[2] = -act_p2[0] * (self.range_x[1] - self.range_x[0]) / 2
            self.data.ctrl[3] = -act_p2[1] * (self.range_y[1] - self.range_y[0]) / 2

        reward = 0.0
        terminated = False
        p1_hit = False
        max_p_forward_vel = 0.0 # 相手方向への最高速度
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            p_pos = self.data.qpos[4:6]
            p_vel = self.data.qvel[4:6]
            s1_pos = self.data.qpos[0:2] + np.array([0, -0.105])
            s2_pos = self.data.qpos[2:4] + np.array([0, 0.105])
            
            # --- 判定 (報酬を大幅増額) ---
            if np.linalg.norm(p_pos - self.hole_p2) < self.hole_radius: reward += 5000.0; terminated = True; break
            if np.linalg.norm(p_pos - self.hole_p1) < self.hole_radius: reward -= 2000.0; terminated = True; break
            if np.linalg.norm(s1_pos - self.hole_p1) < self.hole_radius: reward -= 5000.0; terminated = True; break
            if np.linalg.norm(s2_pos - self.hole_p2) < self.hole_radius: reward += 2000.0; terminated = True; break
            
            # 接触判定 & 相手側への速度成分(vy)を記録
            if np.linalg.norm(s1_pos - p_pos) < 0.016:
                p1_hit = True
                max_p_forward_vel = max(max_p_forward_vel, p_vel[1])

        if not terminated:
            p_pos_after = self.data.qpos[4:6]
            s1_pos_after = self.data.qpos[0:2] + np.array([0, -0.105])
            dist_after = np.linalg.norm(s1_pos_after - p_pos_after)
            
            # 1. 追いかけ報酬 (ある程度近づくまで)
            if dist_after > 0.03: # ボールの近く(3cm)までは引き寄せる
                reward += (dist_before - dist_after) * 30.0
            
            # 2. 攻撃報酬: 相手方向に速く弾く
            if p1_hit and max_p_forward_vel > 0.1:
                reward += max_p_forward_vel * 100.0 # シュートへの大きなインセンティブ
            
            # 3. 陣地ペナルティ: 自陣にボールがあるだけで微量の罰 (早くクリアしろ)
            if p_pos_after[1] < 0:
                reward -= 0.2
            else:
                reward += p_pos_after[1] * 5.0 # 相手陣地に押し込む報酬
            
            # 時間・滑らかさ
            reward -= 0.1
            reward -= np.linalg.norm(action - self.prev_action) * 0.1

        self.prev_action = action.copy()
        obs = self._get_obs_p1()
        truncated = self.data.time > 12.0 # 1試合を少し短縮
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
