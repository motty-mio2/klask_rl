import numpy as np
import torch as th


def get_device() -> str:
    # if th.backends.mps.is_available():
    #     return "mps"
    # if th.cuda.is_available():
    #     return "cuda"
    return "cpu"


class MotorFilter:
    """CoreXY モータスペック（最高速度・加速度）による指令値制限フィルタ"""

    def __init__(self, init_pos: float, max_vel: float, max_acc: float, dt: float) -> None:
        self.pos = init_pos
        self.vel = 0.0
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dt = dt

    def reset(self, init_pos: float) -> None:
        self.pos = init_pos
        self.vel = 0.0

    def update(self, target_pos: float) -> float:
        # 残り距離
        dx = target_pos - self.pos

        # 減速限界：ブレーキ距離 d = v^2 / (2 * a) から逆算される最大許容速度
        # v_limit = sqrt(2 * a_max * |dx|)
        # 非常に小さい dx の領域での数値的不安定性を防ぐため、目標に直接到達可能な速度を上限とする
        if abs(dx) < 1e-6:
            desired_vel = 0.0
        else:
            v_limit = (2.0 * self.max_acc * abs(dx)) ** 0.5
            desired_vel = np.sign(dx) * min(self.max_vel, v_limit)

        # 加速度制限
        desired_acc = (desired_vel - self.vel) / self.dt
        acc = np.clip(desired_acc, -self.max_acc, self.max_acc)

        # 速度更新
        self.vel += acc * self.dt
        self.vel = np.clip(self.vel, -self.max_vel, self.max_vel)

        # 位置更新
        self.pos += self.vel * self.dt
        return self.pos

