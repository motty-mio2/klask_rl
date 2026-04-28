import os
from typing import Any

import torch as th
from env import KlaskEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCallback(BaseCallback):
    """学習中のモデルを環境の練習相手としてセットするコールバック"""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 1000ステップごとに練習相手の思考を更新
        if self.n_calls % 1000 == 0:
            # training_env は VecEnv なので env_method を使用
            self.training_env.env_method("set_opponent_policy", self.model)
        return True


class OnnxablePolicy(th.nn.Module):
    """
    公式ドキュメント推奨のONNX出力用ラッパー。
    不要な分布計算を省き、決定論的な行動出力を保証します。
    """

    def __init__(self, extractor: th.nn.Module, action_net: th.nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # extractor で特徴量を抽出し、action_net で行動を出力
        # 内部的にはこれが deterministic=True の時の挙動と一致します
        action_hidden, _ = self.extractor(observation)
        return self.action_net(action_hidden)


def train() -> None:
    env = KlaskEnv()

    os.makedirs("models", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./klask_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
    )

    print("Starting Self-Play Training (4-in, 2-out)...")
    model.learn(total_timesteps=1000000, callback=SelfPlayCallback(), progress_bar=True)

    # 1. SB3標準形式での保存
    model.save("models/klask_ppo_model")
    print("SB3 Model saved.")

    # 2. ONNX形式での出力 (公式ドキュメント準拠)
    # PPO.policy は ActorCriticPolicy オブジェクト
    policy = model.policy
    onnxable_model = OnnxablePolicy(policy.mlp_extractor, policy.action_net)

    observation_size = model.observation_space.shape
    if observation_size is None:
        return
    dummy_input = th.randn(1, *list(observation_size))
    onnx_path = "models/klask_ppo_model.onnx"

    th.onnx.export(
        onnxable_model,
        (dummy_input,),
        onnx_path,
        opset_version=11,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    print(f"ONNX Model exported to {onnx_path}")


if __name__ == "__main__":
    train()
