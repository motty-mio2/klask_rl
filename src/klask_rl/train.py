import os
from typing import Any

import numpy as np
import torch as th
from env import KlaskEnv
from utils import get_device
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import config


class SelfPlayCallback(BaseCallback):
    """学習中のモデルを環境の練習相手としてセットするコールバック（チェックポイント保存・対戦相手ランダム選択対応）"""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.checkpoint_dir = "models/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.saved_checkpoints = []
        self.policy_cache = {}  # {ファイルパス: PPOモデルオブジェクト}

    def _on_step(self) -> bool:
        # 1. 定期的なチェックポイント保存
        if self.n_calls % config.SAVE_CHECKPOINT_STEPS == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"klask_ppo_model_{self.num_timesteps}_steps")
            self.model.save(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)
            if self.verbose > 0:
                print(f"\n[SelfPlayCallback] Saved checkpoint to {checkpoint_path}")

        # 2. 1000ステップごとに練習相手の思考を更新
        if self.n_calls % 1000 == 0:
            # 80%の確率で最新モデル、20%の確率で過去のチェックポイントからランダム
            # (ただし、過去のチェックポイントが1つ以上保存されている場合)
            if len(self.saved_checkpoints) > 0 and np.random.rand() < 0.2:
                random_cp_path = np.random.choice(self.saved_checkpoints)
                if random_cp_path not in self.policy_cache:
                    self.policy_cache[random_cp_path] = PPO.load(random_cp_path, device=self.model.device)
                
                opponent_model = self.policy_cache[random_cp_path]
                self.training_env.env_method("set_opponent_policy", opponent_model)
                if self.verbose > 0:
                    print(f"\n[SelfPlayCallback] Set opponent to past checkpoint: {random_cp_path}")
            else:
                self.training_env.env_method("set_opponent_policy", self.model)
                if self.verbose > 0:
                    print("\n[SelfPlayCallback] Set opponent to latest model")
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

    device = get_device()
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        tensorboard_log="./klask_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # 探索を強化してすくみから脱出させる
    )

    observation_size = model.observation_space.shape
    if observation_size is None:
        return
    obs_dim = observation_size[0]
    print(f"Starting Self-Play Training ({obs_dim}-in, 2-out)...")
    model.learn(total_timesteps=2000000, callback=SelfPlayCallback(), progress_bar=True)

    # 1. SB3標準形式での保存
    model.save("models/klask_ppo_model")
    print("SB3 Model saved.")

    # 2. ONNX形式での出力 (公式ドキュメント準拠)
    # PPO.policy は ActorCriticPolicy オブジェクト
    policy = model.policy
    onnxable_model = OnnxablePolicy(policy.mlp_extractor, policy.action_net)
    
    # 警告対策: エクスポート前にモデルを評価モード (eval) に設定
    onnxable_model.eval()

    observation_size = model.observation_space.shape
    if observation_size is None:
        return
    
    # 警告対策: テンソルを同じデバイス (CPU / CUDA / MPS) に転送
    dummy_input = th.randn(1, *list(observation_size)).to(device)
    onnx_path = "models/klask_ppo_model.onnx"

    # エラー対策: opset_version を 18 に引き上げて Gemm アダプタエラーを回避
    th.onnx.export(
        onnxable_model,
        (dummy_input,),
        onnx_path,
        opset_version=18,
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

