import os
import argparse
from typing import Any

import numpy as np
import torch as th
from env import KlaskEnv
from utils import get_device
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import config


class SelfPlayCallback(BaseCallback):
    """学習中のモデルを環境の練習相手としてセットするコールバック（チェックポイント保存・対戦相手ランダム選択対応）"""

    def __init__(self, run_name: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_name = run_name
        self.checkpoint_dir = f"models/checkpoints/{run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.saved_checkpoints = []

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
                # サブプロセスにはパスの文字列を転送
                self.training_env.env_method("set_opponent_policy", random_cp_path)
                if self.verbose > 0:
                    print(f"\n[SelfPlayCallback] Set opponent to past checkpoint: {random_cp_path}")
            else:
                # 最新モデルを一時ファイルとして保存し、そのパスを送信
                latest_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_latest")
                self.model.save(latest_path)
                self.training_env.env_method("set_opponent_policy", latest_path)
                if self.verbose > 0:
                    print(f"\n[SelfPlayCallback] Set opponent to latest model: {latest_path}")
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
    parser = argparse.ArgumentParser(description="Train Klask RL Agent with Self-Play and DR")
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Identifier name for this training run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment and model"
    )
    args = parser.parse_args()

    # 並列環境の構築 (SubprocVecEnv による 4 並列実行)
    env = make_vec_env(
        KlaskEnv,
        n_envs=config.NUM_ENVS,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )

    os.makedirs(f"models/checkpoints/{args.run_name}", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Run Name: {args.run_name} | Seed: {args.seed} | Parallel Envs: {config.NUM_ENVS}")

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        tensorboard_log=f"./klask_tensorboard/{args.run_name}/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        seed=args.seed,
    )

    observation_size = model.observation_space.shape
    if observation_size is None:
        return
    obs_dim = observation_size[0]
    print(f"Starting Self-Play Training ({obs_dim}-in, 2-out)...")
    
    callback = SelfPlayCallback(run_name=args.run_name, verbose=1)
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)

    # 1. SB3標準形式での保存
    model_path = f"models/klask_ppo_model_{args.run_name}"
    model.save(model_path)
    print(f"SB3 Model saved to {model_path}.zip")

    # 2. ONNX形式での出力 (公式ドキュメント準拠)
    # PPO.policy は ActorCriticPolicy オブジェクト
    policy = model.policy
    onnxable_model = OnnxablePolicy(policy.mlp_extractor, policy.action_net)
    
    # 警告対策: エクスポート前にモデルを評価モード (eval) に設定
    onnxable_model.eval()

    # 警告対策: テンソルを同じデバイス (CPU / CUDA / MPS) に転送
    dummy_input = th.randn(1, *list(observation_size)).to(device)
    onnx_path = f"models/klask_ppo_model_{args.run_name}.onnx"

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

