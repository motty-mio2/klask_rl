import os

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

    model.save("models/klask_ppo_model")
    print("Training Finished!")


if __name__ == "__main__":
    train()
