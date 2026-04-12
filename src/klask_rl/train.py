from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import KlaskEnv
import os

class SelfPlayCallback(BaseCallback):
    """学習中のモデルを環境の練習相手としてセットするコールバック"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 1000ステップごとに練習相手の思考を更新
        if self.n_calls % 1000 == 0:
            self.training_env.env_method("set_opponent_policy", self.model)
        return True

def train():
    env = KlaskEnv()

    def set_opponent_policy(policy):
        env.opponent_policy = policy

    # env_method で呼べるように monkey patch (DummyVecEnv用)
    env.set_opponent_policy = set_opponent_policy

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
    model.learn(
        total_timesteps=1000000,
        callback=SelfPlayCallback(),
        progress_bar=True
    )

    model.save("models/klask_ppo_model")
    print("Training Finished!")

if __name__ == "__main__":
    train()
