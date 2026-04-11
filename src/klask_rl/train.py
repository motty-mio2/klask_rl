from stable_baselines3 import PPO
from env import KlaskEnv
import os

def train():
    # 環境の作成
    env = KlaskEnv()
    
    # モデルのディレクトリ作成
    os.makedirs("models", exist_ok=True)
    
    # PPO モデルの初期化
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./klask_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    
    print("Starting Training...")
    # 10万ステップ学習させてみる
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # モデルの保存
    model.save("models/klask_ppo_model")
    print("Training Finished! Model saved.")

if __name__ == "__main__":
    train()
