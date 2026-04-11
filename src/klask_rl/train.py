from stable_baselines3 import PPO
from env import KlaskEnv
import os

def train():
    # 環境の作成
    env = KlaskEnv()
    
    # モデルのディレクトリ作成
    os.makedirs("models", exist_ok=True)
    
    # 保存されたモデルがあれば読み込む
    model_path = "models/klask_ppo_model"
    if os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        # PPO モデルの初期化
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
    
    print("Starting Training...")
    # 20万ステップ学習
    model.learn(total_timesteps=200000, progress_bar=True)
    
    # モデルの保存
    model.save("models/klask_ppo_model")
    print("Training Finished! Model saved.")

if __name__ == "__main__":
    train()
