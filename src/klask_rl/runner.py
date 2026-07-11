import subprocess
import time
import sys
import os

def main() -> None:
    # 一括で実行したい実験の設定リスト
    # 形式: (実験名/run-name, seed, ent_coef, penalty_half)
    experiments = [
        # 例1: 異なる探索率（エントロピー係数）の比較
        ("run_ent_001", 42, 0.01, 10.0),
        ("run_ent_002", 42, 0.02, 10.0),
        
        # 例2: 異なる自陣ボール放置ペナルティの比較
        ("run_own_half_p20", 42, 0.01, 20.0),
        
        # 例3: 異なる乱数シードによる再現性の確認
        ("run_seed_100", 100, 0.01, 10.0),
    ]

    processes = []
    print("==================================================")
    print(f"Starting {len(experiments)} training runs in parallel...")
    print("Each run uses 4 sub-processes (SubprocVecEnv) for data collection.")
    print("==================================================")

    # ログ出力用ディレクトリの作成
    log_dir = "models/checkpoints/logs"
    os.makedirs(log_dir, exist_ok=True)

    for run_name, seed, ent_coef, penalty_half in experiments:
        cmd = [
            sys.executable,
            "src/klask_rl/train.py",
            "--run-name", run_name,
            "--seed", str(seed),
            "--ent-coef", str(ent_coef),
            "--penalty-half", str(penalty_half)
        ]
        
        log_path = os.path.join(log_dir, f"{run_name}_runner.log")
        log_file = open(log_path, "w")
        
        # バックグラウンドプロセスとして起動
        p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        processes.append((p, run_name, log_file, log_path))
        
        print(f"🚀 Launched '{run_name}'")
        print(f"   Args -> seed: {seed} | ent_coef: {ent_coef} | penalty_half: {penalty_half}")
        print(f"   Log  -> {log_path}\n")
        
        # プロセス起動時のわずかなファイル競合を防ぐため少し待つ
        time.sleep(2.0)

    print("All runs started. Monitoring progress...")
    print("Waiting for all experiments to finish (Press Ctrl+C to abort all runs)...\n")

    try:
        while len(processes) > 0:
            for p, run_name, log_file, log_path in processes[:]:
                ret = p.poll()
                if ret is not None:
                    # 終了処理
                    log_file.close()
                    processes.remove((p, run_name, log_file, log_path))
                    status_str = "SUCCESS" if ret == 0 else f"FAILED (exit code {ret})"
                    print(f"🏁 Finished: '{run_name}' with status: {status_str}")
            time.sleep(10.0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Runner interrupted by user! Terminating all background runs...")
        for p, run_name, log_file, log_path in processes:
            p.terminate()
            log_file.close()
            print(f"🛑 Terminated: '{run_name}'")
        print("All processes cleaned up.")
        return

    print("\n🎉 All training runs completed!")

if __name__ == "__main__":
    main()
