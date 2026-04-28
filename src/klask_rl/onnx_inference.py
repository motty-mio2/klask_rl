import onnxruntime as ort
import numpy as np
import os

def main() -> None:
    onnx_path = "models/klask_ppo_model.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found. Please run training first.")
        return

    # ONNX Runtime セッションの作成
    # CPU で実行
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    # 入出力名の取得
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Inference using {onnx_path}")
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    # サンプル入力の作成 (バッチサイズ1, 観測値4)
    # 観測値: [Ball_X, Ball_Y, Ego_X, Ego_Y]
    sample_obs = np.array([[0.5, -0.2, 0.1, -0.5]], dtype=np.float32)

    # 推論実行
    # outputs[0] に行動 [Target_X, Target_Y] が返る
    outputs = session.run([output_name], {input_name: sample_obs})
    action = outputs[0]

    print("\nSample Observation: ", sample_obs[0])
    print("Predicted Action (Target XY): ", action[0])

if __name__ == "__main__":
    main()
