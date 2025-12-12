import os
import cv2
import torch
import numpy as np

# モデル読み込み
def load_rvm(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

# アルファマップを適用して背景透明PNGを生成
def composite_foreground(img, fgr, pha):
    fgr = fgr.cpu().numpy().transpose(1,2,0)
    pha = pha.cpu().numpy().transpose(1,2,0)

    out = (fgr * pha + (1 - pha) * 1).clip(0, 1)
    out = (out * 255).astype(np.uint8)
    return out

# 連番画像処理
def process_sequence(input_dir, output_dir, model_path, base_image_path=None, median_blur_ksize=0, erosion_iterations=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_rvm(model_path, device)

    # RVM hidden states
    rec = [None] * 4

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg"))])
    os.makedirs(output_dir, exist_ok=True)

    # If a base image is provided, use it to generate a high-quality initial recurrent state.
    if base_image_path and files:
        first_file = files.pop(0)

        # Load the original first frame
        path = os.path.join(input_dir, first_file)
        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        # Load the provided base image, ensuring it has an alpha channel
        base_img = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)

        if base_img is not None and base_img.shape[2] >= 4:
            # Create a synthetic "perfect" input to generate a clean recurrent state.
            # We composite the foreground (using alpha from base_img) onto a green background.
            alpha = base_img[:, :, 3:4] / 255.0
            green_bg = np.zeros_like(img_rgb)
            green_bg[:, :, 1] = 255 # Green channel

            # Resize alpha mask to match frame dimensions if they differ
            if img_rgb.shape[:2] != alpha.shape[:2]:
                 print(f"Warning: Base image and first frame have different dimensions. Resizing alpha to match.")
                 alpha = cv2.resize(alpha, (img_rgb.shape[1], img_rgb.shape[0]))
                 if alpha.ndim == 2:
                     alpha = alpha[:, :, np.newaxis] # Restore third dimension

            composite_img = (img_rgb * alpha + green_bg * (1 - alpha)).astype(np.uint8)

            # Run inference on the synthetic image to get a good initial 'rec' state
            tensor = torch.from_numpy(composite_img).float() / 255.
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, *rec = model(tensor, *rec)

            # Save the user-provided base image as the output for the first frame
            out_path = os.path.join(output_dir, os.path.splitext(first_file)[0] + ".png")
            cv2.imwrite(out_path, base_img)
            print("processed:", first_file, "(using base image)")
        else:
            print(f"Warning: Could not read base image '{base_image_path}' or it lacks alpha channel. Processing all frames normally.")
            # Put the file back to be processed normally
            files.insert(0, first_file)

    # Process the rest of the sequence
    for file in files:
        path = os.path.join(input_dir, file)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize
        tensor = torch.from_numpy(img_rgb).float() / 255.
        tensor = tensor.permute(2,0,1).unsqueeze(0).to(device)

        # 推論（連続性維持）
        with torch.no_grad():
            fgr, pha, *rec = model(tensor, *rec)

        # --- Alpha Matte Post-processing ---
        alpha_matte = (pha[0].cpu().squeeze().numpy() * 255).astype(np.uint8)

        if median_blur_ksize > 0:
            if median_blur_ksize % 2 == 1:  # Kernel size must be odd
                alpha_matte = cv2.medianBlur(alpha_matte, median_blur_ksize)
            else:
                print(f"Warning: median_blur_ksize ({median_blur_ksize}) is not an odd number. Skipping median blur.")

        if erosion_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha_matte = cv2.erode(alpha_matte, kernel, iterations=erosion_iterations)

        # --- End Post-processing ---

        # 合成して背景透明PNG出力
        rgba = np.dstack([
            (fgr[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8),
            alpha_matte
        ])

        out_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        print("processed:", file)

    print("fin.")


if __name__ == "__main__":
    process_sequence(
        input_dir="input_frames",
        output_dir="output_alpha",
        model_path="models/rvm_resnet50_fp32.torchscript",
        base_image_path=None,  # 精度を上げるために、背景削除済みの画像を指定してください (例: "path/to/your/base_image.png")

        # --- ポストプロセッシング設定 ---
        median_blur_ksize=3,    # アルファマットの平滑化。エッジのジャギーを低減 (奇数、0で無効)
        erosion_iterations=1    # アルファマットの収縮。前景の輪郭を調整し、背景の映り込みを軽減 (0で無効)
    )
