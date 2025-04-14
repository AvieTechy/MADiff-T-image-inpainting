import torch
import json
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.image.fid import FrechetInceptionDistance

def evaluate(model, dataloader, device, num_samples=None):
    """
    Đánh giá mô hình trên tập dữ liệu với SSIM, PSNR và FID.

    Args:
        model: Mô hình MADiffTModel
        dataloader: DataLoader của tập validation
        device: thiết bị torch (cuda hoặc cpu)
        num_samples: số ảnh tối đa dùng để đánh giá

    Returns:
        dict: {'ssim': float, 'psnr': float, 'fid': float}
    """
    model.eval()
    total_ssim = 0
    total_psnr = 0
    count = 0

    fid_metric = FrechetInceptionDistance(feature=64).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Evaluating"):
            # ✅ unpack 5 phần tử (có cả fname)
            masked_img, mask, target_img, text_embed, _ = batch

            masked_img = masked_img.to(device)
            mask = mask.to(device)
            target_img = target_img.to(device)
            text_embed = text_embed.to(device)

            output = model(masked_img, mask, text_embed).clamp(0, 1)
            target_img = target_img.clamp(0, 1)

            # Tính PSNR và SSIM
            total_ssim += ssim(output, target_img, data_range=1.0).item()
            total_psnr += psnr(output, target_img, data_range=1.0).item()

            # Chuyển sang uint8 (0-255) cho FID
            output_uint8 = (output * 255).byte()
            target_uint8 = (target_img * 255).byte()

            fid_metric.update(target_uint8, real=True)
            fid_metric.update(output_uint8, real=False)

            count += 1
            if num_samples and count >= num_samples:
                break

    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    fid_score = fid_metric.compute().item()
    fid_metric.reset()

    print("📊 Evaluation Metrics:")
    print(f"   SSIM: {avg_ssim:.4f}")
    print(f"   PSNR: {avg_psnr:.2f} dB")
    print(f"   FID : {fid_score:.2f}")

    return {
        "ssim": avg_ssim,
        "psnr": avg_psnr,
        "fid": fid_score
    }

def save_metrics(metrics, save_path="evaluation_metrics.json", format="json"):
    """
    Lưu các chỉ số đánh giá vào file JSON hoặc TXT.

    Args:
        metrics: dict {'ssim', 'psnr', 'fid'}
        save_path: đường dẫn file lưu
        format: 'json' hoặc 'txt'
    """
    if format == "json":
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"💾 Metrics saved to {save_path} (JSON)")
    elif format == "txt":
        with open(save_path, "w") as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"FID : {metrics['fid']:.2f}\n")
        print(f"💾 Metrics saved to {save_path} (TXT)")
    else:
        raise ValueError("❌ format phải là 'json' hoặc 'txt'")
