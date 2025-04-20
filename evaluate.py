import torch
import json
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.image.fid import FrechetInceptionDistance

def evaluate(model, dataloader, device, num_samples=None):
    """
    Đánh giá mô hình trên tập dữ liệu với các chỉ số SSIM, PSNR và FID.

    Args:
        model       : mô hình MADiffTModel đã huấn luyện
        dataloader  : DataLoader cho tập validation/test
        device      : thiết bị tính toán ('cuda' hoặc 'cpu')
        num_samples : nếu không None, dừng sau khi xử lý num_samples batch

    Trả về:
        dict với các khóa 'ssim', 'psnr', 'fid'
    """
    model.eval()  # chuyển sang chế độ đánh giá, tắt dropout, batchnorm…
    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    # Khởi tạo metric FID
    fid_metric = FrechetInceptionDistance(feature=64).to(device)

    with torch.no_grad():  # không tính gradient để tiết kiệm bộ nhớ
        for batch in tqdm(dataloader, desc="🔍 Evaluating"):
            # batch gồm: masked_img, mask, ảnh gốc, embedding caption, filename
            masked_img, mask, target_img, text_embed, _ = batch

            # Đưa dữ liệu lên device
            masked_img = masked_img.to(device)
            mask       = mask.to(device)
            target_img = target_img.to(device)
            text_embed = text_embed.to(device)

            # Inference và clamp về [0,1]
            output = model(masked_img, mask, text_embed).clamp(0, 1)
            target = target_img.clamp(0, 1)

            # Tính SSIM và PSNR trên thang độ 0-1
            total_ssim += ssim(output, target, data_range=1.0).item()
            total_psnr += psnr(output, target, data_range=1.0).item()

            # Chuyển sang uint8 [0-255] để tính FID
            out_uint8    = (output * 255).byte()
            target_uint8 = (target * 255).byte()

            fid_metric.update(target_uint8, real=True)   # mẫu thật
            fid_metric.update(out_uint8,    real=False)  # mẫu sinh

            count += 1
            if num_samples and count >= num_samples:
                break

    # Trung bình các chỉ số
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    fid_score = fid_metric.compute().item()  # tính FID cuối cùng
    fid_metric.reset()

    # In kết quả
    print("Evaluation Metrics:")
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
    Lưu các chỉ số đánh giá ra file JSON hoặc TXT.

    Args:
        metrics   : dict {'ssim', 'psnr', 'fid'}
        save_path : đường dẫn file lưu
        format    : 'json' hoặc 'txt'
    """
    if format == "json":
        # Ghi file JSON với indent để đọc dễ
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {save_path} (JSON)")
    elif format == "txt":
        # Ghi file văn bản đơn giản
        with open(save_path, "w") as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"FID : {metrics['fid']:.2f}\n")
        print(f"Metrics saved to {save_path} (TXT)")
    else:
        # Nếu format không đúng, raise lỗi
        raise ValueError("format phải là 'json' hoặc 'txt'")
