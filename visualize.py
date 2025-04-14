import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def visualize_batch_sample(loader, caption_map, n=4):
    """
    Hiển thị ảnh gốc, mask, masked, caption từ 1 batch DataLoader
    - loader: train_loader trả về thêm fname
    - caption_map: dict {filename: caption}
    """

    # Lấy 1 batch
    masked_imgs, masks, images, text_embeds, fnames = next(iter(loader))

    plt.figure(figsize=(n * 3, 11))

    for i in range(n):
        img = TF.to_pil_image(images[i])
        mask = TF.to_pil_image(masks[i].expand(3, -1, -1))
        masked = TF.to_pil_image(masked_imgs[i])
        fname = fnames[i]
        caption = caption_map.get(fname, "No caption")

        # Ảnh gốc
        plt.subplot(3, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Original")

        # Mask
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.title("Mask")

        # Masked ảnh + caption
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(masked)
        plt.axis("off")
        plt.title(f"{caption}", fontsize=9, wrap=True)

    plt.tight_layout()
    plt.show()
