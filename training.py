import os
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

def load_checkpoint(model, checkpoint_path, device="cuda"):
    """
    Load model weights from a checkpoint file.
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded model weights.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def training(model, train_loader, is_continue=False,checkpoint_path="best_checkpoint.pth", num_epochs=100, device="cuda", lr=1e-4, start_epoch=0):
    """
    Train the model with optional resume from checkpoint.

    Args:
        is_continue (bool): If True, resume from the given checkpoint.
        checkpoint_path (str): Path to the checkpoint file.
        num_epochs (int): Total number of epochs to train.
    """
    # Setup optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    best_loss = float("inf")
    # Resume from checkpoint if required
    if is_continue:
        load_checkpoint(model, checkpoint_path)
        # If you store epoch info in checkpoint, set start_epoch accordingly
    num_epochs += start_epoch
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for masked_img, mask, target_img, text_embed, _ in pbar:
            # Move inputs to device
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            target_img = target_img.to(device)
            text_embed = text_embed.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward and backward
            with autocast(device_type=device.type):
                output = model(masked_img, mask, text_embed)
                loss = total_inpainting_loss(output, target_img, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Avg loss @ epoch {epoch+1}: {avg_loss:.4f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint with loss {best_loss:.4f}")
