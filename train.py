import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from Utils.utils import *
from Utils.models import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_dir", type=str, default="content_data")
    parser.add_argument("--style_dir", type=str, default="style_data")
    parser.add_argument("--vgg", type=str, default="vgg_normalised.pth")
    parser.add_argument("--experiment", type=str, default="adain_6000")
    parser.add_argument("--final_size", type=int, default=256)
    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_batches_per_epoch", type=int, default=6000)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    return parser.parse_args()


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main():
    args = parse_arguments()

    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device, flush=True)

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "args.txt", "w", encoding="utf-8") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")

    content_transform = get_transform(
        args.content_size, args.crop, args.final_size
    )
    style_transform = get_transform(
        args.style_size, args.crop, args.final_size
    )

    print("Loading content dataset...", flush=True)
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)

    print("Loading style dataset...", flush=True)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    if len(content_dataset) < args.batch_size:
        raise ValueError(
            f"Content dataset has {len(content_dataset)} images, "
            f"but batch_size is {args.batch_size}."
        )
    if len(style_dataset) < args.batch_size:
        raise ValueError(
            f"Style dataset has {len(style_dataset)} images, "
            f"but batch_size is {args.batch_size}."
        )

    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    content_dataloader = DataLoader(content_dataset, **loader_options)
    style_dataloader = DataLoader(style_dataset, **loader_options)

    total_batches = min(len(content_dataloader), len(style_dataloader))
    batches_per_epoch = min(args.max_batches_per_epoch, total_batches)

    if batches_per_epoch <= 0:
        raise ValueError("No batches available. Check the dataset folders and batch size.")

    print("Content images:", len(content_dataset), flush=True)
    print("Style images:", len(style_dataset), flush=True)
    print("Available batches:", total_batches, flush=True)
    print("Batches per epoch:", batches_per_epoch, flush=True)
    print("Total epochs:", args.epochs, flush=True)

    vgg_path = Path(args.vgg)
    if not vgg_path.is_file():
        raise FileNotFoundError(f"VGG weights not found: {vgg_path.resolve()}")

    print("Loading VGG encoder...", flush=True)
    encoder = VGGEncoder(str(vgg_path)).to(device)
    decoder = Decoder().to(device)

    encoder.eval()
    encoder.requires_grad_(False)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    start_epoch = 0
    resume_batch = 0
    global_step = 0

    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        required_keys = (
            "decoder_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "batch_in_epoch",
            "global_step",
        )
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint is missing keys: {missing_keys}")

        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"])
        resume_batch = int(checkpoint["batch_in_epoch"])
        global_step = int(checkpoint["global_step"])

        print(
            f"Resuming from epoch {start_epoch + 1}, "
            f"starting at batch {resume_batch + 1}.",
            flush=True
        )
        print("Global step:", global_step, flush=True)

    print("Training starting...", flush=True)

    mse_loss = torch.nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        skip_batches = resume_batch if epoch == start_epoch else 0

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        processed_batches = 0
        last_content_batch = None
        last_style_batch = None
        last_output = None

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=batches_per_epoch,
            initial=min(skip_batches, batches_per_epoch),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            mininterval=0.5
        )

        for batch_idx, (content_batch, style_batch) in enumerate(progress_bar):
            if batch_idx < skip_batches:
                continue
            if batch_idx >= batches_per_epoch:
                break

            content_batch = content_batch.to(device, non_blocking=True)
            style_batch = style_batch.to(device, non_blocking=True)

            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)
                target = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            output = decoder(target)
            generated_feats = encoder(output)

            loss_c = mse_loss(generated_feats[-1], target) * args.content_weight

            loss_s = torch.zeros((), device=device)
            for generated_feature, style_feature in zip(generated_feats, s_feats):
                generated_mean, generated_std = cal_mean_std(generated_feature)
                style_mean, style_std = cal_mean_std(style_feature)
                loss_s = loss_s + mse_loss(generated_mean, style_mean)
                loss_s = loss_s + mse_loss(generated_std, style_std)

            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            processed_batches += 1
            global_step += 1

            last_content_batch = content_batch.detach()
            last_style_batch = style_batch.detach()
            last_output = output.detach()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                content=f"{loss_c.item():.4f}",
                style=f"{loss_s.item():.4f}",
                step=global_step
            )

            if global_step % args.checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "batch_in_epoch": batch_idx + 1,
                    "global_step": global_step,
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                }
                checkpoint_path = save_dir / "latest_checkpoint.pt"
                save_checkpoint(checkpoint_path, checkpoint)
                tqdm.write(f"Checkpoint saved: {checkpoint_path} at step {global_step}")

        progress_bar.close()
        resume_batch = 0

        if processed_batches > 0:
            running_loss /= processed_batches
            running_closs /= processed_batches
            running_sloss /= processed_batches

            if (epoch + 1) % args.log_interval == 0:
                print(
                    f"Epoch {epoch + 1}: Loss={running_loss:.4f}, "
                    f"Content Loss={running_closs:.4f}, "
                    f"Style Loss={running_sloss:.4f}",
                    flush=True
                )

            if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                image_grid = torch.cat(
                    [last_content_batch, last_style_batch, last_output],
                    dim=0
                )
                save_image(
                    image_grid.cpu(),
                    save_dir / f"output_{epoch + 1}.png",
                    nrow=args.batch_size
                )

        scheduler.step()

        checkpoint = {
            "epoch": epoch + 1,
            "batch_in_epoch": 0,
            "global_step": global_step,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        }
        checkpoint_path = save_dir / "latest_checkpoint.pt"
        save_checkpoint(checkpoint_path, checkpoint)

        print(
            f"Epoch {epoch + 1} completed. Checkpoint saved to {checkpoint_path}.",
            flush=True
        )

    print("Training completed!", flush=True)


if __name__ == "__main__":
    main()
