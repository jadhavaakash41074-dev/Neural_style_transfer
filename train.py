
import argparse
import torch
from pathlib import Path
from Utils.utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
from Utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str,
                        default=r"A:\Downloads\Adain_NST_project\content_data",
                        help="Location of content dataset")

    parser.add_argument('--style_dir', type=str,
                        default=r"A:\Downloads\Adain_NST_project\style_data",
                        help="Location of style dataset")

    parser.add_argument('--vgg', type=str,
                        default=r"A:\Downloads\Adain_NST_project\vgg_normalised.pth",
                        help="Location of pre-trained VGG")

    parser.add_argument('--experiment', type=str, default="adain_6000",
                        help="Name of experiment")

    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')

    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')

    parser.add_argument('--style_size', type=int, default=512,
                        help="Size of style image")

    parser.add_argument('--crop', action='store_true', default=True,
                        help="Crop image")

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')

    parser.add_argument('--max_batches_per_epoch', type=int, default=6000,
                        help='Maximum batches per epoch')

    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save checkpoint every N batches')

    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content Weight')

    parser.add_argument('--style_weight', type=float, default=10,
                        help="Style weight")

    parser.add_argument('--log_interval', type=int, default=1,
                        help="Log Interval")

    parser.add_argument('--save_interval', type=int, default=1,
                        help="Save output image every N epochs")

    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to a full training checkpoint")

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Training device:", device, flush=True)

    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

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

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=2,
        persistent_workers=True
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=2,
        persistent_workers=True
    )

    total_batches = min(
        len(content_dataloader),
        len(style_dataloader)
    )
    batches_per_epoch = min(
        args.max_batches_per_epoch,
        total_batches
    )

    if batches_per_epoch <= 0:
        raise ValueError("No batches available. Check your datasets.")

    print("Content images:", len(content_dataset), flush=True)
    print("Style images:", len(style_dataset), flush=True)
    print("Available batches:", total_batches, flush=True)
    print("Batches per epoch:", batches_per_epoch, flush=True)
    print("Total epochs:", args.epochs, flush=True)

    print("Loading VGG encoder...", flush=True)
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

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
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        resume_batch = checkpoint['batch_in_epoch']
        global_step = checkpoint['global_step']

        print(
            f"Resuming epoch {start_epoch + 1}, "
            f"after batch {resume_batch}",
            flush=True
        )
        print("Global step:", global_step, flush=True)

    print("Training starting...", flush=True)

    mse_loss = torch.nn.MSELoss()
    encoder.eval()

    for epoch in range(start_epoch, args.epochs):
        decoder.train()

        skip_batches = resume_batch if epoch == start_epoch else 0

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        processed_batches = 0

        # Progress bar starts before batches are processed
        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=batches_per_epoch,
            initial=min(skip_batches, batches_per_epoch),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            mininterval=0.5
        )

        for batch_idx, (content_batch, style_batch) in enumerate(progress_bar):

            # Skip batches already processed before the last checkpoint
            if batch_idx < skip_batches:
                continue

            if batch_idx >= batches_per_epoch:
                break

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(
                c_feats[-1], s_feats[-1]
            )

            g = decoder(t)
            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = cal_mean_std(g_f)
                s_mean, s_std = cal_mean_std(s_f)

                loss_s += (
                    mse_loss(g_mean, s_mean)
                    + mse_loss(g_std, s_std)
                )

            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            processed_batches += 1
            global_step += 1

            # Update progress bar after every batch
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                content=f"{loss_c.item():.4f}",
                style=f"{loss_s.item():.4f}",
                step=global_step
            )

            # Save a complete checkpoint every N batches
            if global_step % args.checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch_in_epoch': batch_idx + 1,
                    'global_step': global_step,
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'args': vars(args)
                }

                checkpoint_path = save_dir / 'latest_checkpoint.pt'
                torch.save(checkpoint, checkpoint_path)

                tqdm.write(
                    f"Checkpoint saved: {checkpoint_path} "
                    f"(epoch {epoch + 1}, batch {batch_idx + 1}, "
                    f"step {global_step})"
                )

        progress_bar.close()
        resume_batch = 0

        if processed_batches > 0:
            running_loss /= processed_batches
            running_closs /= processed_batches
            running_sloss /= processed_batches

            if (epoch + 1) % args.log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}: "
                    f"Loss: {running_loss:.4f}, "
                    f"Content Loss: {running_closs:.4f}, "
                    f"Style Loss: {running_sloss:.4f}"
                )

            if (epoch + 1) % args.save_interval == 0:
                with torch.no_grad():
                    with torch.no_grad():
                        output = torch.cat(
                            [
                                content_batch.to(device),
                                style_batch.to(device),
                                g.detach().to(device)
                            ],
                            dim=0
                        )

                        save_image(
                            output.cpu(),
                            save_dir / f'output_{epoch + 1}.png',
                            nrow=args.batch_size
                        )

        # Complete the epoch and update the learning-rate scheduler
        scheduler.step()

        # Save checkpoint at the epoch boundary
        checkpoint = {
            'epoch': epoch + 1,
            'batch_in_epoch': 0,
            'global_step': global_step,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args)
        }

        checkpoint_path = save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)

        print(
            f"Epoch {epoch + 1} completed. "
            f"Checkpoint saved.",
            flush=True
        )

    print("Training completed!", flush=True)


if __name__ == '__main__':
    main()
