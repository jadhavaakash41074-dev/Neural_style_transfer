from torch.utils.data import Dataset
import os
import json

from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform

        # Cache file: ek baar scan ke baad list yahan save hoti hai
        manifest_path = os.path.join(root, "_manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.files = json.load(f)
            print(f"Loaded manifest for {root}: {len(self.files)} images", flush=True)
            return

        self.files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        max_pixels = 50_000_000

        # Pehle sirf file paths collect karo (fast)
        candidates = []
        for current_root, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(valid_extensions):
                    candidates.append(os.path.join(current_root, filename))

        for image_path in tqdm(candidates, desc=f"Scanning {os.path.basename(root)}"):
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    # Skip unusually large images
                    if width * height > max_pixels:
                        continue
                self.files.append(image_path)
            except (OSError, Image.DecompressionBombError):
                continue

        self.files.sort()

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.files, f)

        print(f"Valid images found in {root}: {len(self.files)}", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]

        try:
            with Image.open(image_path) as image:
                image = image.convert('RGB')

                if self.transform:
                    image = self.transform(image)

                return image

        except (OSError, Image.DecompressionBombError) as e:
            # If a file becomes unreadable after scanning, try another image
            print(f"Skipping unreadable image: {image_path}")
            print(f"Reason: {e}")

            if len(self.files) <= 1:
                raise RuntimeError("No alternative readable image available.")

            return self.__getitem__((idx + 1) % len(self.files))


def get_transform(size, crop, final_size):
    transform_list = []

    # Pehle bada size (512), phir random crop (256)
    transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    #[batch size, channels, height, width]
    size = content_feat.size()
    style_mean, style_std = cal_mean_std(style_feat)
    content_mean, content_std = cal_mean_std(content_feat)

    normalized_content_feat = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)

    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


def cal_mean_std(feat, eps=1e-5):
    #[batch size, channels, height, width]
    size = feat.size()
    assert len(size) == 4

    batch_size, channels = size[:2]

    feat_mean = (
        feat.view(batch_size, channels, -1)
        .mean(dim=2)
        .view(batch_size, channels, 1, 1)
    )

    feat_var = (
        feat.view(batch_size, channels, -1)
        .var(dim=2, unbiased=False) + eps
    )

    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std