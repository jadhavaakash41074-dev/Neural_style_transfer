import os, shutil
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
root = r'A:\Downloads\Adain_NST_project'
out = os.path.join(root, 'small')
exts = ('.jpg', '.jpeg', '.png', '.bmp')

for name in ['content_data', 'style_data']:
    src = os.path.join(root, name)
    dst = os.path.join(out, name)
    os.makedirs(dst, exist_ok=True)
    files = [os.path.join(r, f) for r, _, fs in os.walk(src)
             for f in fs if f.lower().endswith(exts)]
    for i, p in enumerate(tqdm(files, desc=name)):
        try:
            with Image.open(p) as im:
                w, h = im.size
                if w * h > 50_000_000:
                    continue
                im = im.convert('RGB')
                s = 512 / min(w, h)
                if s < 1:
                    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
                im.save(os.path.join(dst, f'{i:06d}.jpg'), quality=92)
        except Exception:
            continue

shutil.make_archive(os.path.join(root, 'adain_data'), 'zip', out)
print('zip ready: adain_data.zip')