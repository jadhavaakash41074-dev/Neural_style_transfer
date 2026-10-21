import json, os, random, shutil
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

root = r'A:\Downloads\Adain_NST_project'
random.seed(5)
N = 3000          # kitni images chahiye
SAMPLE = 9000     # itni random images check karenge (jaldi hone ke liye)

def stats(path):
    im = Image.open(path).convert('L')
    im.draft('L', (256, 256))
    im = im.resize((128, 128))
    a = np.asarray(im, dtype=np.float32)
    brightness = a.mean()                                   # bahut kam = dark
    contrast = a.std()                                      # bahut kam = flat/sky jaisa
    edges = np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32).mean()  # bahut zyada = busy
    return brightness, contrast, edges

def ok(kind, b, c, e):
    if b < 50 or b > 215:   return False   # bahut dark / bahut bright
    if c < 25:              return False   # flat, detail nahi
    if kind == 'content':
        return e > 6                       # content mein kuch detail honi chahiye
    return 6 < e < 28                      # style: na flat, na heavy engraving jaisa busy

for kind in ('content', 'style'):
    files = json.load(open(f'{root}\\{kind}_data\\_manifest.json'))
    random.shuffle(files)
    out = f'{root}\\best\\{kind}_data'
    os.makedirs(out, exist_ok=True)
    kept = 0
    for p in tqdm(files[:SAMPLE], desc=kind):
        try:
            if ok(kind, *stats(p)):
                shutil.copy(p, out)
                kept += 1
                if kept >= N:
                    break
        except Exception:
            continue
    print(kind, 'kept:', kept)