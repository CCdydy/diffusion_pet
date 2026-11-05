"""Remove the studio background from extracted H3 frames and normalize their canvas."""
import json
import sys
import os
from pathlib import Path
os.environ['U2NET_HOME'] = os.path.expanduser('~/.cache/rembg')
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import new_session, remove

ROOT = Path(sys.argv[1]).resolve()
for folder in ['cutouts', 'normalized', 'masks']:
    (ROOT / folder).mkdir(exist_ok=True)
session = new_session('u2net', providers=['CPUExecutionProvider'])
metadata = []
previews = []
for path in sorted((ROOT / 'frames').glob('*.png')):
    image = Image.open(path).convert('RGB')
    cutout = remove(image, session=session)
    alpha = np.array(cutout.getchannel('A'))
    mask = alpha > 127
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f'Empty foreground: {path}')
    bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cutout.save(ROOT / 'cutouts' / path.name)
    cutout.getchannel('A').save(ROOT / 'masks' / path.name)
    crop = cutout.crop(bounds)
    size = 640
    scale = 0.84 * size / max(crop.size)
    crop = crop.resize(tuple(round(x * scale) for x in crop.size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    position = ((size - crop.width) // 2, (size - crop.height) // 2)
    canvas.paste(crop, position, crop.getchannel('A'))
    canvas.save(ROOT / 'normalized' / path.name)
    preview = Image.new('RGB', image.size, '#72a5ac')
    preview.paste(cutout, mask=cutout.getchannel('A'))
    previews.append((path.name, preview))
    metadata.append({'file': path.name, 'bbox_xyxy': bounds, 'foreground_pixels': int(mask.sum()), 'canvas_scale': scale})
    print(path.name, bounds, flush=True)
sheet = Image.new('RGB', (1728, 540), '#eeeeee')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
for i, (name, im) in enumerate(previews):
    x, y = (i % 4) * 432, (i // 4) * 270
    sheet.paste(im.resize((432, 240)), (x, y + 30))
    draw.text((x + 8, y + 5), name, font=font, fill='black')
sheet.save(ROOT / 'cutout_quality.jpg', quality=95)
(ROOT / 'preprocessing.json').write_text(json.dumps(metadata, indent=2))
print('Prepared', len(metadata), 'views.', flush=True)
