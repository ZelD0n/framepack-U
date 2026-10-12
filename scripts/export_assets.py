"""Export source figures for the static project page.

Usage: python scripts/export_assets.py /path/to/FramePack-U
Dependencies: pymupdf, pillow (only needed to regenerate images).
"""
from pathlib import Path
import argparse
import io
import shutil
import pymupdf
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
args = parser.parse_args()
source = args.source
output = Path(__file__).resolve().parents[1] / 'assets'
output.mkdir(exist_ok=True)


def save_webp(image, name, max_width=2400):
    image = image.convert('RGB')
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    image.save(output / name, 'WEBP', quality=92, method=6)
    print(f'{name}: {image.width} × {image.height}')


def render_pdf(filename, name, clip=None, width=2400):
    with pymupdf.open(source / filename) as doc:
        page = doc[0]
        rect = pymupdf.Rect(clip) if clip else page.rect
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(width / rect.width, width / rect.width), clip=rect, alpha=False)
        save_webp(Image.open(io.BytesIO(pixmap.tobytes('png'))), name, width)


logo = Image.open(source / 'assets/logo2.png')
save_webp(logo, 'logo.webp', 1672)
# Use only the illustration above the embedded title, avoiding duplicate typography.
scale_x, scale_y = logo.width / 1672, logo.height / 941
illustration = logo.crop(tuple(round(value * scale) for value, scale in zip((150, 0, 1415, 672), (scale_x, scale_y, scale_x, scale_y))))
save_webp(illustration, 'elephant.webp', 1265)
save_webp(Image.open(source / 'assets/framepack_framework.png'), 'framework.webp')
render_pdf('framepack_vis.pdf', 'quality-phone.webp', clip=(0, 60, 1783.92, 632))
render_pdf('framepack_vis.pdf', 'quality-opening.webp', clip=(0, 640, 1783.92, 1301.04))
for case in (4, 6, 9):
    render_pdf(f'longvideobench_sample{case}_framepack_vs_uniform.pdf', f'tsne-{case}.webp', width=2800)
render_pdf('memory_qwen35_9b_lines_raw.pdf', 'memory.webp')
render_pdf('latency_qwen35_9b_lines_raw.pdf', 'latency.webp')
shutil.copy2(source / 'FramePack_for_VLM_NeurIPS_2026___arXiv_version.pdf', output / 'paper.pdf')
