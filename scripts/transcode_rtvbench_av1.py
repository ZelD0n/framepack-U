#!/usr/bin/env python3
"""Transcode known RTV-Bench AV1 videos to H264 in place.

The videos below cannot be decoded by the current OpenCV/FFmpeg build used
by offline CLIP extraction. This script keeps a .av1_backup.mp4 copy, writes a
temporary H264 mp4, verifies that OpenCV can read the first frame, then replaces
the original path so existing RTV-Bench metadata keeps working.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path('/gaoshuo-sh/benchmarks/RTV-Bench')
TARGETS = (
    Path('videos/sports_cam/SportsCam_004.mp4'),
    Path('videos/sports_cam/SportsCam_017.mp4'),
    Path('videos/sports_cam/SportsCam_031.mp4'),
    Path('videos/sports_cam/SportsCam_039.mp4'),
    Path('videos/sports_cam/SportsCam_055.mp4'),
    Path('videos/sports_cam/SportsCam_056.mp4'),
    Path('videos/sports_cam/SportsCam_067.mp4'),
    Path('videos/sports_cam/SportsCam_076.mp4'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_ROOT), help='RTV-Bench root directory')
    parser.add_argument('--ffmpeg', default=None, help='Path to ffmpeg binary; defaults to PATH lookup')
    parser.add_argument('--crf', type=int, default=18, help='H264 quality; lower is better/larger')
    parser.add_argument('--preset', default='medium', help='x264 preset, e.g. fast, medium, slow')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing files')
    parser.add_argument('--force', action='store_true', help='Overwrite existing tmp output if present')
    return parser.parse_args()


def find_ffmpeg(path: str | None) -> str:
    if path:
        if '/' not in path:
            ffmpeg = shutil.which(path)
            if ffmpeg:
                return ffmpeg
            raise FileNotFoundError(
                f'ffmpeg command not found in PATH: {path}. '
                'Pass an absolute path with --ffmpeg /path/to/ffmpeg, '
                'or install/load ffmpeg first.'
            )
        if not Path(path).exists():
            raise FileNotFoundError(f'ffmpeg not found: {path}')
        return path

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    candidates = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.append(Path(conda_prefix) / 'bin' / 'ffmpeg')
    candidates.extend(
        [
            Path('/usr/bin/ffmpeg'),
            Path('/usr/local/bin/ffmpeg'),
            Path('/opt/conda/bin/ffmpeg'),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    try:
        import imageio_ffmpeg

        imageio_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_ffmpeg_path and Path(imageio_ffmpeg_path).exists():
            return imageio_ffmpeg_path
    except Exception:
        pass

    raise FileNotFoundError(
        'ffmpeg is not available. Install or load ffmpeg first, then rerun. '
        'Examples: `conda install -c conda-forge ffmpeg`, `module load ffmpeg`, '
        'or `FFMPEG=/path/to/ffmpeg sh run_rtv_clip.sh`. '
        'If the two RTV videos were already transcoded on another machine, run with `RTV_TRANSCODE=0`.'
    )


def verify_cv2_readable(path: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'OpenCV could not open {path}')
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'OpenCV could not read first frame from {path}')
    finally:
        cap.release()


def is_cv2_readable(path: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def transcode_one(ffmpeg: str, src: Path, crf: int, preset: str, dry_run: bool, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f'Missing source video: {src}')

    backup = src.with_suffix('.av1_backup.mp4')
    tmp = src.with_suffix('.h264_tmp.mp4')

    if is_cv2_readable(src):
        print(f'\nReadable by OpenCV, skip: {src}')
        return

    if tmp.exists():
        if force:
            tmp.unlink()
        else:
            raise FileExistsError(f'Temporary output already exists: {tmp}. Pass --force to replace it.')

    cmd = [
        ffmpeg,
        '-hide_banner',
        '-y',
        '-i',
        str(src),
        '-map',
        '0:v:0',
        '-map',
        '0:a?',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-preset',
        preset,
        '-crf',
        str(crf),
        '-c:a',
        'aac',
        '-movflags',
        '+faststart',
        str(tmp),
    ]

    print(f'\nSource : {src}')
    print(f'Backup : {backup}')
    print(f'Tmp    : {tmp}')
    print('Command:', ' '.join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, check=True)
    verify_cv2_readable(tmp)

    if not backup.exists():
        src.rename(backup)
    else:
        src.unlink()
    tmp.rename(src)
    verify_cv2_readable(src)
    print(f'Done: {src}')


def main() -> None:
    args = parse_args()
    if args.dry_run and args.ffmpeg is None and shutil.which('ffmpeg') is None:
        ffmpeg = 'ffmpeg'
    else:
        ffmpeg = find_ffmpeg(args.ffmpeg)
    root = Path(args.root)

    for rel in TARGETS:
        transcode_one(
            ffmpeg=ffmpeg,
            src=root / rel,
            crf=args.crf,
            preset=args.preset,
            dry_run=args.dry_run,
            force=args.force,
        )


if __name__ == '__main__':
    main()
