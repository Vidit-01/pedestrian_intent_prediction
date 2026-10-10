"""Download JAAD: annotations (git clone of the official repo) and, optionally, the video clips.

Model-agnostic: every model in this repo reads from the JAAD root this script creates.

    python download_jaad.py --root /kaggle/working/JAAD            # annotations only (enough for Baseline A, kinematics)
    python download_jaad.py --root /kaggle/working/JAAD --clips_from /kaggle/input/jaad-video/JAAD_clips
        # preferred on Kaggle: attach dataset `christopheronyiuke/jaad-video` (346 official mp4s), symlinked, no download
    python download_jaad.py --root /kaggle/working/JAAD --clips    # fallback: download the official zip (~3.1 GB) from York

Layout after running (this is what jaad_data.JAAD expects):
    <root>/annotations*/  <root>/split_ids/  <root>/jaad_data.py  <root>/JAAD_clips/video_XXXX.mp4
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

JAAD_REPO = 'https://github.com/ykotseruba/JAAD.git'
JAAD_CLIPS_URL = 'http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip'


def git_clone(url, dst):
    if os.path.isdir(os.path.join(dst, '.git')):
        print(f'[skip] {dst} already cloned')
        return
    subprocess.run(['git', 'clone', '--depth', '1', url, dst], check=True)


def download_file(url, dst, chunk=1 << 20):
    if os.path.exists(dst):
        print(f'[skip] {dst} exists')
        return
    tmp = dst + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0))
        done = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total and done % (200 * chunk) < chunk:
                print(f'  {done / 1e9:.2f} / {total / 1e9:.2f} GB', flush=True)
    os.replace(tmp, dst)


def unzip(path, dst):
    with zipfile.ZipFile(path) as z:
        z.extractall(dst)


def download_jaad(root, clips=False):
    git_clone(JAAD_REPO, root)
    if clips:
        clips_dir = os.path.join(root, 'JAAD_clips')
        n = len([f for f in os.listdir(clips_dir) if f.endswith('.mp4')]) if os.path.isdir(clips_dir) else 0
        if n == 346:
            print(f'[skip] {clips_dir} has all 346 clips')
        else:
            zip_path = os.path.join(root, 'JAAD_clips.zip')
            download_file(JAAD_CLIPS_URL, zip_path)
            unzip(zip_path, root)
            os.remove(zip_path)
            n = len([f for f in os.listdir(clips_dir) if f.endswith('.mp4')])
            print(f'{n} clips in {clips_dir}')


def link_clips(src_dir, root):
    """Use clips from an attached Kaggle dataset instead of downloading (symlinks, no copy)."""
    dst = os.path.join(root, 'JAAD_clips')
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.endswith('.mp4') and not os.path.exists(os.path.join(dst, f)):
            os.symlink(os.path.join(src_dir, f), os.path.join(dst, f))
    n = len([f for f in os.listdir(dst) if f.endswith('.mp4')])
    print(f'linked {n} clips into {dst}' + ('' if n == 346 else '  WARNING: JAAD has 346 clips'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='JAAD root directory to create')
    ap.add_argument('--clips', action='store_true', help='also download the mp4 clips (~3.1 GB)')
    ap.add_argument('--clips_from', default=None, help='dir with video_XXXX.mp4 (e.g. a Kaggle input dataset); symlinked instead of downloading')
    args = ap.parse_args()
    if shutil.which('git') is None:
        sys.exit('git not found')
    download_jaad(args.root, clips=args.clips and args.clips_from is None)
    if args.clips_from:
        link_clips(args.clips_from, args.root)
