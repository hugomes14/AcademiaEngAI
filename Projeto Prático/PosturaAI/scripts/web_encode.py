"""Encode the annotated video as a browser-compatible H.264 MP4."""
import argparse
from pathlib import Path
import subprocess
import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output cannot overwrite input')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(args.input),
                    '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '22',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(args.output)], check=True)


if __name__ == '__main__':
    main()
