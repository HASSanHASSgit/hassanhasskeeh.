"""Render the full cinematic shot: 1080p master + 4K finish, with score."""
import os
import sys
import cv2
import render as R
from motion import Shot

HERE = os.path.dirname(os.path.abspath(__file__))


def frames(shot):
    for i in range(R.N):
        yield R.render_frame(shot, i / R.FPS, i)


def main():
    shot = Shot()
    audio = os.path.join(HERE, 'score.wav')
    audio = audio if os.path.exists(audio) else None

    jobs = [
        # path, width, height, crf, preset
        (os.path.join(HERE, 'cinematic_shot_1080p.mp4'), 1920, 1080, 18, 'slow'),
        (os.path.join(HERE, 'cinematic_shot_4k.mp4'), 3840, 2160, 21, 'medium'),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for path, w, h, crf, preset in jobs:
        if only and only not in path:
            continue
        R.set_output(w, h)
        R.encode(path, frames(shot), audio_path=audio, crf=crf, preset=preset)
        print('wrote', path, round(os.path.getsize(path) / 1e6, 1), 'MB', flush=True)


if __name__ == '__main__':
    main()
