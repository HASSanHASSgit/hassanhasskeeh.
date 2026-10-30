"""Cinematic render: timeline, breathing, camera, grade, grain, encode."""
import os
import sys
import math
import subprocess
import numpy as np
import cv2
import imageio_ffmpeg

from motion import Shot, W, H

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 24
DUR = 10.0
N = int(FPS * DUR)


# --------------------------------------------------------------------- easing
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_io(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)


def blink_curve(t, start, close=0.16, hold=0.07, open_=0.26):
    """Return blink phase 0(open)..1(shut)..0(open) for a blink starting at `start`."""
    dt = t - start
    if dt <= 0 or dt >= close + hold + open_:
        return 0.0
    if dt < close:
        return ease_io(dt / close)
    dt -= close
    if dt < hold:
        return 1.0
    dt -= hold
    return 1.0 - ease_io(dt / open_)


# --------------------------------------------------------------------- timeline
TURN_IN, TURN_FULL = 1.55, 4.55
BACK_IN, BACK_FULL = 7.10, 9.30


def timeline(t):
    # head turn amount
    if t < TURN_IN:
        u = 0.0
    elif t < TURN_FULL:
        u = smoother((t - TURN_IN) / (TURN_FULL - TURN_IN))
    elif t < BACK_IN:
        u = 1.0
    else:
        u = 1.0 - 0.80 * smoother(min(1.0, (t - BACK_IN) / (BACK_FULL - BACK_IN)))
    # blinks (two, one facing camera, one facing the light)
    bl = max(blink_curve(t, 0.85),
             blink_curve(t, 5.35))
    blink_a = bl if u < 0.5 else 0.0
    blink_b = bl if u >= 0.5 else 0.0
    # smooth hand-off of blink between poses to avoid pops
    return u, blink_a, blink_b


# --------------------------------------------------------------------- output geometry
OW, OH = 1920, 1080
SCALE = 1.0
SCALED_H = 0
CROP_Y = 0
VIG_W = None


def set_output(w, h, crop_bias=0.55):
    """Configure the finishing resolution (16:9 crop out of the 3:2 source)."""
    global OW, OH, SCALE, SCALED_H, CROP_Y, VIG_W
    OW, OH = int(w), int(h)
    SCALE = OW / W
    SCALED_H = int(round(H * SCALE))
    CROP_Y = int(round((SCALED_H - OH) * crop_bias))
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    nx, ny = xx / OW, yy / OH
    vig = 1.0 - 0.34 * np.clip(np.sqrt(((nx - 0.5) * 1.22) ** 2 + ((ny - 0.53) * 1.05) ** 2) / 0.74, 0, 1) ** 2.2
    VIG_W = vig.astype(np.float32)[..., None]
    return OW, OH


set_output(1920, 1080)

# --------------------------------------------------------------------- fx assets
rng0 = np.random.default_rng(7)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
NX, NY = XX / W, YY / H

VIGNETTE = 1.0 - 0.42 * np.clip(
    np.sqrt(((NX - 0.5) * 1.18) ** 2 + ((NY - 0.52) * 1.02) ** 2) / 0.72, 0, 1) ** 2.1
VIGNETTE = VIGNETTE.astype(np.float32)[..., None]

# low-frequency haze fields
def noise_field(seed, scale):
    small = rng0.normal(0.5, 0.18, (max(2, int(scale)), max(2, int(scale * W / H)))).astype(np.float32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


HAZE1 = cv2.GaussianBlur(noise_field(1, 9), (0, 0), 40)
HAZE2 = cv2.GaussianBlur(noise_field(2, 14), (0, 0), 70)
HAZE1 = (HAZE1 - HAZE1.min()) / (HAZE1.max() - HAZE1.min() + 1e-6)
HAZE2 = (HAZE2 - HAZE2.min()) / (HAZE2.max() - HAZE2.min() + 1e-6)

# warm/cool split for atmosphere
WARM_SIDE = cv2.GaussianBlur((NX ** 1.6).astype(np.float32), (0, 0), 60)
COOL_SIDE = 1.0 - WARM_SIDE

# anamorphic streak around the warm key highlight
STREAK = None


def build_streak(hotspot=(0.68, 0.09)):
    y0 = int(hotspot[1] * H)
    core = np.zeros((H, W), np.float32)
    cv2.circle(core, (int(hotspot[0] * W), y0), 9, 1.0, -1)
    streak = cv2.GaussianBlur(core, (0, 0), sigmaX=180, sigmaY=7)
    streak /= streak.max() + 1e-6
    streak = streak * 0.9 + cv2.GaussianBlur(core, (0, 0), 26) * 1.4
    return streak[..., None]


STREAK = build_streak()

# camera handheld drift (sum of incommensurate sines + slow noise)
def handheld(t):
    dx = 0.9 * math.sin(2 * math.pi * 0.13 * t + 0.6) + 0.55 * math.sin(2 * math.pi * 0.29 * t + 2.1)
    dy = 0.8 * math.sin(2 * math.pi * 0.11 * t + 1.9) + 0.5 * math.sin(2 * math.pi * 0.23 * t + 0.3)
    rot = 0.055 * math.sin(2 * math.pi * 0.07 * t + 1.1)
    return dx, dy, rot


def camera_matrix(t, scale_extra=1.0):
    s = 1.0 + 0.052 * smoother(min(1.0, t / DUR)) + 0.010 * (t / DUR)
    dx, dy, rot = handheld(t)
    M = cv2.getRotationMatrix2D((W / 2, H * 0.46), rot, s * scale_extra)
    M[0, 2] += dx
    M[1, 2] += dy
    return M.astype(np.float32)


def breathing_field(t, amp=2.6):
    """Vertical chest/shoulder rise, strongest at the bottom of frame."""
    ph = 2 * math.pi * t / 4.15
    w = np.clip((YY - 560) / (H - 560), 0, 1) ** 1.8
    dy = -amp * math.sin(ph) * w
    dx = 0.9 * math.sin(ph + 0.7) * w
    return dx.astype(np.float32), dy.astype(np.float32)


# --------------------------------------------------------------------- grade
def grade(img):
    """Film-style grade on float32 BGR 0..1."""
    # gentle filmic S-curve
    x = np.clip(img * 1.045, 0, 1)
    s = x * x * (3 - 2 * x)
    img = x * 0.76 + s * 0.24
    # lift shadows toward teal, push highlights warm
    lum = img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299
    lum = lum[..., None]
    shadow = np.clip(1.0 - lum * 2.4, 0, 1)
    high = np.clip((lum - 0.45) * 2.0, 0, 1)
    img = img + shadow * np.array([0.030, 0.012, -0.008], np.float32)   # BGR: cool shadows
    img = img + high * np.array([-0.010, 0.004, 0.022], np.float32)     # warm highlights
    # saturation
    lum2 = img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299
    img = lum2[..., None] + (img - lum2[..., None]) * 1.02
    return np.clip(img, 0, 1)


def halation(img, strength=0.16):
    lum = (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)[..., None]
    hi = np.clip((lum - 0.66) / 0.34, 0, 1) * np.clip(img, 0, 1)
    glow = cv2.GaussianBlur(hi, (0, 0), 16)
    glow2 = cv2.GaussianBlur(hi, (0, 0), 48)
    tint = np.array([0.55, 0.80, 1.0], np.float32)   # BGR -> warm
    return np.clip(img + (glow * 0.75 + glow2 * 0.55) * tint * strength, 0, 1)


def atmosphere(img, t):
    """Drifting air / haze in the shadows, warm right / cool left."""
    lum = (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)[..., None]
    air_mask = np.clip(1.0 - lum * 3.2, 0, 1) ** 1.4
    scroll1 = cv2.warpAffine(HAZE1, np.float32([[1, 0, 26 * t], [0, 1, 9 * t]]), (W, H),
                             borderMode=cv2.BORDER_REFLECT)
    scroll2 = cv2.warpAffine(HAZE2, np.float32([[1, 0, -15 * t], [0, 1, -6 * t]]), (W, H),
                             borderMode=cv2.BORDER_REFLECT)
    breath = 0.55 + 0.45 * math.sin(2 * math.pi * t / 6.2 - 0.8)
    haze = (scroll1 * 0.55 + scroll2 * 0.45) * breath
    warm = WARM_SIDE[..., None] * np.array([0.30, 0.62, 1.0], np.float32)
    cool = COOL_SIDE[..., None] * np.array([1.0, 0.72, 0.42], np.float32)
    tint = warm * 0.55 + cool * 0.45
    return np.clip(img + (haze[..., None] * air_mask * tint) * 0.085, 0, 1)


def light_flicker(img, t):
    """Subtle pulsing of the rim light + anamorphic streak."""
    k = 0.030 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 5.4 + 1.2))
    k2 = 0.022 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 3.1 + 2.6))
    streak = STREAK * np.array([0.45, 0.78, 1.0], np.float32)
    lum = (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)[..., None]
    warm_beams = np.clip(img - 0.45, 0, 1) * WARM_SIDE[..., None]
    out = img + warm_beams * k
    out = out + streak * (k2 * 1.6) * np.clip(1.0 - lum * 0.4, 0, 1)
    return np.clip(out, 0, 1)


def add_grain(img, frame_idx):
    """Fine 35mm-style grain, slightly stronger in the shadows."""
    rng = np.random.default_rng(5000 + frame_idx * 13)
    h, w = img.shape[:2]
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.62)
    n /= (n.std() + 1e-6)
    lum = (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)
    amp = 0.0075 + 0.0115 * np.clip(1.0 - lum, 0, 1) ** 1.3
    grain = n[..., None] * amp[..., None]
    ch = np.stack([cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.6)
                   for _ in range(3)], -1)
    ch /= (ch.std() + 1e-6)
    return np.clip(img + grain + ch * 0.0022, 0, 1)


# --------------------------------------------------------------------- render
def render_frame(shot, t, idx):
    u, ba, bb = timeline(t)
    img = shot.frame(u, ba, bb)                       # BGR float 0..255

    # breathing + handheld camera
    dx, dy = breathing_field(t)
    M = camera_matrix(t)
    img = cv2.remap(img, XX + dx, YY + dy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

    img = np.clip(img / 255.0, 0, 1).astype(np.float32)
    img = grade(img)
    img = halation(img, 0.14)
    img = atmosphere(img, t)
    img = light_flicker(img, t)

    # finish at 16:9 deliverable resolution
    img = cv2.resize(img, (OW, SCALED_H), interpolation=cv2.INTER_LANCZOS4)[CROP_Y:CROP_Y + OH]
    img = img * VIG_W
    img = add_grain(img, idx)

    fade = smoothstep(0.0, 0.55, t) * (1.0 - smoothstep(DUR - 0.80, DUR - 0.02, t))
    img = img * fade
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def encode(out_path, frames_iter, audio_path=None, crf=16, preset='slow'):
    cmd = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-']
    if audio_path:
        cmd += ['-i', audio_path]
    cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-x264-params', 'keyint=48:min-keyint=24:ref=4',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    if audio_path:
        cmd += ['-c:a', 'aac', '-b:a', '192k', '-shortest']
    cmd += [out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    for fr in frames_iter:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode('utf8', 'ignore')
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f'ffmpeg failed {code}: {err}')


if __name__ == '__main__':
    import time
    shot = Shot()
    out = os.path.join(HERE, 'preview_frames')
    os.makedirs(out, exist_ok=True)
    ts = [0.2, 1.0, 2.6, 3.8, 5.45, 8.5, 9.6]
    for t in ts:
        t0 = time.time()
        fr = render_frame(shot, t, int(t * FPS))
        cv2.imwrite(os.path.join(out, f'p_{t:04.2f}.png'), fr)
        print('frame', t, 'in', round(time.time() - t0, 2), 's')
