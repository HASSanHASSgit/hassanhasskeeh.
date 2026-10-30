"""Motion stage: optical-flow morphing between keyframes + breathing + camera.

Produces a full-resolution float/uint8 frame stack for the cinematic shot.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1536, 1024
EYE_LEFT = (560, 295, 250, 105)   # x, y, w, h
EYE_RIGHT = (935, 275, 255, 105)


# ----------------------------------------------------------------------------- utils
def load(name):
    img = cv2.imread(os.path.join(HERE, name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(name)
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    return img


def gray(img, sigma=1.0):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    return g.astype(np.float32)


def flow(a, b, sigma=1.1, levels=5, winsize=25):
    return cv2.calcOpticalFlowFarneback(
        gray(a, sigma), gray(b, sigma), None,
        pyr_scale=0.5, levels=levels, winsize=winsize,
        iterations=4, poly_n=7, poly_sigma=1.5, flags=0)


def smooth_flow(fl, sigma, edge_hold=0.35):
    """Smooth the flow field but keep some edge fidelity."""
    s = np.dstack([cv2.GaussianBlur(fl[..., i], (0, 0), sigma) for i in range(2)])
    if edge_hold > 0:
        s = s * (1.0 - edge_hold) + fl * edge_hold
    return s.astype(np.float32)


GRID = None


def grid():
    global GRID
    if GRID is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        GRID = (xx, yy)
    return GRID


def warp(img, fl, extra=None):
    xx, yy = grid()
    mx, my = xx + fl[..., 0], yy + fl[..., 1]
    if extra is not None:
        mx = mx + extra[0]
        my = my + extra[1]
    return cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)


def align_to(ref, img):
    """Affine-align img to ref using ECC on luminance."""
    g1 = gray(ref, 2.0) / 255.0
    g2 = gray(img, 2.0) / 255.0
    warp_m = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp_m = cv2.findTransformECC(
            g1, g2, warp_m, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
    except cv2.error:
        cc, warp_m = 0.0, np.eye(2, 3, dtype=np.float32)
    out = cv2.warpAffine(img, warp_m, (W, H),
                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out, cc, warp_m


class Pose:
    """Three blink states (open / half / closed) at one head pose."""

    def __init__(self, open_, half, closed):
        self.open = open_
        self.half = half
        self.closed = closed
        self.f_oh = smooth_flow(flow(open_, half, 1.0), 5.0)
        self.f_ho = smooth_flow(flow(half, open_, 1.0), 5.0)
        self.f_hc = smooth_flow(flow(half, closed, 1.0), 5.0)
        self.f_ch = smooth_flow(flow(closed, half, 1.0), 5.0)

    def at(self, b):
        """b: 0 open, 0.5 half, 1 fully closed. Returns warped-blend frame."""
        b = float(np.clip(b, 0.0, 1.0))
        if b <= 0.5:
            t = b / 0.5
            if t <= 0.0:
                return self.open.astype(np.float32)
            A = warp(self.open, self.f_oh * t)
            B = warp(self.half, self.f_ho * (1.0 - t))
            return cv2.addWeighted(A.astype(np.float32), 1 - t, B.astype(np.float32), t, 0)
        t = (b - 0.5) / 0.5
        if t <= 0.0:
            return self.half.astype(np.float32)
        A = warp(self.half, self.f_hc * t)
        B = warp(self.closed, self.f_ch * (1.0 - t))
        return cv2.addWeighted(A.astype(np.float32), 1 - t, B.astype(np.float32), t, 0)


class Shot:
    def __init__(self, verbose=True):
        base = load('keyframe_01_base.png')
        half = load('keyframe_04_halfblink.png')
        closed = load('keyframe_03_blink.png')

        turn = load('keyframe_02_turn.png')
        turn_half = load('keyframe_06_turn_half.png')
        turn_closed = load('keyframe_05_turn_closed.png')

        # Align the whole turned family into the base framing with one affine.
        turn_a, cc, _ = align_to(base, turn)
        # blend the affine toward identity a little for natural framing
        t_half_a = cv2.warpAffine(turn_half, np.eye(2, 3, dtype=np.float32), (W, H))
        # reuse the same transform as 'turn' by re-aligning: cheap trick ->
        # warp turn_half/turn_closed with transform derived from turn alignment
        # (they share the pose, so one transform is right).
        # Recompute directly:
        g1 = gray(base, 2.0) / 255.0
        g2 = gray(turn, 2.0) / 255.0
        wm = np.eye(2, 3, dtype=np.float32)
        try:
            _, wm = cv2.findTransformECC(
                g1, g2, wm, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
        except cv2.error:
            pass
        kw = dict(flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                  borderMode=cv2.BORDER_REFLECT_101)
        turn_a = cv2.warpAffine(turn, wm, (W, H), **kw)
        turn_half_a = cv2.warpAffine(turn_half, wm, (W, H), **kw)
        turn_closed_a = cv2.warpAffine(turn_closed, wm, (W, H), **kw)
        if verbose:
            print('turn alignment affine:\n', np.round(wm, 4))

        self.A = Pose(base, half, closed)
        self.B = Pose(turn_a, turn_half_a, turn_closed_a)

        self.f_ab = smooth_flow(flow(base, turn_a, 1.2), 9.0, 0.25)
        self.f_ba = smooth_flow(flow(turn_a, base, 1.2), 9.0, 0.25)

        self.base_u8 = base
        self.turn_u8 = turn_a

    # -- pose blending -------------------------------------------------------
    def frame(self, u, blink_a, blink_b):
        """u in [0,1] turn amount; blink_a / blink_b blink phase per pose."""
        a = self.A.at(blink_a) if blink_a > 0 else self.A.open.astype(np.float32)
        b = self.B.at(blink_b) if blink_b > 0 else self.B.open.astype(np.float32)
        if u <= 0.001:
            return a
        if u >= 0.999:
            return b
        uu = float(u)
        A = warp(a, self.f_ab * uu)
        B = warp(b, self.f_ba * (1.0 - uu))
        return cv2.addWeighted(A.astype(np.float32), 1 - uu, B.astype(np.float32), uu, 0)
