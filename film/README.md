# "Turn Toward the Light" — cinematic shot

A 10-second, 24 fps cinematic shot generated from a single photoreal keyframe.
Everything you see is produced by this folder's code — no video model involved.

| File | What it is |
|---|---|
| `cinematic_shot_4k.mp4` | 3840×2160 master, H.264 High, AAC 48 kHz stereo (30 MB) |
| `cinematic_shot_1080p.mp4` | 1920×1080 delivery version (18 MB) |
| `poster.jpg` | 4K hero frame from the money shot |
| `contact_sheet.jpg` | Six beats of the shot in order |
| `keyframe_0*.png` | The six generated source keyframes (3 poses × 3 blink states) |
| `score.wav` | Synthesized ambient score / air bed |
| `motion.py`, `render.py`, `audio.py`, `render_all.py` | The pipeline |

## The shot (beat sheet)

| Time | Beat |
|---|---|
| 0.00–0.55 | Fade up out of black on a held, breathing portrait |
| 0.85 | First blink — slow, 0.5 s |
| 1.55–4.55 | Posture shift: chest rises, shoulders settle, head eases into an 8° turn toward the warm key light |
| 4.55–7.10 | Hold on the light, eyes catch the rim, atmosphere drifts |
| 5.35 | Second blink, mid-turn, eyes closed as the light crosses them |
| 7.10–9.30 | He settles back toward camera, chin lifting slightly |
| 9.20–10.00 | Fade to black |

## How it is built

1. **Keyframes** — one photoreal portrait plus five consistent variants
   (head-turn, half-blink, full-blink, and the turned pose in both blink states).
2. **Optical-flow morphing** (`motion.py`) — dense Farneback flow between each pose's
   blink states and between the two head poses, with one ECC affine alignment so the
   turned frames share the base framing. Frames are produced by flow-warping both
   ends of a transition and cross-dissolving, so eyelids close and the head rotates
   smoothly instead of cross-fading.
3. **Performance layer** (`render.py`) — time-remapped easing curves for the head turn,
   an asymmetric vertical breathing field locked to the chest and shoulders, handheld
   micro-drift (dual-sine translation + roll), a slow 5% push-in.
4. **Grade & air** — filmic S-curve, teal-shadow / warm-highlight split tone, halation
   on the key-light highlight, animated low-frequency haze with a warm-right /
   cool-left tint, a subtle light flicker, anamorphic streak, vignette, fine 35 mm grain.
5. **Finishing** — 16:9 crop from the 3:2 source, Lanczos upscale, H.264.
6. **Score** (`audio.py`) — sub drone, band-passed air synced to the breath and blinks,
   soft sub pulses, gentle master fade.

## Re-render

```bash
pip install pillow numpy opencv-python-headless scipy imageio-ffmpeg
python film/render_all.py            # 1080p + 4K
python film/render_all.py 1080p      # just the 1080p
```
