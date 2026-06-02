"""Generate a bright, dense, high-contrast colour-checker PNG for the UE platform
material. Vivid distinct cells + thin dark grid lines = strong, distinct,
view-independent features for the splat to lock geometry. Committed so the UE
capture is reproducible."""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "ue_capture" / "assets" / "floor_tex.png"
N = 8        # cells per side (tiled further by the material's TexCoord)
S = 512
cell = S // N
# bright, saturated, maximally-distinct palette (sRGB 0-255)
PALETTE = [
    (235, 70, 55), (255, 145, 35), (250, 225, 45), (95, 205, 80),
    (45, 205, 205), (70, 120, 240), (160, 90, 225), (240, 80, 185),
    (245, 245, 248), (130, 235, 130), (240, 170, 205), (175, 130, 70),
]

img = np.zeros((S, S, 3), np.uint8)
for i in range(N):
    for j in range(N):
        c = PALETTE[(i * 5 + j * 3) % len(PALETTE)]   # decorrelate neighbours
        img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = c
# thin dark grid lines for extra high-frequency edges
g = max(S // 128, 2)
for k in range(N + 1):
    p = min(k * cell, S - 1)
    img[max(p - g // 2, 0):p + g // 2 + 1, :] = (25, 25, 30)
    img[:, max(p - g // 2, 0):p + g // 2 + 1] = (25, 25, 30)

OUT.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(img).save(OUT)
print("wrote", OUT, img.shape)
