"""Drawings for the demo and the papers: crossing frames, animations and timelines."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .controller import EpisodeResult
from .perception import Observation
from .sim import Episode, MOBILITY_AID_CLASSES

# Colour theme (matches the project documents)
INK = (43, 38, 34)
SAND = (244, 236, 223)
TERRA = (185, 72, 42)
AMBER = (227, 161, 58)
GREEN = (60, 122, 78)
ROAD = (74, 71, 66)
CLASS_COLOURS = {"pedestrian": (70, 140, 90), "wheelchair": TERRA, "rollator": (196, 98, 45),
                 "crutches": (150, 60, 110), "cane": (120, 80, 160)}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_frame(ep: Episode, k: int, observations: list[list[Observation]] | None = None,
               result: EpisodeResult | None = None, title: str = "", W: int = 720, H: int = 440) -> Image.Image:
    """Top-down view of the crossing at time step k.

    Filled circle  = real person (colour = class)
    Amber ring     = the camera detected this person
    Red dashed X   = real person on the crossing that the camera MISSED
    Purple square  = phantom detection (false positive)
    """
    img = Image.new("RGB", (W, H), SAND)
    d = ImageDraw.Draw(img)
    f_small, f_big = _font(13), _font(17)
    top, bottom = 60, H - 44                                   # road band (y = 0 at bottom kerb)
    left, right = 140, W - 140
    d.rectangle([0, top, W, bottom], fill=ROAD)
    d.rectangle([0, top - 14, W, top], fill=(200, 190, 175))   # far kerb
    d.rectangle([0, bottom, W, bottom + 14], fill=(200, 190, 175))  # near kerb

    def to_px(x_m, y_m):
        px = left + (x_m / ep.width) * (right - left)
        py = bottom - (y_m / ep.length) * (bottom - top)
        return px, py

    stripe_h = (bottom - top) / 13
    for s in range(13):
        if s % 2 == 0:
            d.rectangle([left, top + s * stripe_h, right, top + (s + 1) * stripe_h], fill=(232, 228, 220))

    t = float(ep.times[k])
    for i, p in enumerate(ep.pedestrians):
        y = ep.positions[i, k]
        if y < -1.0 or y > ep.length + 1.2:
            continue
        px, py = to_px(p.x, min(max(y, -0.9), ep.length + 1.0))
        c = CLASS_COLOURS.get(p.cls, (70, 140, 90))
        r = 11 if p.cls in MOBILITY_AID_CLASSES else 8
        d.ellipse([px - r, py - r, px + r, py + r], fill=c, outline=INK)
        d.text((px + r + 3, py - 8), f"P{p.pid}", fill=INK, font=f_small)

    if observations is not None:
        seen = {o.gt_pid for o in observations[k]}
        for o in observations[k]:
            px, py = to_px(o.x, o.y)
            if o.gt_pid < 0:
                d.rectangle([px - 10, py - 10, px + 10, py + 10], outline=(130, 70, 170), width=3)
            else:
                d.ellipse([px - 16, py - 16, px + 16, py + 16], outline=AMBER, width=3)
        for i, p in enumerate(ep.pedestrians):
            if p.pid not in seen and ep.on_crossing(i, k):
                px, py = to_px(p.x, ep.positions[i, k])
                for dx in range(-14, 15, 7):
                    d.line([px + dx, py - 14, px + dx + 4, py - 10], fill=(220, 40, 40), width=2)
                d.line([px - 12, py - 12, px + 12, py + 12], fill=(220, 40, 40), width=3)
                d.line([px - 12, py + 12, px + 12, py - 12], fill=(220, 40, 40), width=3)

    # signal head
    walk = result is None or t < result.end_time
    d.rounded_rectangle([18, 120, 98, 290], radius=12, fill=INK)
    d.ellipse([36, 135, 80, 179], fill=(90, 40, 30) if walk else (220, 60, 40))
    d.ellipse([36, 230, 80, 274], fill=(95, 200, 120) if walk else (40, 70, 50))
    d.text((18, 296), "WALK" if walk else "DON'T WALK", fill=GREEN if walk else TERRA, font=f_small)
    if result is not None:
        state = result.occupancy[k] if k < len(result.occupancy) else "ended"
        d.rounded_rectangle([W - 128, 120, W - 12, 178], radius=8, fill=SAND)
        d.text((W - 120, 126), "occupancy", fill=INK, font=f_small)
        d.text((W - 120, 146), state, fill=TERRA if state == "uncertain" else INK, font=f_big)
    d.text((14, 12), title, fill=INK, font=f_big)
    d.text((14, 34), f"t = {t:5.1f} s", fill=INK, font=f_small)
    legend = "filled = person   amber ring = detected   red X = missed   purple square = phantom"
    d.text((14, H - 22), legend, fill=INK, font=f_small)
    return img


def make_gif(ep: Episode, observations, result: EpisodeResult, path: str, title: str = "",
             every_s: float = 0.4, extra_s: float = 4.0) -> str:
    stop = min(len(ep.times) - 1, int((result.end_time + extra_s) / ep.dt))
    step = max(1, int(every_s / ep.dt))
    frames = [draw_frame(ep, k, observations, result, title) for k in range(0, stop + 1, step)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(every_s * 1000 / 2), loop=0)
    return path


def timeline_figure(ep: Episode, results: list[EpisodeResult], title: str = ""):
    """Bars = time each person spends on the crossing. Vertical lines = when each controller ended the phase."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ep.pedestrians) + 1.8), dpi=110)
    for i, p in enumerate(ep.pedestrians):
        t0, t1 = p.t_start, ep.exit_time(i)
        c = np.array(CLASS_COLOURS.get(p.cls, (70, 140, 90))) / 255
        ax.barh(i, t1 - t0, left=t0, color=c, height=0.5)
        ax.text(t1 + 0.3, i, f"P{p.pid} {p.cls}", va="center", fontsize=8)
    styles = {"baseline": ("#B9482A", "--"), "uncertainty_aware": ("#3C7A4E", "-")}
    for r in results:
        col, ls = styles.get(r.controller, ("k", ":"))
        ax.axvline(r.end_time, color=col, ls=ls, lw=2, label=f"{r.controller} ends at {r.end_time:.1f}s")
        for i, p in enumerate(ep.pedestrians):
            if p.t_start < r.end_time and ep.exit_time(i) > r.end_time:
                ax.plot(r.end_time, i, marker="x", color=col, ms=10, mew=3)
    ax.set_yticks([])
    ax.set_xlabel("time since walk phase started (s)")
    ax.set_title(title or "Who was still crossing when the phase ended?  (x = cut off)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8, frameon=False)
    ax.set_facecolor("#FBF7F0")
    fig.tight_layout()
    return fig


def fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
