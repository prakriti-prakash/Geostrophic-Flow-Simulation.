"""
geoflow/render.py
=================
Everything needed to turn fields and particles into a good-looking video.

The look is built from three layers:
  1. WATER SURFACE  - sea surface height is coloured with a deep-ocean palette
                      and "hill-shaded" like a real 3-D surface lit from the
                      top-left, so slopes and ripples are visible.
  2. TRACERS        - thousands of particles leave glowing trails that fade out.
                      Colour shows speed (cyan = slow, white = medium,
                      orange = fast).
  3. HUD            - title, date and a colour bar, drawn with Pillow.
"""

from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (binary_erosion, gaussian_filter, zoom)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
def make_lut(stops, n=256):
    """Build an n x 3 colour table (values 0-1) from (position, (r, g, b)) stops."""
    pos = [s[0] for s in stops]
    cols = np.array([s[1] for s in stops], dtype=float)
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], axis=1)


OCEAN_LUT = make_lut([
    (0.00, (0.010, 0.030, 0.090)),
    (0.30, (0.015, 0.110, 0.230)),
    (0.60, (0.030, 0.260, 0.400)),
    (0.85, (0.080, 0.440, 0.540)),
    (1.00, (0.300, 0.680, 0.700)),
])

TRACER_LUT = make_lut([
    (0.00, (0.25, 0.65, 1.00)),
    (0.35, (0.55, 0.92, 1.00)),
    (0.60, (1.00, 1.00, 0.95)),
    (0.80, (1.00, 0.82, 0.45)),
    (1.00, (1.00, 0.50, 0.20)),
])

LAND_COLOUR = np.array([0.085, 0.095, 0.105])
COAST_COLOUR = np.array([0.42, 0.50, 0.55])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def upsample(field, shape_px, order=3):
    """Smoothly enlarge a (ny, nx) array to the video size (H, W)."""
    zy, zx = shape_px[0] / field.shape[0], shape_px[1] / field.shape[1]
    out = zoom(field, (zy, zx), order=order, mode="nearest")
    return out[:shape_px[0], :shape_px[1]]


def fill_nan_nearest(field):
    """Replace NaN (land) with the nearest water value so smoothing behaves."""
    from scipy.ndimage import distance_transform_edt
    mask = ~np.isfinite(field)
    if not mask.any():
        return field
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def cells_to_pixels(ix, iy, cell_shape, shape_px):
    """Fractional grid indices -> pixel positions (north / +y is up)."""
    ny, nx = cell_shape
    px = ix * (shape_px[1] - 1) / (nx - 1)
    py = (ny - 1 - iy) * (shape_px[0] - 1) / (ny - 1)
    return px, py


# ---------------------------------------------------------------------------
# Layer 1: water surface
# ---------------------------------------------------------------------------
def water_surface(eta, shape_px, vmin, vmax, relief=30.0, gloss=0.35,
                  light=(-0.55, -0.65, 0.52)):
    """
    Colour + hill-shade a height field to look like a lit water surface.

    eta      : (ny, nx) heights with no NaN (use fill_nan_nearest first)
    relief   : how steep the surface looks (bigger = more dramatic)
    gloss    : strength of the bright specular highlight
    Returns float RGB (H, W, 3) in 0-1.
    """
    z = upsample(np.flipud(eta), shape_px)
    zn = (z - vmin) / (vmax - vmin)
    base = OCEAN_LUT[(np.clip(zn, 0, 1) * 255).astype(np.uint8)]

    gy, gx = np.gradient(zn)                       # rows go down, cols go right
    normal = np.stack([-relief * gx, -relief * gy, np.ones_like(zn)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    light = np.asarray(light, float)
    light /= np.linalg.norm(light)
    shade = normal @ light
    diffuse = np.clip(1.0 + 1.9 * (shade - light[2]), 0.45, 1.75)

    half = light + np.array([0.0, 0.0, 1.0])       # viewer looks straight down
    half /= np.linalg.norm(half)
    specular = np.clip(normal @ half, 0, 1) ** 28 * gloss

    rgb = base * diffuse[..., None] + specular[..., None] * np.array([0.55, 0.80, 1.0])
    return np.clip(rgb, 0, 1)


def land_layer(land_cells, shape_px):
    """Pixel masks for land and for a thin bright coastline."""
    if land_cells is None or not land_cells.any():
        empty = np.zeros(shape_px, dtype=bool)
        return empty, empty
    soft = upsample(np.flipud(land_cells).astype(float), shape_px, order=1)
    land = gaussian_filter(soft, sigma=6) > 0.5          # smooth the staircase coast
    coast = land & ~binary_erosion(land, iterations=2)
    return land, coast


def apply_land(rgb, land, coast):
    out = rgb.copy()
    out[land] = LAND_COLOUR
    out[coast] = COAST_COLOUR
    return out


def vignette(shape_px, strength=0.35):
    """Soft darkening towards the edges (also hides sponge layers)."""
    h, w = shape_px
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2))
    return (1.0 - strength * np.clip(r - 0.55, 0, 1) ** 1.6)[..., None]


# ---------------------------------------------------------------------------
# Layer 2: tracer trails
# ---------------------------------------------------------------------------
class TracerCanvas:
    """A float image that remembers where tracers have been and fades slowly."""

    def __init__(self, shape_px, fade=0.94, glow=0.45, glow_sigma=1.8):
        self.h, self.w = shape_px
        self.fade, self.glow, self.glow_sigma = fade, glow, glow_sigma
        self.buf = np.zeros((self.h, self.w, 3), dtype=np.float32)

    def fade_step(self):
        self.buf *= self.fade

    def stamp(self, px, py, rgb):
        """Paint particles (2x2 pixels each). rgb is (n, 3) in 0-1."""
        x = np.rint(px).astype(np.int64)
        y = np.rint(py).astype(np.int64)
        keep = (x >= 0) & (x < self.w - 1) & (y >= 0) & (y < self.h - 1)
        x, y, rgb = x[keep], y[keep], rgb[keep].astype(np.float32)
        for dy in (0, 1):
            for dx in (0, 1):
                cur = self.buf[y + dy, x + dx]
                self.buf[y + dy, x + dx] = np.maximum(cur, rgb)

    def composite(self, background):
        """Screen-blend the glowing trails over the background."""
        halo = gaussian_filter(self.buf, sigma=(self.glow_sigma, self.glow_sigma, 0))
        trail = np.clip(self.buf + self.glow * halo, 0, 1)
        return 1.0 - (1.0 - background) * (1.0 - trail)


# ---------------------------------------------------------------------------
# Layer 3: HUD text
# ---------------------------------------------------------------------------
def _font(bold, size):
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


class Hud:
    """Title block, colour bar and footer; only the date text changes per frame."""

    def __init__(self, shape_px, title, subtitle, footer, cbar_label, cbar_max,
                 cbar_unit="m/s"):
        h, w = shape_px
        self.f_title, self.f_sub = _font(True, 30), _font(False, 17)
        self.f_small, self.f_date = _font(False, 15), _font(True, 22)
        self.f_cap = _font(True, 18)
        self.size = (w, h)
        overlay = Image.new("RGBA", self.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        # title panel
        d.rounded_rectangle([24, 22, 24 + 640, 22 + 92], radius=14, fill=(4, 10, 22, 150))
        d.text((44, 30), title, font=self.f_title, fill=(235, 245, 255, 255))
        d.text((44, 74), subtitle, font=self.f_sub, fill=(150, 200, 225, 255))

        # colour bar
        bx, by, bw, bh = 44, h - 70, 260, 12
        d.rounded_rectangle([24, h - 108, 24 + 300, h - 24], radius=14, fill=(4, 10, 22, 150))
        strip = (TRACER_LUT[np.linspace(0, 255, bw).astype(int)] * 255).astype(np.uint8)
        strip = np.repeat(strip[None], bh, axis=0)
        overlay.paste(Image.fromarray(strip, "RGB"), (bx, by))
        d.text((bx, h - 98), cbar_label, font=self.f_small, fill=(200, 225, 240, 255))
        d.text((bx, by + bh + 6), "0", font=self.f_small, fill=(200, 225, 240, 255))
        d.text((bx + bw - 62, by + bh + 6), f"{cbar_max:.2f} {cbar_unit}",
               font=self.f_small, fill=(200, 225, 240, 255))

        # footer (right aligned)
        tw = d.textlength(footer, font=self.f_small)
        d.rounded_rectangle([w - 28 - tw - 16, h - 44, w - 12, h - 14], radius=10,
                            fill=(4, 10, 22, 150))
        d.text((w - 28 - tw, h - 37), footer, font=self.f_small, fill=(185, 210, 228, 255))
        self.overlay = overlay
        self.date_pos = (w - 28, 34)

    def draw(self, frame_rgb_uint8, date_text, caption=None):
        img = Image.alpha_composite(Image.fromarray(frame_rgb_uint8).convert("RGBA"), self.overlay)
        d = ImageDraw.Draw(img)
        tw = d.textlength(date_text, font=self.f_date)
        d.rounded_rectangle([self.date_pos[0] - tw - 20, 22, self.date_pos[0] + 14, 66],
                            radius=12, fill=(4, 10, 22, 150))
        d.text((self.date_pos[0] - tw, 30), date_text, font=self.f_date,
               fill=(255, 235, 190, 255))
        if caption:                                   # stage description, bottom centre
            cw = d.textlength(caption, font=self.f_cap)
            x0 = (self.size[0] - cw) / 2
            d.rounded_rectangle([x0 - 18, self.size[1] - 122, x0 + cw + 18, self.size[1] - 80],
                                radius=12, fill=(4, 10, 22, 175))
            d.text((x0, self.size[1] - 113), caption, font=self.f_cap, fill=(255, 235, 190, 255))
        return np.asarray(img.convert("RGB"))


def to_uint8(rgb):
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------
class VideoWriter:
    """Write frames to an H.264 MP4 (uses the ffmpeg bundled with imageio)."""

    def __init__(self, path, fps=30, crf=26):
        import imageio
        self.path = str(path)
        self.writer = imageio.get_writer(
            self.path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
            ffmpeg_params=["-crf", str(crf), "-preset", "medium",
                           "-movflags", "+faststart"])

    def add(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


def mp4_to_gif(mp4_path, gif_path, width=640, fps=15):
    """Make a smaller GIF (good for a GitHub README) from the MP4."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4_path),
                    "-vf", vf, str(gif_path)], check=True)
