"""
run_flow_video.py
=================
Animate surface currents derived from sea surface height (ADT).

    python run_flow_video.py                      # built-in synthetic ocean
    python run_flow_video.py --file dataset.nc    # your DUACS / Copernicus file

Particles are released over the ocean and carried by the geostrophic current,
so what you see is the path real water would follow.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from geoflow import core, render


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="NetCDF file with an 'adt' variable (default: synthetic ocean)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                   help="crop the region")
    p.add_argument("--seconds", type=float, default=20, help="video length")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--particles", type=int, default=16000)
    p.add_argument("--width", type=int, default=1280, help="video width in pixels")
    p.add_argument("--days-per-frame", type=float, default=None,
                   help="simulated days per video frame (default: fit the record, max 0.15)")
    p.add_argument("--steady", action="store_true",
                   help="freeze the field at the first map (like your original notebook)")
    p.add_argument("--out", default="media/surface_currents.mp4")
    p.add_argument("--gif", action="store_true", help="also write a small GIF for a README")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. data -----------------------------------------------------------
    if args.file:
        data = core.load_netcdf(args.file, bbox=args.bbox)
        footer = f"Data: {Path(args.file).name}"
    else:
        data = core.synthetic_ocean()
        footer = "Synthetic demo ocean (Gulf-Stream-like)"
    if args.steady:
        data.adt, data.days, data.labels = data.adt[:1], data.days[:1], data.labels[:1]
    print(f"Data: {data.source} | {data.shape[0]} maps, {data.shape[1]}x{data.shape[2]} grid")

    # ---- 2. physics --------------------------------------------------------
    u, v = core.geostrophic_velocity(data.adt, data.lat, data.lon)
    speed = np.hypot(u, v)
    vmax = float(np.nanpercentile(speed, 97))
    print(f"Surface speed: max {np.nanmax(speed):.2f} m/s, 97th percentile {vmax:.2f} m/s")
    check = core.compare_with_official(data, u, v)
    if check:
        print(f"Check against DUACS ugos/vgos: correlation {check['correlation']:.3f}, "
              f"RMSE {check['rmse_m_per_s']:.3f} m/s")

    adv = core.TracerAdvector(data, u, v)
    cell_shape = (len(data.lat), len(data.lon))

    # ---- 3. video geometry -------------------------------------------------
    scale = args.width / cell_shape[1]
    shape_px = (int(round(cell_shape[0] * scale)) // 2 * 2, args.width // 2 * 2)
    n_frames = int(args.seconds * args.fps)
    span = data.days[-1] - data.days[0]
    dpf = args.days_per_frame or (min(0.15, span / n_frames) if span > 0 else 0.1)
    substeps = 3
    print(f"Video: {shape_px[1]}x{shape_px[0]}, {n_frames} frames, {dpf:.3f} simulated days per frame")

    adt_filled = np.stack([render.fill_nan_nearest(m) for m in data.adt])
    vmin_h, vmax_h = np.nanpercentile(data.adt, [1, 99])
    land_cells = ~np.isfinite(data.adt).all(axis=0)
    land, coast = render.land_layer(land_cells, shape_px)
    shade_vignette = render.vignette(shape_px)
    canvas = render.TracerCanvas(shape_px)
    hud = render.Hud(shape_px, "Surface currents from sea-surface height",
                     "geostrophic balance:  u = -(g/f) dη/dy ,  v = (g/f) dη/dx",
                     footer, "Current speed", vmax)

    # ---- 4. particles ------------------------------------------------------
    n = args.particles
    lon, lat = adv.random_positions(n, rng)
    life = rng.integers(45, 130, n).astype(float)
    age = rng.uniform(0, 1, n) * life                  # staggered start

    writer = render.VideoWriter(out, fps=args.fps)
    still_frame = int(n_frames * 0.7)
    t0 = time.time()
    for frame in range(n_frames):
        t = frame * dpf
        canvas.fade_step()

        # move particles a little, stamping a point after each sub-step
        for s in range(substeps):
            lon, lat = adv.step(t + s * dpf / substeps, lon, lat, dpf / substeps)
            uu, vv = adv.sample(t + (s + 1) * dpf / substeps, lon, lat)
            spd = np.hypot(uu, vv)
            fade_in_out = np.sin(np.pi * np.clip(age / life, 0, 1)) ** 0.6
            rel = np.clip(spd / vmax, 0, 1)
            colour = render.TRACER_LUT[(rel * 255).astype(np.uint8)]
            # slow water is drawn dim, fast water bright, so the jet stands out
            # and the shaded water surface stays visible underneath
            brightness = (0.30 + 0.70 * fade_in_out) * (0.22 + 0.78 * rel ** 0.7)
            ix, iy = adv.to_cell_coords(lon, lat)
            px, py = render.cells_to_pixels(ix, iy, cell_shape, shape_px)
            canvas.stamp(px, py, colour * brightness[:, None])

        # age, retire and respawn
        age += 1
        dead = (age >= life) | ~adv.in_ocean(lon, lat)
        if dead.any():
            k = int(dead.sum())
            lon[dead], lat[dead] = adv.random_positions(k, rng)
            age[dead] = 0
            life[dead] = rng.integers(45, 130, k)

        # background at this time
        if adt_filled.shape[0] == 1:
            eta = adt_filled[0]
        else:
            k0 = min(int(np.searchsorted(data.days, t, side="right") - 1), len(data.days) - 2)
            k0 = max(k0, 0)
            w = np.clip((t - data.days[k0]) / (data.days[k0 + 1] - data.days[k0]), 0, 1)
            eta = (1 - w) * adt_filled[k0] + w * adt_filled[k0 + 1]
        bg = render.water_surface(eta, shape_px, vmin_h, vmax_h)
        bg = render.apply_land(bg, land, coast)
        img = canvas.composite(bg) * shade_vignette

        label = data.labels[int(np.clip(np.searchsorted(data.days, t, side="right") - 1,
                                        0, len(data.labels) - 1))]
        frame_u8 = hud.draw(render.to_uint8(img), label)
        writer.add(frame_u8)

        if frame == still_frame:
            from PIL import Image
            Image.fromarray(frame_u8).save(out.with_suffix(".png"))
        if frame % 60 == 0:
            print(f"  frame {frame:4d}/{n_frames}   {time.time() - t0:5.1f}s")

    writer.close()
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)  in {time.time() - t0:.0f}s")
    print(f"Saved still {out.with_suffix('.png')}")
    if args.gif:
        gif = out.with_suffix(".gif")
        render.mp4_to_gif(out, gif)
        print(f"Saved {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
