"""
run_slope_simulation.py
=======================
Watch water run down a slope, get turned by the Earth's rotation, and settle
into the geostrophic current that your original notebook calculates.

    python run_slope_simulation.py                    # a hill next to a hollow
    python run_slope_simulation.py --shape mound      # a single hill of water
    python run_slope_simulation.py --shape ridge      # a long wall of water
    python run_slope_simulation.py --check            # test against theory, no video

The background is the water surface (lit from the top-left). The glowing
streaks are tracer particles carried by the flow.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from geoflow import render
from geoflow.shallow_water import (RotatingShallowWater, initial_height,
                                   rossby_adjustment_check)

NX, NY, DX = 320, 180, 4000.0          # 1280 km x 720 km, 4 km cells
DT = 150.0                             # seconds
SPONGE = 40                            # cells


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", choices=["dipole", "mound", "ridge"], default="dipole")
    p.add_argument("--amplitude", type=float, default=60.0, help="initial height (m)")
    p.add_argument("--size", type=float, default=70.0, help="size of the hill (km)")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--particles", type=int, default=14000)
    p.add_argument("--out", default="media/slope_simulation.mp4")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="only run the comparison with the textbook solution")
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args()


def run_check():
    r = rossby_adjustment_check()
    print("Rossby adjustment test (water 'dam' on a rotating planet)")
    print(f"  Rossby radius              : {r['rossby_radius_km']:.1f} km")
    print(f"  jet speed  theory / model  : {r['v_peak_theory']:.5f} / {r['v_peak_model']:.5f} m/s "
          f"({100 * r['v_peak_error']:+.1f}%)")
    print(f"  height profile RMS error   : {100 * r['eta_rel_error']:.2f}%")


def make_sim(args):
    sim = RotatingShallowWater(NX, NY, DX, DT, sponge_cells=SPONGE)
    sim.set_initial_height(initial_height(args.shape, NX, NY, DX,
                                          amplitude=args.amplitude, size=args.size * 1e3))
    return sim


def stage_caption(t_days):
    if t_days < 0.35:
        return "1  Gravity pulls the water down the slope; ripples race outwards"
    if t_days < 1.5:
        return "2  Rotation turns the moving water sideways"
    return "3  Balance: flow runs ALONG the height contours (geostrophic current)"


def main():
    args = parse_args()
    if args.check:
        run_check()
        return
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- pre-run: how fast is the settled flow? (sets colours and tracer speed)
    pre = make_sim(args)
    pre.run(int(3 * 86400 / DT))
    uc, vc = pre.velocity_at_centres()
    inner = (slice(SPONGE, NY - SPONGE), slice(SPONGE, NX - SPONGE))
    speed_p97 = float(np.percentile(np.hypot(uc, vc)[inner], 97))
    print(f"Rossby radius {pre.rossby_radius / 1e3:.0f} km, wave speed {pre.c:.1f} m/s, "
          f"settled flow (97th pct) {speed_p97:.3f} m/s")

    # ---- time plan: slow motion first, then faster
    n_frames = int(args.seconds * args.fps)
    slow = int(n_frames * 0.25)
    steps_per_frame = np.where(np.arange(n_frames) < slow, 3, 12)
    frame_seconds = steps_per_frame * DT
    # tracers move `boost` times faster than the simulated clock so they are visible
    boost = max(1.0, round(3000.0 / (speed_p97 * frame_seconds[-1])))
    print(f"{n_frames} frames, {frame_seconds.sum() / 86400:.1f} simulated days, tracer boost x{boost:.0f}")

    # ---- video set-up
    shape_px = (NY * 4, NX * 4)
    cell_shape = (NY, NX)
    canvas = render.TracerCanvas(shape_px, fade=0.95)
    shade_vignette = render.vignette(shape_px, strength=0.75)
    hud = render.Hud(shape_px, "Water released on a slope",
                     "rotating shallow-water simulation",
                     f"Tracers shown x{boost:.0f} faster than the model clock",
                     "Current speed", speed_p97)
    vlim = args.amplitude * 1.05

    sim = make_sim(args)
    margin = 8                       # tracers live away from the very edge
    def spawn(k):
        return (rng.uniform(margin, NX - 1 - margin, k), rng.uniform(margin, NY - 1 - margin, k))
    ix, iy = spawn(args.particles)
    life = rng.integers(45, 130, args.particles).astype(float)
    age = rng.uniform(0, 1, args.particles) * life

    writer = render.VideoWriter(out, fps=args.fps)
    still_frames = {int(n_frames * 0.10): "stage1", int(n_frames * 0.35): "stage2",
                    int(n_frames * 0.90): "stage3"}
    t0 = time.time()
    for frame in range(n_frames):
        sim.run(int(steps_per_frame[frame]))
        uc, vc = sim.velocity_at_centres()
        canvas.fade_step()

        sub_dt = frame_seconds[frame] * boost / 3.0          # seconds of tracer time
        for _ in range(3):
            # midpoint (RK2) step in grid-cell units
            def rate(x, y):
                c = np.array([y, x])
                return (map_coordinates(uc, c, order=1, mode="nearest") / DX,
                        map_coordinates(vc, c, order=1, mode="nearest") / DX)
            k1 = rate(ix, iy)
            k2 = rate(ix + 0.5 * sub_dt * k1[0], iy + 0.5 * sub_dt * k1[1])
            ix = ix + sub_dt * k2[0]
            iy = iy + sub_dt * k2[1]

            spd = np.hypot(*[map_coordinates(m, np.array([iy, ix]), order=1, mode="nearest")
                             for m in (uc, vc)])
            rel = np.clip(spd / speed_p97, 0, 1)
            colour = render.TRACER_LUT[(rel * 255).astype(np.uint8)]
            fade_in_out = np.sin(np.pi * np.clip(age / life, 0, 1)) ** 0.6
            brightness = (0.30 + 0.70 * fade_in_out) * (0.03 + 0.97 * rel ** 1.3)
            # fade tracers out smoothly towards the edges (hides the sponge layer)
            edge = np.minimum.reduce([ix, NX - 1 - ix, iy, NY - 1 - iy])
            e = np.clip((edge - 8) / 32.0, 0, 1)
            brightness = brightness * e * e * (3 - 2 * e)
            px, py = render.cells_to_pixels(ix, iy, cell_shape, shape_px)
            canvas.stamp(px, py, colour * brightness[:, None])

        age += 1
        gone = ((age >= life) | (ix < margin) | (ix > NX - 1 - margin)
                | (iy < margin) | (iy > NY - 1 - margin))
        if gone.any():
            k = int(gone.sum())
            ix[gone], iy[gone] = spawn(k)
            age[gone] = 0
            life[gone] = rng.integers(45, 130, k)

        bg = render.water_surface(sim.eta, shape_px, -vlim, vlim, relief=170.0, gloss=0.6)
        img = canvas.composite(bg) * shade_vignette
        t_days = sim.time / 86400.0
        frame_u8 = hud.draw(render.to_uint8(img), f"t = {t_days:4.1f} days", stage_caption(t_days))
        writer.add(frame_u8)

        if frame in still_frames:
            from PIL import Image
            Image.fromarray(frame_u8).save(out.with_name(f"{out.stem}_{still_frames[frame]}.png"))
        if frame % 60 == 0:
            print(f"  frame {frame:4d}/{n_frames}   t={t_days:5.2f} d   {time.time() - t0:5.1f}s")

    writer.close()
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)  in {time.time() - t0:.0f}s")
    if args.gif:
        gif = out.with_suffix(".gif")
        render.mp4_to_gif(out, gif)
        print(f"Saved {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
