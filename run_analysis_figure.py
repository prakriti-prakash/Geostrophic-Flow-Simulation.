"""
run_analysis_figure.py
======================
A clean 4-panel figure: the improved version of the plot in the original notebook.

    python run_analysis_figure.py                       # synthetic ocean
    python run_analysis_figure.py --file dataset.nc     # your data
    python run_analysis_figure.py --file dataset.nc --time-index 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geoflow import core


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file")
    p.add_argument("--bbox", type=float, nargs=4,
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--time-index", type=int, default=30)
    p.add_argument("--out", default="media/analysis.png")
    return p.parse_args()


def main():
    args = parse_args()
    data = core.load_netcdf(args.file, bbox=args.bbox) if args.file else core.synthetic_ocean()
    k = min(args.time_index, data.shape[0] - 1)
    eta = data.adt[k]
    u, v = core.geostrophic_velocity(eta, data.lat, data.lon)
    speed = np.hypot(u, v)
    rossby = core.rossby_number(u, v, data.lat, data.lon)
    lon, lat = data.lon, data.lat
    extent = [lon[0], lon[-1], lat[0], lat[-1]]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    fig.suptitle(f"Geostrophic surface currents  -  {data.labels[k]}   ({Path(data.source).name})",
                 fontsize=15, fontweight="bold")

    # (a) height + streamlines (fixed version of the original plot)
    a = ax[0, 0]
    im = a.imshow(eta, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    a.contour(lon, lat, eta, levels=15, colors="white", linewidths=0.4, alpha=0.7)
    a.streamplot(lon, lat, np.ma.masked_invalid(u), np.ma.masked_invalid(v), color="k",
                 density=1.6, linewidth=0.6, arrowsize=0.8)
    fig.colorbar(im, ax=a, label="ADT (m)")
    a.set_title("Sea surface height with streamlines")

    # (b) speed
    b = ax[0, 1]
    im = b.imshow(speed, origin="lower", extent=extent, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=b, label="speed (m/s)")
    b.set_title("Current speed")

    # (c) Rossby number
    c = ax[1, 0]
    lim = float(np.nanpercentile(np.abs(rossby), 98))
    im = c.imshow(rossby, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim,
                  aspect="auto")
    fig.colorbar(im, ax=c, label="relative vorticity / f")
    c.set_title("Rossby number  (red = anticlockwise, blue = clockwise)")

    # (d) transect across the strongest current: the geostrophic relation in one picture
    d = ax[1, 1]
    i = int(np.nanargmax(np.nanmax(speed, axis=0)))
    (l1,) = d.plot(lat, eta[:, i], color="tab:blue", lw=2, label="sea surface height (m)")
    d.set_ylabel("height (m)", color="tab:blue")
    d.set_xlabel("latitude")
    d2 = d.twinx()
    (l2,) = d2.plot(lat, u[:, i], color="tab:red", lw=2, label="eastward current u (m/s)")
    (l3,) = d2.plot(lat, v[:, i], color="tab:orange", lw=1.5, ls="--", label="northward current v (m/s)")
    d2.axhline(0, color="gray", lw=0.5)
    d2.set_ylabel("current (m/s)", color="tab:red")
    hemi = "E" if lon[i] >= 0 else "W"
    d.set_title(f"Transect at {abs(lon[i]):.2f}\u00b0{hemi}: the current peaks where the height changes fastest")
    d.legend([l1, l2, l3], [l.get_label() for l in (l1, l2, l3)], loc="upper right", fontsize=9)

    for panel in (a, b, c):
        panel.set_xlabel("longitude")
        panel.set_ylabel("latitude")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")

    check = core.compare_with_official(data, *core.geostrophic_velocity(data.adt, data.lat, data.lon))
    if check:
        print(f"Against DUACS ugos/vgos: correlation {check['correlation']:.3f}, "
              f"RMSE {check['rmse_m_per_s']:.3f} m/s over {check['points']} points")


if __name__ == "__main__":
    main()
