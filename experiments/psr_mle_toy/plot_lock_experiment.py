"""Render an episode-versus-TV SVG from four-method lock run rows."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
from math import log10
from pathlib import Path
from statistics import median


COLORS = {"ours": "#2369a8", "omle": "#c43d35",
          "uniform": "#d39526", "count": "#438e5a"}
METHODS = ("ours", "omle", "uniform", "count")


def curve(paths, budget):
    times = sorted({int(row["episodes"]) for path in paths for row in path
                    if int(row["episodes"]) <= budget})
    points = []
    for time in times:
        values = []
        for path in paths:
            observed = [float(row["tv"]) for row in path
                        if int(row["episodes"]) <= time]
            if observed:
                values.append(observed[-1])
        if len(values) == len(paths):
            points.append((time, median(values)))
    return points


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bonus-output", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["horizon"]), row["method"], int(row["seed"]))].append(row)
    horizons = sorted({key[0] for key in grouped})
    width, height = 1000, 400 * len(horizons) + 60
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
              '<style>text{font-family:Arial,sans-serif;fill:#25313d}'
              '.axis{stroke:#758390;stroke-width:1}.grid{stroke:#e3e9ed;stroke-width:1}'
              '.line{fill:none;stroke-width:2.7;stroke-linejoin:round}</style>',
              f'<rect width="{width}" height="{height}" fill="white"/>',
              '<text x="70" y="34" font-size="20" font-weight="bold">'
              'Partially observed lock: maximum policy TV</text>']
    left, right = 80, 950
    ticks_x = (64, 256, 1024, 4096, 16384, 65536, 131072)
    for panel, horizon in enumerate(horizons):
        top = 80 + panel * 400
        bottom = top + 270

        def xcoord(t):
            return left + (log10(max(1, t)) / log10(131072)) * (right - left)

        def ycoord(tv):
            return bottom - min(.8, max(0., tv)) / .8 * (bottom - top)

        pieces.append(f'<text x="{left}" y="{top-22}" font-size="17" '
                      f'font-weight="bold">H = {horizon}</text>')
        for tick in (0, .2, .4, .6, .8):
            y = ycoord(tick)
            pieces.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                          f'x2="{right}" y2="{y:.1f}"/>')
            pieces.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
                          f'font-size="11">{tick:g}</text>')
        for tick in ticks_x:
            x = xcoord(tick)
            pieces.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" '
                          f'x2="{x:.1f}" y2="{bottom}"/>')
            pieces.append(f'<text x="{x:.1f}" y="{bottom+18}" text-anchor="middle" '
                          f'font-size="10">{tick:,}</text>')
        pieces.append(f'<line class="axis" x1="{left}" y1="{bottom}" '
                      f'x2="{right}" y2="{bottom}"/>')
        pieces.append(f'<text x="{(left+right)/2}" y="{bottom+39}" '
                      f'text-anchor="middle" font-size="12">Episodes (log scale)</text>')
        pieces.append(f'<text x="20" y="{(top+bottom)/2}" '
                      f'transform="rotate(-90 20 {(top+bottom)/2})" '
                      f'text-anchor="middle" font-size="12">Median worst-policy TV</text>')
        for method in METHODS:
            paths = [path for (h, m, _), path in grouped.items()
                     if h == horizon and m == method]
            budget = 131072 if method == "ours" else 4096
            points = curve(paths, budget)
            if points:
                coords = " ".join(f"{xcoord(t):.1f},{ycoord(v):.1f}" for t, v in points)
                pieces.append(f'<polyline class="line" points="{coords}" '
                              f'stroke="{COLORS[method]}"/>')
        legend_y = bottom + 64
        for index, method in enumerate(METHODS):
            x = left + index * 190
            pieces.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+25}" '
                          f'y2="{legend_y}" stroke="{COLORS[method]}" stroke-width="3"/>')
            pieces.append(f'<text x="{x+33}" y="{legend_y+4}" '
                          f'font-size="12">{method}</text>')
    pieces.append('</svg>')
    args.output.write_text("\n".join(pieces) + "\n")
    if args.bonus_output:
        bonus = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 480">',
                 '<style>text{font-family:Arial,sans-serif;fill:#25313d}'
                 '.grid{stroke:#e3e9ed}.line{fill:none;stroke-width:2.7}</style>',
                 '<rect width="900" height="480" fill="white"/>',
                 '<text x="72" y="35" font-size="20" font-weight="bold">'
                 'Ours: fitted bonus and stopping threshold</text>']
        xleft, xright, ytop, ybottom = 80, 850, 70, 390
        max_episodes = max(int(row["episodes"]) for row in rows
                           if row["method"] == "ours")
        ymax = max(float(row["bonus"]) for row in rows
                   if row["method"] == "ours")

        def bx(t):
            return xleft + t / max_episodes * (xright - xleft)

        def by(value):
            return ybottom - value / ymax * (ybottom - ytop)

        for tick in (i * ymax / 5 for i in range(6)):
            y = by(tick)
            bonus.append(f'<line class="grid" x1="{xleft}" y1="{y:.1f}" '
                         f'x2="{xright}" y2="{y:.1f}"/>')
            bonus.append(f'<text x="{xleft-10}" y="{y+4:.1f}" '
                         f'text-anchor="end" font-size="11">{tick:.2f}</text>')
        threshold = .05  # epsilon=.2 in the reproducible primary command
        y = by(threshold)
        bonus.append(f'<line x1="{xleft}" y1="{y:.1f}" x2="{xright}" '
                     f'y2="{y:.1f}" stroke="#c43d35" stroke-dasharray="5 4"/>')
        bonus.append(f'<text x="{xright-5}" y="{y-7:.1f}" '
                     f'text-anchor="end" font-size="11">stop at ε/4 = 0.05</text>')
        for index, horizon in enumerate(horizons):
            paths = [path for (h, m, _), path in grouped.items()
                     if h == horizon and m == "ours"]
            times = sorted({int(row["episodes"]) for path in paths for row in path})
            values = []
            for time in times:
                latest = []
                for path in paths:
                    observed = [float(row["bonus"]) for row in path
                                if int(row["episodes"]) <= time]
                    if observed:
                        latest.append(observed[-1])
                if len(latest) == len(paths):
                    values.append((time, median(latest)))
            color = ("#2369a8", "#438e5a", "#9b59b6")[index % 3]
            coords = " ".join(f"{bx(t):.1f},{by(v):.1f}" for t, v in values)
            bonus.append(f'<polyline class="line" points="{coords}" '
                         f'stroke="{color}"/>')
            lx = 90 + 170 * index
            bonus.append(f'<line x1="{lx}" y1="435" x2="{lx+25}" y2="435" '
                         f'stroke="{color}" stroke-width="3"/>')
            bonus.append(f'<text x="{lx+32}" y="439" font-size="12">'
                         f'H={horizon}</text>')
        bonus.append('<text x="465" y="470" text-anchor="middle" '
                     'font-size="12">Episodes</text></svg>')
        args.bonus_output.write_text("\n".join(bonus) + "\n")


if __name__ == "__main__":
    main()
