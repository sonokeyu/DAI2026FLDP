"""Plot each method's episode/update frontier at a fixed mean TV."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
from pathlib import Path


METHODS = {
    "omle": ("omleline", "o", 3.0),
    "ours": ("oursline", "square*", 1.9),
}


def figure(source, target):
    rows = defaultdict(list)
    with source.open() as file:
        for row in csv.DictReader(file):
            if float(row["target_tv"]) == target and row["method"] in METHODS:
                rows[float(row["gap"]), row["method"]].append(row)
    gaps = sorted({gap for gap, _ in rows})
    if not gaps:
        raise ValueError("the requested mean TV target is not in the sweep")
    budget = int(next(iter(rows.values()))[0]["budget"])
    ticks = ",".join(str(n) for n in (16, 64, 256, 1024, 4096) if n <= budget)
    lines = [r"""
\documentclass[border=8pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{oursline}{HTML}{00747C}
\definecolor{omleline}{HTML}{773588}
\definecolor{ink}{HTML}{17242D}
\definecolor{muted}{HTML}{4A5C67}
\definecolor{gridline}{HTML}{E3EAED}
\begin{document}
\begin{tikzpicture}[text=ink]
\begin{groupplot}[
  group style={group size=3 by 1,horizontal sep=0.8cm},
  width=5.1cm,height=5.1cm,scale only axis,
  xmode=log,log basis x=2,
  ymode=log,log basis y=2,
  xmin=0.85, xmax=900, ymin=14,
  xtick={1,4,16,64,256},xticklabels={1,4,16,64,256},
  tick align=outside,tick style={muted!75},
  tick label style={font=\footnotesize,text=muted},
  label style={font=\small,text=ink},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},
  axis lines*=left,axis line style={muted!80},
  xlabel={Model updates},
  clip=false,
]"""]
    for panel, gap in enumerate(gaps, start=1):
        options = (f"title={{$\\Delta={gap:g}$}},ymax={1.12 * budget:g},"
                   f"ytick={{{ticks}}}")
        if panel == 1:
            options += f",yticklabels={{{ticks}}},ylabel={{Environment episodes}}"
        else:
            options += ",yticklabels={}"
        lines.append(r"\nextgroupplot[" + options + "]")
        for method, (color, mark, size) in METHODS.items():
            selected = rows[gap, method]
            frontier = sorted({(int(r["updates"]), int(r["episodes"])) for r in selected
                               if r["method_frontier"] == "True"})
            points = " ".join(f"({updates},{episodes})" for updates, episodes in frontier)
            if points:
                lines.append(rf"\addplot[{color},thick,mark={mark},mark size={size:g}pt] "
                             + "coordinates {" + points + "};")
            if method == "ours":
                labeled = set()
                for row in selected:
                    if row["method_frontier"] != "True":
                        continue
                    point = (int(row["updates"]), int(row["episodes"]))
                    if point in labeled:
                        continue
                    labeled.add(point)
                    x, y = point
                    lines.append(rf"\node[anchor=west,xshift=2pt,font=\scriptsize,text={color}] "
                                 rf"at (axis cs:{x},{y}) {{$B={row['batch']}$}};")
            anchor_batch = "2" if method == "omle" else "8"
            anchors = [(int(r["updates"]), int(r["episodes"])) for r in selected
                       if r["batch"] == anchor_batch and r["reached"] == "True"]
            for updates, episodes in anchors:
                lines.append(rf"\addplot[only marks,{color},mark=+,mark size=4pt] "
                             rf"coordinates {{({updates},{episodes})}};")

    lines.append(r"\end{groupplot}")
    lines.append(r"\node[anchor=west,inner sep=0pt,font=\Large\bfseries] at "
                 r"([yshift=26mm]group c1r1.north west) {Tradeoff across batch sizes};")
    lines.append(r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at "
                 r"([yshift=19mm]group c1r1.north west) "
                 rf"{{Costs to sustain mean worst-policy TV $\leq {target:.3f}$ through episode {budget}.}};")
    lines.append(r"\node[anchor=west,font=\small,text=omleline] at "
                 r"([xshift=-1mm,yshift=11mm]group c1r1.north west) {"
                 r"$\circ$\enspace\textbf{omle}};")
    lines.append(r"\node[anchor=west,font=\small,text=oursline] at "
                 r"([xshift=22mm,yshift=11mm]group c1r1.north west) {"
                 r"\rule{4pt}{4pt}\enspace\textbf{ours}};")
    lines.append(r"\node[anchor=west,font=\footnotesize,text=muted] at "
                 r"([xshift=43mm,yshift=11mm]group c1r1.north west) {"
                 r"$+$ marks the original intervals: omle 2, ours 8.};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines).lstrip() + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--target", type=float, default=0.01)
    args = parser.parse_args()
    args.figure.write_text(figure(args.summary, args.target))


if __name__ == "__main__":
    main()
