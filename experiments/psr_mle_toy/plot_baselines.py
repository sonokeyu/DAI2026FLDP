"""Plot sampling cost at fixed TV with a shared update interval."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
from pathlib import Path


METHODS = {
    "omle_matched": ("omle", "omleline", "circle"),
    "ours": ("ours", "oursline", "rectangle"),
    "uniform": ("uniform", "uniformline", "diamond"),
    "count": ("count", "countline", "regular polygon,regular polygon sides=3"),
}


def figure(source):
    rows = defaultdict(dict)
    with source.open() as file:
        for row in csv.DictReader(file):
            if row["condition"] in METHODS:
                rows[float(row["gap"]), float(row["target_tv"])][row["condition"]] = row
    gaps = sorted({gap for gap, _ in rows})
    targets = sorted({target for _, target in rows}, reverse=True)
    budget = int(next(iter(rows.values()))["ours"]["budget"])
    ticks = ",".join(str(2 ** k) for k in range(4, budget.bit_length(), 2))
    bands = [2.5 + (len(targets) - i - 1) * 5 for i in range(len(targets))]
    lines = [r"""
\documentclass[border=8pt]{standalone}
\usepackage{pgfplots}
\usetikzlibrary{shapes.geometric}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{oursline}{HTML}{00747C}
\definecolor{omleline}{HTML}{773588}
\definecolor{uniformline}{HTML}{9D5000}
\definecolor{countline}{HTML}{2557A7}
\definecolor{ink}{HTML}{17242D}
\definecolor{muted}{HTML}{4A5C67}
\definecolor{gridline}{HTML}{E3EAED}
\begin{document}
\begin{tikzpicture}[text=ink]
\begin{groupplot}[
  group style={group size=3 by 1,horizontal sep=0.7cm},
  width=5.4cm,height=7.4cm,scale only axis,
  xmode=log,log basis x=2,
  clip=false,
  tick align=outside,tick style={muted!75},
  tick label style={font=\footnotesize,text=muted},
  label style={font=\small,text=ink},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},
  axis lines*=left,axis line style={muted!80},
  xlabel={Environment episodes},
  log ticks with fixed point,
]"""]
    for panel, gap in enumerate(gaps, start=1):
        options = (
            f"title={{$\\Delta={gap:g}$}},xmin=7,xmax={2.0 * budget:g},"
            f"ymin=0,ymax={5 * len(targets)},"
            f"ytick={{{','.join(map(str, bands))}}},"
            f"xtick={{{ticks}}},xticklabels={{{ticks}}}"
        )
        if panel == 1:
            options += ",yticklabels={" + ",".join(f"{t:.3f}" for t in targets) + "},ylabel={Target mean TV}"
        else:
            options += ",yticklabels={}"
        lines.append(r"\nextgroupplot[" + options + "]")
        for target, center in zip(targets, bands):
            for offset, (name, (_, color, marker)) in enumerate(METHODS.items()):
                row = rows[gap, target][name]
                reached = row["reached"] == "True"
                value = int(row["episodes"] if reached else row["budget"])
                y = center + 1.5 - offset
                annotation = str(value) if reached else rf"$>{{}}{value}$"
                lines.append(rf"\node[{marker},draw={color},"
                             rf"fill={'white' if not reached else color},inner sep=1.6pt] "
                             rf"at (axis cs:{value},{y:g}) {{}};")
                lines.append(rf"\node[anchor=west,font=\scriptsize,text={color},xshift=3pt] "
                             rf"at (axis cs:{value},{y:g}) {{{annotation}}};")
    lines.append(r"\end{groupplot}")
    lines.append(r"\node[anchor=west,inner sep=0pt,font=\Large\bfseries] at "
                 r"([yshift=26mm]group c1r1.north west) {Sampling cost with the same update interval};")
    lines.append(r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at "
                 r"([yshift=19mm]group c1r1.north west) "
                 rf"{{Every method updates after 8 episodes. TV must stay at or below target through episode {budget}.}};")
    for offset, (label, color, marker) in enumerate(METHODS.values()):
        x = offset * 24
        lines.append(rf"\node[{marker},fill={color},inner sep=1.6pt] at "
                     rf"([xshift={x + 1}mm,yshift=11mm]group c1r1.north west) {{}};")
        lines.append(rf"\node[anchor=west,font=\small\bfseries,text={color}] at "
                     rf"([xshift={x + 3}mm,yshift=11mm]group c1r1.north west) {{{label}}};")
    lines.append(r"\node[anchor=west,font=\footnotesize,text=muted] at "
                 r"([xshift=96mm,yshift=11mm]group c1r1.north west) "
                 r"{$>$ indicates a target was not reached by the budget cap.};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines).lstrip() + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    args.figure.write_text(figure(args.summary))


if __name__ == "__main__":
    main()
