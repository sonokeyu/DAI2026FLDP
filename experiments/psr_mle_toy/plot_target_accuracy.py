"""Plot episode and update counts at common sustained TV targets."""
from collections import defaultdict
import csv
from pathlib import Path
import sys


METHODS = {
    "omle": ("omleline", "o", 0.14),
    "ours": ("oursline", "square*", -0.14),
}


def figure(source):
    rows = defaultdict(dict)
    with source.open() as file:
        for row in csv.DictReader(file):
            if row["condition"] in METHODS:
                rows[float(row["gap"]), float(row["target_tv"])][row["condition"]] = row
    targets = sorted(rows, key=lambda key: (key[0], -key[1]))
    slots = [len(targets) - i + (len(targets) - i - 1) // 3 * 0.7 for i in range(len(targets))]
    budget = int(next(iter(rows.values()))["ours"]["budget"])
    heights = ",".join(f"{y:.2f}" for y in slots)
    labels = ",".join(rf"{{$\Delta={gap:g}$\quad {target:.3f}}}" for gap, target in targets)

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
  group style={group size=2 by 1,horizontal sep=1.3cm},
  width=7.3cm,height=7.3cm,scale only axis,
  clip=false,
  xmode=log,log basis x=2,
  ymin=0.4,ymax=11.0,
  tick align=outside,tick style={muted!75},
  tick label style={font=\footnotesize,text=muted},
  label style={font=\small,text=ink},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},
  axis lines*=left,axis line style={muted!80},
  log ticks with fixed point,
]"""]
    axes = (
        ("updates", "updates_at_budget", "Model updates", 1, budget // 2),
        ("episodes", "budget", "Environment episodes", 8, budget),
    )
    for metric, limit, title, xmin, xmax in axes:
        ticks = ",".join(str(2 ** k) for k in range(xmin.bit_length() - 1, xmax.bit_length()))
        options = (f"title={{{title}}},xmin={xmin * 0.9:g},xmax={xmax * 1.75:g},"
                   f"ytick={{{heights}}},xtick={{{ticks}}},xticklabels={{{ticks}}}")
        if metric == "updates":
            options += rf",yticklabels={{{labels}}},ylabel={{Gap and target TV}}"
        else:
            options += r",yticklabels={}"
        lines.append(r"\nextgroupplot[" + options + "]")
        for key, y in zip(targets, slots):
            for method, (color, marker, dy) in METHODS.items():
                row = rows[key][method]
                reached = row["reached"] == "True"
                value = int(row[metric] if reached else row[limit])
                yi = y + dy
                annotation = str(value) if reached else rf"$>{{}}{value}$"
                if not reached:
                    marker = "triangle"
                lines.append(rf"\addplot[only marks,mark={marker},mark size=2pt,{color}] "
                             rf"coordinates {{({value},{yi:.3f})}};")
                lines.append(rf"\node[anchor=west,font=\scriptsize,text={color}] "
                             rf"at (axis cs:{value},{yi:.3f}) {{{annotation}}};")
    lines.append(r"\end{groupplot}")
    lines.append(r"\node[anchor=west,inner sep=0pt,font=\bfseries\Large] at "
                 r"([yshift=26mm]group c1r1.north west) {Cost to reach the same mean TV};")
    lines.append(r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at "
                 r"([yshift=19mm]group c1r1.north west) "
                 rf"{{First checkpoint that stays at or below target through episode {budget}.}};")
    lines.append(r"\node[anchor=west,font=\small,text=omleline] at "
                 r"([xshift=-1mm,yshift=11mm]group c1r1.north west) {"
                 r"$\circ$\enspace\textbf{omle}};")
    lines.append(r"\node[anchor=west,font=\small,text=oursline] at "
                 r"([xshift=22mm,yshift=11mm]group c1r1.north west) {"
                 r"\rule{4pt}{4pt}\enspace\textbf{ours}};")
    lines.append(r"\node[anchor=west,font=\footnotesize,text=muted] at "
                 r"([xshift=44mm,yshift=11mm]group c1r1.north west) {"
                 r"$>$ indicates a target was not reached by the budget cap.};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines).lstrip() + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_target_accuracy.py target_summary.csv figure.tex")
    Path(sys.argv[2]).write_text(figure(Path(sys.argv[1])))
