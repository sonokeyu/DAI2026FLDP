"""Create a PGFPlots figure from the equal-budget experiment summary."""
from collections import defaultdict
import csv
from pathlib import Path
import sys


STYLES = {
    "bonus_test": ("blue!75!black, densely dashed", "Stage 2 bonus + test"),
    "random_test": ("orange!85!black, densely dashdotted", "Random route + test"),
    "bonus_no_test": ("gray, densely dotted", "Stage 2 bonus only"),
    "fitted_psr": ("teal!75!black, solid", "Fitted PSR"),
    "reward_free_omle": ("purple!85!black, solid", "Reward-free OMLE"),
}


def figure(source):
    curves = defaultdict(list)
    with source.open() as file:
        for row in csv.DictReader(file):
            curves[float(row["gap"]), row["method"]].append(
                (int(row["episodes"]), float(row["mean_tv"])))
    lines = [r"""
\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=3 by 1,horizontal sep=0.95cm},
  width=5.3cm,height=5.0cm,xmin=0,xmax=256,ymin=0,
  xtick={0,64,128,192,256},xlabel={Episodes},
  tick label style={font=\footnotesize},label style={font=\small},
  scaled y ticks=false,
  yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},
  title style={font=\small},grid=major,grid style={gray!20},
  axis line style={gray!65},legend style={draw=none},
]"""]
    for gap in (0.1, 0.3, 0.6):
        options = f"title={{$\\Delta={gap}$}},ymax={gap * 0.275}"
        if gap == 0.1:
            options += (r",ylabel={Mean worst-policy TV},legend columns=3,"
                        r"legend to name=key,legend style={font=\footnotesize}")
        lines.append(r"\nextgroupplot[" + options + "]")
        for method, (style, label) in STYLES.items():
            points = " ".join(f"({n},{value:.8g})" for n, value in sorted(curves[gap, method]))
            lines.append(r"\addplot+[thick,mark=none," + style + "] coordinates {" + points + "};")
            if gap == 0.1:
                lines.append(r"\addlegendentry{" + label + "}")
    lines.extend([r"\end{groupplot}",
                  r"\path (group c1r1.south west) -- node[midway,below=1.15cm]{\pgfplotslegendfromname{key}} (group c3r1.south east);",
                  r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_comparison.py comparison_summary.csv figure.tex")
    Path(sys.argv[2]).write_text(figure(Path(sys.argv[1])))
