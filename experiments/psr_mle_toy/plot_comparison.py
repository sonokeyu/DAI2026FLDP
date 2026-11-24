"""Plot the two primary methods under equal episode budgets."""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys


METHODS = {
    "reward_free_omle": ("omle", "omleline", "densely dashed", "*"),
    "fitted_psr": ("ours", "oursline", "solid", "square*"),
}

Y_TICKS = {
    0.1: "0,0.01,0.02,0.03",
    0.3: "0,0.03,0.06,0.09",
    0.6: "0,0.05,0.10,0.15",
}


def coordinates(rows, column):
    return " ".join(f"({int(row['episodes'])},{float(row[column]):.8g})" for row in rows)


def figure(source):
    curves = {}
    with source.open() as file:
        for row in csv.DictReader(file):
            if row["method"] in METHODS:
                curves.setdefault((float(row["gap"]), row["method"]), []).append(row)
    for rows in curves.values():
        rows.sort(key=lambda row: int(row["episodes"]))
    budget = max(int(row["episodes"]) for rows in curves.values() for row in rows)

    lines = [r"""
\documentclass[border=8pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
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
  group style={group size=3 by 1,horizontal sep=1.05cm},
  width=5.2cm,height=4.5cm,scale only axis,
  xmin=0,ymin=0,
  xtick={0,64,128,192,256},
  tick align=outside,tick style={muted!75},
  tick label style={font=\footnotesize,text=muted},
  label style={font=\small,text=ink},
  scaled y ticks=false,
  yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},
  axis lines=left,axis line style={muted!80},
  xlabel={Environment episodes},
]"""]
    for panel, (gap, ticks) in enumerate(Y_TICKS.items(), start=1):
        ymax = max(float(value) for value in ticks.split(","))
        options = f"title={{$\\Delta={gap}$}},xmax={budget},ymax={ymax * 1.025},ytick={{{ticks}}}"
        if panel == 1:
            options += r",ylabel={Mean worst-policy TV}"
        lines.append(r"\nextgroupplot[" + options + "]")

        for method, (name, color, _, _) in METHODS.items():
            rows = curves[gap, method]
            for side in ("low", "high"):
                path = f"{name}{panel}{side}"
                points = coordinates(rows, f"bootstrap_{side}")
                lines.append(rf"\addplot[draw=none,name path={path}] coordinates {{{points}}};")
            lines.append(rf"\addplot[{color},fill opacity=0.13,draw=none] "
                         rf"fill between[of={name}{panel}low and {name}{panel}high];")

        for method, (_, color, style, mark) in METHODS.items():
            rows = curves[gap, method]
            assert int(rows[-1]["episodes"]) == budget
            lines.append(rf"\addplot[{color},thick,{style},mark=none] coordinates "
                         + "{" + coordinates(rows, "mean_tv") + "};")
            final = float(rows[-1]["mean_tv"])
            lines.append(rf"\addplot[only marks,mark={mark},mark size=2pt,{color},clip marker paths=false] "
                         rf"coordinates {{({budget},{final:.8g})}};")

    lines.append(r"\end{groupplot}")
    lines.append(r"\node[anchor=west,inner sep=0pt,font=\bfseries\Large] at "
                 r"([yshift=27mm]group c1r1.north west) {Worst-policy TV at equal episode budgets};")
    lines.append(r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at "
                 r"([yshift=20mm]group c1r1.north west) "
                 r"{Mean and pointwise 95\% bootstrap intervals. Lower TV is better. Panel y-axis scales differ.};")

    first_gap = next(iter(Y_TICKS))
    for method, (name, color, style, _) in METHODS.items():
        n_updates = int(curves[first_gap, method][-1]["updates"])
        x_shift = 0 if name == "omle" else 70
        lines.append(rf"\draw[{color},very thick,{style}] "
                     rf"([xshift={x_shift}mm,yshift=12mm]group c1r1.north west) -- ++(7mm,0);")
        lines.append(rf"\node[anchor=west,font=\small] at "
                     rf"([xshift={x_shift + 8}mm,yshift=12mm]group c1r1.north west) "
                     rf"{{\textbf{{{name}}} \enspace \textcolor{{muted}}{{{n_updates} updates by episode {budget}}}}};")

    for panel, gap in enumerate(Y_TICKS, start=1):
        lines.append(rf"\node[anchor=north,font=\footnotesize,text=muted] at "
                     rf"([yshift=-11mm]group c{panel}r1.south) {{TV at {budget} episodes}};")
        values = []
        for method, (name, color, _, _) in METHODS.items():
            final = Decimal(curves[gap, method][-1]["mean_tv"]).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP)
            values.append(rf"\textcolor{{{color}}}{{\textbf{{{name}}} {final:.4f}}}")
        lines.append(rf"\node[anchor=north,font=\small] at "
                     rf"([yshift=-16mm]group c{panel}r1.south) "
                     + "{" + r"\qquad ".join(values) + "};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines).lstrip() + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_comparison.py comparison_summary.csv figure.tex")
    Path(sys.argv[2]).write_text(figure(Path(sys.argv[1])))
