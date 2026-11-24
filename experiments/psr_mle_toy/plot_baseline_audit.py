"""Build a three-method oracle-call / episode-cost figure as PGFPlots."""
from argparse import ArgumentParser
import csv
from pathlib import Path


def plot(oracle_csv, psr_ucb_csv):
    with oracle_csv.open() as stream:
        oracle_rows = list(csv.DictReader(stream))
    with psr_ucb_csv.open() as stream:
        ucb_rows = list(csv.DictReader(stream))
    methods = {name: [] for name in ("omle", "ours", "psr_ucb")}
    for name in ("omle", "ours"):
        methods[name] = sorted((r for r in oracle_rows if r["method"] == name),
                               key=lambda r: -float(r["epsilon"]))
    for eps in sorted({float(r["epsilon"]) for r in ucb_rows}, reverse=True):
        pair = [r for r in ucb_rows if float(r["epsilon"]) == eps]
        # Both truth orientations have identical median costs by construction.
        methods["psr_ucb"].append({
            "epsilon": str(eps),
            "joint_calls": str((float(pair[0]["median_mle_calls"])
                                + float(pair[1]["median_mle_calls"])) / 2),
            "episodes": str((float(pair[0]["median_episodes"])
                             + float(pair[1]["median_episodes"])) / 2),
        })

    coords = {}
    for method, metric in (("omle", "calls"), ("ours", "calls"), ("psr_ucb", "calls"),
                          ("omle", "episodes"), ("ours", "episodes"), ("psr_ucb", "episodes")):
        points = []
        for index, row in enumerate(methods[method]):
            key = "joint_calls" if method in ("omle", "psr_ucb") and metric == "calls" else metric
            if method == "ours" and metric == "calls":
                value = float(row["ordinary_mle_calls"])
            else:
                value = float(row[key])
            points.append(f"({index},{value:.14g})")
        coords[method, metric] = " ".join(points)
    epsilons = [r["epsilon"] for r in methods["ours"]]
    labels = ",".join(f"{float(x):g}" for x in epsilons)
    colors = {"omle": "#773588", "ours": "#00747C", "psr_ucb": "#D06B24"}
    marks = {"omle": "o", "ours": "square*", "psr_ucb": "triangle*"}
    lines = [r"""
\documentclass[border=8pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{omleline}{HTML}{773588}
\definecolor{oursline}{HTML}{00747C}
\definecolor{ucbline}{HTML}{D06B24}
\definecolor{ink}{HTML}{17242D}
\definecolor{muted}{HTML}{4A5C67}
\definecolor{gridline}{HTML}{E3EAED}
\begin{document}
\begin{tikzpicture}[text=ink]
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=1.4cm},
  width=7.4cm,height=5.2cm,scale only axis,
  xmin=-.18,xmax=3.18,xtick={0,1,2,3},ymode=log,
  axis lines*=left,axis line style={muted!80},
  tick align=outside,tick label style={font=\footnotesize,text=muted},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},xlabel={Accuracy parameter $\epsilon$},
  clip=false,
]"""]
    for metric, title, limits in (
        ("calls", "Measured optimization calls", "ymin=1,ymax=30000,ytick={1,10,100,1000,10000}"),
        ("episodes", "Episodes to the common TV target", "ymin=10,ymax=1e14,ytick={10,1e4,1e7,1e10,1e13}"),
    ):
        lines.append(rf"\nextgroupplot[title={{{title}}},xticklabels={{{labels}}},{limits}]")
        for method, color, mark in (("omle", "omleline", marks["omle"]),
                                    ("ours", "oursline", marks["ours"]),
                                    ("psr_ucb", "ucbline", marks["psr_ucb"])):
            points = coords[method, metric]
            lines.append(rf"\addplot[{color},thick,mark={mark},mark size=2pt] coordinates {{{points}}};")
            for index, row in enumerate(methods[method]):
                key = "joint_calls" if method in ("omle", "psr_ucb") and metric == "calls" else metric
                value = (float(row["ordinary_mle_calls"]) if method == "ours" and metric == "calls"
                         else float(row[key]))
                if metric == "calls":
                    lines.append(rf"\node[anchor=south,font=\scriptsize,text={color}] at (axis cs:{index},{value:.10g}) {{{value:g}}};")
    lines.extend([
        r"\end{groupplot}",
        r"\node[anchor=west,inner sep=0pt,font=\bfseries\Large] at ([yshift=27mm]group c1r1.north west) {Accuracy and computation cost on the routing PSR};",
        r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at ([yshift=20mm]group c1r1.north west) {All methods evaluated against max-policy TV $\leq\epsilon/4$.};",
        r"\node[anchor=west,inner sep=0pt,text=omleline,font=\small\bfseries] at ([yshift=12mm]group c1r1.north west) {omle};",
        r"\node[anchor=west,inner sep=0pt,text=oursline,font=\small\bfseries] at ([xshift=35mm,yshift=12mm]group c1r1.north west) {ours};",
        r"\node[anchor=west,inner sep=0pt,text=ucbline,font=\small\bfseries] at ([xshift=75mm,yshift=12mm]group c1r1.north west) {psr\_ucb};",
        r"\end{tikzpicture}", r"\end{document}",
    ])
    return "\n".join(lines).lstrip() + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("oracle_summary", type=Path)
    parser.add_argument("psr_ucb_summary", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    args.figure.write_text(plot(args.oracle_summary, args.psr_ucb_summary))


if __name__ == "__main__":
    main()
