"""Plot optimization calls and episode costs under the theorem audit."""
from argparse import ArgumentParser
import csv
from pathlib import Path


def figure(source):
    with source.open() as file:
        rows = list(csv.DictReader(file))
    methods = {name: sorted((row for row in rows if row["method"] == name),
                            key=lambda row: float(row["epsilon"]), reverse=True)
               for name in ("omle", "ours")}
    labels = ",".join(row["epsilon"] for row in methods["ours"])
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
  group style={group size=2 by 1,horizontal sep=1.4cm},
  width=7.3cm,height=5.3cm,scale only axis,
  xmin=-.18,xmax=3.18,xtick={0,1,2,3},ymode=log,
  axis lines*=left,axis line style={muted!80},
  tick align=outside,tick label style={font=\footnotesize,text=muted},
  title style={font=\small\bfseries,text=ink,yshift=1mm},
  grid=major,grid style={gridline},xlabel={Accuracy parameter $\epsilon$},
  clip=false,
]"""]
    for metric, title, limits in (
        ("calls", "Optimization calls", "ymin=1,ymax=2300,ytick={1,10,100,1000},yticklabels={1,10,100,1000}"),
        ("episodes", "Environment episodes", "ymin=10,ymax=1e14,ytick={10,1e4,1e7,1e10,1e13}"),
    ):
        lines.append(rf"\nextgroupplot[title={{{title}}},xticklabels={{{labels}}},{limits}]")
        for method, color, mark in (("omle", "omleline", "o"), ("ours", "oursline", "square*")):
            key = ("joint_calls" if method == "omle" else "ordinary_mle_calls") if metric == "calls" else "episodes"
            data = [(i, float(row[key])) for i, row in enumerate(methods[method]) if row[key]]
            coordinates = " ".join(f"({i},{value:.14g})" for i, value in data)
            lines.append(rf"\addplot[{color},thick,mark={mark},mark size=2.1pt] coordinates "
                         + "{" + coordinates + "};")
            if metric == "calls":
                for i, value in data:
                    lines.append(rf"\node[anchor=south,font=\scriptsize,text={color}] at "
                                 rf"(axis cs:{i},{value:.10g}) {{{int(value)}}};")
    lines.extend([
        r"\end{groupplot}",
        r"\node[anchor=west,inner sep=0pt,font=\bfseries\Large] at ([yshift=27mm]group c1r1.north west) {Oracle calls at the required trajectory accuracy};",
        r"\node[anchor=west,inner sep=0pt,text=muted,font=\small] at ([yshift=20mm]group c1r1.north west) {TV target $\epsilon/4$. OMLE uses two episodes per round. Ours uses the appendix batches.};",
        r"\node[anchor=west,inner sep=0pt,text=omleline,font=\small] at ([yshift=12mm]group c1r1.north west) {$\circ$\enspace\textbf{omle}\quad joint model and policy searches};",
        r"\node[anchor=west,inner sep=0pt,text=oursline,font=\small] at ([xshift=78mm,yshift=12mm]group c1r1.north west) {\rule{4pt}{4pt}\enspace\textbf{ours}\quad calls to each oracle};",
        r"\end{tikzpicture}", r"\end{document}",
    ])
    return "\n".join(lines).lstrip() + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    args.figure.write_text(figure(args.summary))


if __name__ == "__main__":
    main()
