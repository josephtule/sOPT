window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"],
  },
  tex: {
    packages: {
      "[+]": ["boldsymbol"],
    },
    macros: {
      R: "\\mathbb{R}",
      E: "\\mathbb{E}",
      N: "\\mathbb{N}",
      Z: "\\mathbb{Z}",
      Q: "\\mathbb{Q}",
      C: "\\mathbb{C}",
      vecb: ["\\boldsymbol{#1}", 1],
      unitv: ["\\hat{\\boldsymbol{#1}}", 1],
      norm: ["\\left\\lVert#1\\right\\rVert", 1],
      abs: ["\\left\\lvert#1\\right\\rvert", 1],
      tbf: ["\\textbf{#1}", 1],
      Span: "\\operatorname{span}",
      rank: "\\operatorname{rank}",
      diag: "\\operatorname{diag}",
      image: "\\operatorname{Im}",
      bmat: ["\\begin{bmatrix}#1\\end{bmatrix}", 1],
      pmat: ["\\begin{pmatrix}#1\\end{pmatrix}", 1],
      cmat: ["\\begin{Bmatrix}#1\\end{Bmatrix}", 1],
      vmat: ["\\begin{vmatrix}#1\\end{vmatrix}", 1],
      vvmat: ["\\begin{Vmatrix}#1\\end{Vmatrix}", 1],
      matt: ["\\begin{bmatrix}#1\\end{bmatrix}", 1],
    },
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
