from __future__ import annotations


def pre_js_render(soup, logger):
    body = soup.find("body")
    if body is None:
        return soup

    if soup.find("script", attrs={"id": "sopt-mathjax-config"}) is not None:
        return soup

    # mkdocs-to-pdf drops local script sources when rendering JS.
    # Inject MathJax config inline so custom macros are available in PDF builds.
    cfg = soup.new_tag("script", id="sopt-mathjax-config")
    cfg.string = r"""
window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    packages: {
      "[+]": ["boldsymbol"]
    },
    macros: {
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      Z: "\\mathbb{Z}",
      Q: "\\mathbb{Q}",
      C: "\\mathbb{C}",
      vecb: ["\\boldsymbol{#1}", 1],
      unitv: ["\\hat{\\boldsymbol{#1}}", 1],
      norm: ["\\left\\lVert#1\\right\\rVert", 1],
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
      matt: ["\\begin{bmatrix}#1\\end{bmatrix}", 1]
    },
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
"""
    body.insert(0, cfg)

    # Fallback: define macros from inside TeX itself so they are available even
    # if external/local JS config handling is altered by mkdocs-to-pdf internals.
    if soup.find(id="sopt-mathjax-macro-bootstrap") is None:
        boot = soup.new_tag(
            "div",
            id="sopt-mathjax-macro-bootstrap",
            attrs={"class": "arithmatex"},
        )
        boot["style"] = "height:0;overflow:hidden;opacity:0;pointer-events:none;"
        boot.string = (
            r"\("
            r"\require{boldsymbol}"
            r"\gdef\R{\mathbb{R}}"
            r"\gdef\N{\mathbb{N}}"
            r"\gdef\Z{\mathbb{Z}}"
            r"\gdef\Q{\mathbb{Q}}"
            r"\gdef\C{\mathbb{C}}"
            r"\gdef\vecb#1{\boldsymbol{#1}}"
            r"\gdef\unitv#1{\hat{\boldsymbol{#1}}}"
            r"\gdef\norm#1{\left\lVert#1\right\rVert}"
            r"\gdef\tbf#1{\textbf{#1}}"
            r"\gdef\Span{\operatorname{span}}"
            r"\gdef\rank{\operatorname{rank}}"
            r"\gdef\diag{\operatorname{diag}}"
            r"\gdef\image{\operatorname{Im}}"
            r"\gdef\bmat#1{\begin{bmatrix}#1\end{bmatrix}}"
            r"\gdef\pmat#1{\begin{pmatrix}#1\end{pmatrix}}"
            r"\gdef\cmat#1{\begin{Bmatrix}#1\end{Bmatrix}}"
            r"\gdef\vmat#1{\begin{vmatrix}#1\end{vmatrix}}"
            r"\gdef\vvmat#1{\begin{Vmatrix}#1\end{Vmatrix}}"
            r"\gdef\matt#1{\begin{bmatrix}#1\end{bmatrix}}"
            r"\)"
        )
        body.insert(1, boot)
    return soup
