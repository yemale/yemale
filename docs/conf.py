"""Build the API reference from the installed package's docstrings."""

from importlib.metadata import version as package_version

project = "yemale"
author = "Eugene Ndiaye"
release = package_version(project)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autoclass_content = "class"
myst_enable_extensions = ["dollarmath"]
myst_fence_as_directive = ["math"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "README.md"]

html_theme = "sphinx_book_theme"
html_title = "yemale"
html_baseurl = "https://yemale.github.io/"
html_context = {"default_mode": "auto"}
html_theme_options = {
    "show_toc_level": 2,
    "footer_content_items": [],
    "use_download_button": False,
    "repository_url": "https://github.com/yemale/yemale",
    "use_repository_button": True,
}
html_static_path = ["_static"]
html_css_files = ["yemale.css"]
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False


def hide_factory_constructor(
    app, what, name, obj, options, signature, return_annotation
):
    if what == "class" and name.rsplit(".", 1)[-1] in {
        "Transport",
        "QuantileRegion",
        "DensityRegion",
        "SmoothMap",
    }:
        return "", None


def setup(app):
    app.connect("autodoc-process-signature", hide_factory_constructor)
