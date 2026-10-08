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
html_context = {"default_mode": "auto"}
html_theme_options = {
    "repository_url": "https://github.com/yemale/yemale",
    "use_repository_button": True,
    "navbar_persistent": [],
    "show_toc_level": 2,
    "footer_content_items": [],
    "use_download_button": False,
    "use_fullscreen_button": False,
}
html_static_path = ["_static"]
html_css_files = ["yemale.css"]
# Hide .rst/.md source links; viewcode still links to Python implementations.
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False


def hide_factory_constructor(
    app, what, name, obj, options, signature, return_annotation
):
    """Hide internal constructor signatures for objects created by public factories."""
    if what == "class" and name.rsplit(".", 1)[-1] in {
        "Transport",
        "QuantileRegion",
        "Region",
        "DensityRegion",
        "SmoothMap",
        "Conformalizer",
        "CPD",
        "Extended",
    }:
        return "", None


def fix_conformal_backlinks(app, pagename, templatename, context, doctree):
    """Keep each object's public anchor despite viewcode's shared module alias."""
    if not pagename.startswith("_modules/yemale/conformal/"):
        return
    for name, _, _, docname, anchor, _ in app.env.get_domain("py").get_objects():
        if name.startswith("yemale.conformal."):
            url = app.builder.get_relative_uri(pagename, docname)
            short_name = name.replace("yemale.conformal.", "yemale.", 1)
            context["body"] = context["body"].replace(
                f'href="{url}#{short_name}"', f'href="{url}#{anchor}"'
            )


def setup(app):
    app.connect("autodoc-process-signature", hide_factory_constructor)
    app.connect("html-page-context", fix_conformal_backlinks)
