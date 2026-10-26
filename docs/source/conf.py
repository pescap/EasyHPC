project = "EasyHPC"
author = "EasyHPC contributors"
copyright = "2022-2026, EasyHPC contributors"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
]

html_theme = "furo"
html_title = "EasyHPC"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/pescap/EasyHPC/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

copybutton_prompt_text = "$ "

ogp_site_url = "https://easyhpc.readthedocs.io/en/latest/"
ogp_image = "_static/easyhpc-preview.png"
ogp_image_alt = "EasyHPC: a practical guide to High Performance Computing"
ogp_custom_meta_tags = ['<meta name="twitter:card" content="summary_large_image" />']
