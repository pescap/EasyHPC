project = "EasyHPC"
author = "EasyHPC contributors"
copyright = "2022-2026, EasyHPC contributors"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

html_theme = "furo"
html_title = "EasyHPC"
html_theme_options = {
    "source_repository": "https://github.com/pescap/EasyHPC/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

copybutton_prompt_text = "$ "
