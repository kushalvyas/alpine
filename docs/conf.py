# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../alpine'))

project = 'Alpine'
# copyright = '2025, Kushal Vyas, Vishwanath Saragadam, Ashok Veeraraghavan, Guha Balakrishnan'
copyright = "2025, Alpine."
author = 'Kushal Vyas, Vishwanath Saragadam, Ashok Veeraraghavan, Guha Balakrishnan'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 
              'sphinx_design',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.todo',
              'sphinx.ext.githubpages',
              'sphinx.ext.mathjax',
              'sphinxcontrib.bibtex',
            #   'numpydoc.numpydoc',
              ]

myst_enable_extensions = [
    "colon_fence",  # Needed for ::: directives
]
bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/alpine_logo.png"
html_theme_options = {
    "show_prev_next": False,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/kushalvyas/alpine",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ]
}
html_context = {
  "default_mode": "light",
}