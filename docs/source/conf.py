# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import xarray_jsonschema

project = 'xarray-jsonschema'
copyright = '2025, Mike Blackett'
author = 'Mike Blackett'
version = xarray_jsonschema.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

# AutoDoc configuration
autosummary_generate = True
autodoc_typehints = 'none'

# Napoleon configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True

source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'dask': ('https://docs.dask.org/en/latest', None),
    'hypothesis': ('https://hypothesis.readthedocs.io/en/latest/', None),
    'jsonschema': (
        'https://python-jsonschema.readthedocs.io/en/stable/',
        None,
    ),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', '/')
html_theme = 'pydata_sphinx_theme'
html_title = 'xarray-jsonschema'
html_context = {
    'github_user': 'mikeblackett',
    'github_repo': 'xarray-jsonschema',
    'github_version': 'main',
    'doc_path': 'doc',
}
html_theme_options = {
    'github_url': 'https://github.com/mikeblackett/xarray-jsonschema',
    'use_edit_page_button': False,
    'header_links_before_dropdown': 8,
    'navbar_align': 'left',
    'footer_center': ['last-updated'],
}
