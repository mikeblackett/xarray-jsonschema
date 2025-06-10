# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import xarray_model

project = 'xarray-model'
copyright = '2025, Mike Blackett'
author = 'Mike Blackett'
version = xarray_model.__version__
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

# AutoDoc config
autosummary_generate = True
autodoc_typehints = 'description'


source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
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
html_title = 'xarray-model'
html_theme = 'pydata_sphinx_theme'
