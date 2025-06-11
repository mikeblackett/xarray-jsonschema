..  _index:
..  currentmodule:: xarray_jsonschema

#################################################
JSON Schema powered validation for xarray objects
#################################################

:py:mod:`xarray-jsonschema` v\ |version|


Xarray-jsonschema is an open source project that provides a simple API for
validating xarray objects against `JSON Schema <https://json-schema.org/>`_.
This project was heavily inspired by the
`xarray-schema <https://github.com/xarray-contrib/xarray-schema>`_ project.

..  warning::

    This project is under active development. Frequent and breaking changes are expected.

.. toctree::
    :maxdepth: 1

    User Guide <user-guide/index>
    API <api>

Quick start
===========
.. code-block:: python

    import xarray as xr
    import xarray_jsonschema as xrjs

    da_model = xrjs.DataArraySchema(
        dims=['x', 'y'],
        shape=[4, 5],
        dtype=np.int16,
        name='foo',
    )

    da = xr.DataArray(np.arange(20, dtype=np.int16).reshape(4, 5), dims=['x', 'y'])
    da_model.validate(da)
