Schema components
=================
..  currentmodule:: xarray_jsonschema

The :py:class:`DataArraySchema`

NameSchema
----------
The :py:class:`Name` schema can be used to specify constraints on names of
``DataArray``s and their dimensions.

Regular expressions
~~~~~~~~~~~~~~~~~~~
Want to verify that a name matches a regular expression? Pass a regular
expression pattern to the ``name`` parameter:

.. code-block:: python
    from enum import StrEnum

    import xarray as xr
    from xarray_jsonschema import NameSchema

    class NAMES(StrEnum):
        T = "time"
        X = "longitude"
        Y = "latitude"

    schema = NameSchema(NAMES)

Allowed values
~~~~~~~~~~~~~~
Want to verify that a name belongs to a set of allowed values? Pass an
iterable of strings to the ``name`` parameter:

.. code-block:: python
    from enum import StrEnum

    import xarray as xr
    from xarray_jsonschema import NameSchema

    class NAMES(StrEnum):
        T = "time"
        X = "longitude"
        Y = "latitude"

    schema = NameSchema(NAMES)

    da = xr.DataArray(
        [1, 2, 3, 4],
        dims=["time"],
        name="time",
    )
    schema.validate(da.name)
