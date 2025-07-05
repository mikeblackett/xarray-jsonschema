DataArray Schemas
=================

The :py:class:`~.DataArraySchema` object enables the specification of schema that
describe :py:class:`xarray.DataArray` objects. The
:py:class:`~.DataArraySchema` object consists of a set of optional schema components
that prescribe the key properties of :py:class:`xarray.DataArray` objects.

Currently, the following components are available:

- ``attrs``: an :py:class:`~.AttrsSchema` object describing the array's attributes;
- ``chunks``: a :py:class:`~.ChunksSchema` object describing the array's chunking;
- ``coords``: a :py:class:`~.CoordsSchema` object describing the array's coordinates;
- ``dims``: a :py:class:`~.DimsSchema` object describing the array's dimensions;
- ``dtype``: a :py:class:`~.DTypeSchema` object describing the array's data type;
- ``shape``: a :py:class:`~.ShapeSchema` object describing the array's shape;
- ``name``: a :py:class:`~.NameSchema` object describing the array's name.

In addition to being used in the context of a ``DataArraySchema`` all of the
schema components can be used as standalone validation objects.

To create a :py:class:`~.DataArraySchema` object, simply pass the
components you want to validate as arguments to the constructor:

..  testsetup::

    >>> import numpy as np
    >>> import xarray_jsonschema as xrjs

..  doctest::

    >>> dims = xrjs.DimsSchema(['x', 'y'])
    >>> shape = xrjs.ShapeSchema([4, 5])
    >>> dtype = xrjs.DTypeSchema(np.int16)
    >>> name = xrjs.NameSchema('foo')
    >>> da_schema = xrjs.DataArraySchema(dims=dims, shape=shape, dtype=dtype, name=name)

Omitting a component, or setting it to ``None``, will cause the
component to be ignored during validation.

To validate an :py:class:`xarray.DataArray` object against the schema, you can
use the :py:meth:`~.DataArraySchema.validate` method, or alternatively, call the schema
object directly (which calls :py:meth:`~.DataArraySchema.validate` under the hood):

..  doctest::

    >>> da = xr.DataArray(np.arange(20, dtype=np.int16).reshape(4, 5), dims=['x', 'y'])

    >>> da_schema.validate(da)
    >>> da_schema(da)

If validation is successful, the :py:meth:`~.DataArraySchema.validate` method
returns ``None``. If validation fails, a :py:class:`jsonschema.exceptions.ValidationError`
exception is raised.

NameSchema
---------------
Use the :py:class:`~.NameSchema` component to impose constraints on the name of an
array.

..  doctest::

    >>> da = xr.DataArray(np.random.randn(4), dims=['x'], name='jsonschema')

    >>> schema = xrjs.DataArraySchema(name=xrjs.NameSchema('jsonschema'))
    >>> schema.validate(da)


Sequence matching
~~~~~~~~~~~~~~~~~
Want to validate that the name comes from a set of acceptable values?
Pass a sequence of strings to the ``name`` argument:

..  doctest::

    >>> class NAMES(StrEnum):
            A = 'a'
            B = 'b'
            C = 'c'
    >>> da = xr.DataArray(np.random.randn(4), dims=['x'], name='a')

    >>> schema = xrjs.DataArraySchema(name=xrjs.NameSchema(['tas', 'tasmax', 'tasmin']))
    >>> schema.validate(da)

Regex pattern matching
~~~~~~~~~~~~~~~~~~~~~~
Want to validate that the name matches a certain pattern?
Set the ``regex`` argument to ``True`` and the ``name`` argument will be
interpreted as a regular expression pattern.

..  doctest::

    >>> da = xr.DataArray(np.random.randn(4, 5), dims=['x', 'y'], name='num_var_1')

    >>> schema = xrjs.DataArraySchema(name=xrjs.NameSchema(r'num_var_.+', regex=True))
    >>> schema.validate(da)

Length constraints
~~~~~~~~~~~~~~~~~~
Perhaps you need to verify that your array names satisfy a certain length
constraint? Set the `min_length` and/or `max_length` arguments as required:

.. doctest::

    >>> da = xr.DataArray(np.random.randn(4, 5), dims=['x', 'y'], name='foo')

    >>> schema = xrjs.DataArraySchema(name=xrjs.NameSchema(min_length=3, max_length=5))
    >>> schema.validate(da)

DimsSchema validation
---------------
Use the :py:class:`~.DimsSchema` component to impose constraints on the names of the
dimensions of an array.

.. doctest::

    >>> da = xr.DataArray(np.random.randn(4, 5), dims=['x', 'y'])

    >>> schema = xrjs.DataArraySchema(dims=xrjs.DimsSchema(['x', 'y']))
    >>> schema.validate(da)

Advanced matching
~~~~~~~~~~~~~~~~~
Want to create complex or generic matching rules?
You can pass ``NameSchema`` objects to the ``dims`` argument:

.. doctest::

    >>> da = xr.DataArray(np.random.randn(4, 5, 6), dims=['x', 'y1', 'anything'])

    >>> schema = xrjs.DataArraySchema(
            dims=xrjs.DimsSchema(
                [
                    xrjs.NameSchema(['x', 'y', 'z']),
                    xrjs.NameSchema(r'$y{0-9}+', regex=True),
                    xrjs.NameSchema(), # will match anything...
                ]
            ),
        )
    >>> schema.validate(da)

Partial matching
~~~~~~~~~~~~~~~~
Only care that the array's dimensions contain a certain name?
Pass a string or a ``NameSchema`` object to the ``contains`` argument:

.. doctest::

    >>> da = xr.DataArray(np.random.randn(7, 8, 9), dims=['time', 'lat', 'lon'])

    >>> schema = xrjs.DataArraySchema(
            dims=xrjs.DimsSchema(contains='time'),
        )
    >>> schema.validate(da)

Length constraints
~~~~~~~~~~~~~~~~~~
Only care about the number of dimensions? Set the `min_length` and/or `max_length`
arguments as required:

.. doctest::

    >>> da = xr.DataArray(np.random.randn(4, 5), dims=['x', 'y'])

    >>> schema = xrjs.DataArraySchema(dims=xrjs.DimsSchema(max_length=2))
    >>> schema.validate(da)
