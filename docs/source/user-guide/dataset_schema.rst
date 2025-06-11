###############
Dataset Schemas
###############

The :py:class:`~.DatasetSchema` object enables the specification of schema that
prescribe the key properties of :py:class:`xarray.Dataset` objects. The :py:class:`~.DatasetSchema`
object consists of a set of optional components that specify properties of the dataset.

Currently, the following components are available:

- :py:class:`~.AttrsSchema`
- :py:class:`~.CoordsSchema`
- :py:class:`~.DataVarsSchema`

Creating a dataset schema
-------------------------

To create a :py:class:`~.DatasetSchema` object, simply include the
components you want to validate:

..  doctest::

    >>> data_vars = xarray_jsonschema.DataVarsSchema(data_vars=[da_model])
    >>> ds_schema = xarray_jsonschema.DatasetSchema(data_vars=data_vars)

Validating a dataset
--------------------

To validate an :py:class:`xarray.Dataset` object against the schema, you can
use the :py:meth:`~.DatasetSchema.validate` method, or alternatively, call the schema
object directly (which calls :py:meth:`~.DatasetSchema.validate` under the hood):

..  doctest::

    >>> ds = da.to_dataset()
    >>> ds_schema.validate(ds)
    >>> ds_schema(ds)

If validation is successful, the :py:meth:`~.DatasetSchema.validate` method
returns ``None``. If validation fails, a :py:class:`jsonschema.exceptions.ValidationError`
exception is raised.
