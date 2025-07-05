###############
Dataset Schemas
###############

The :py:class:`~.DatasetSchema` object enables the specification of schema that
describe :py:class:`xarray.Dataset` objects. The
:py:class:`~.DatasetSchema` object consists of a set of optional schema components
that prescribe the key properties of :py:class:`xarray.Dataset` objects.

Currently, the following components are available:

- ``data_vars``: a :py:class:`~.DataVarsSchema` object describing the dataset's data variables;
- ``coords``: a :py:class:`~.CoordsSchema` object describing the dataset's coordinates;
- ``attrs``: a :py:class:`~.AttrsSchema` object describing the dataset's attributes.

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
