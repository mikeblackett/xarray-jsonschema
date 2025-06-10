###########
Get started
###########

..  currentmodule:: xarray_model


With xarray-model you can create composable and reusable validation models for
:py:class:`xarray.DataArray` and :py:class:`xarray.Dataset` objects.

..  contents:: Contents
    :local:
    :backlinks: none
    :depth: 2

Validating DataArrays
=====================

The :py:class:`~.DataArrayModel` enables the specification of schema that
verify the key properties of an :py:class:`xarray.DataArray` object. The
:py:class:`~.DataArrayModel` object consists of a set of optional components
that specify the validation rules for the data array's key properties.

Currently, the following components are available:

- :py:class:`~.Attrs`
- :py:class:`~.Chunks`
- :py:class:`~.Coords`
- :py:class:`~.Dims`
- :py:class:`~.DType`
- :py:class:`~.Shape`
- :py:class:`~.Name`

You can create a :py:class:`~.DataArrayModel` object by specifying the
components you want to include:

..  testsetup::

    >>> import numpy as np
    >>> import xarray_model as xrm

..  doctest::

    >>> dims = xrm.Dims(['x', 'y'])
    >>> shape = xrm.Shape([4, 5])
    >>> dtype = xrm.DType(np.int16)
    >>> name = xrm.Name('foo')
    >>> da_model = xrm.DataArrayModel(dims=dims, shape=shape, dtype=dtype, name=name)

You can then use the :py:meth:`~.DataArrayModel.validate` method to validate
an :py:class:`xarray.DataArray` object:

..  doctest::

    >>> da = xr.DataArray(np.arange(20, dtype=np.int16).reshape(4, 5), dims=['x', 'y'])
    >>> model.validate(da)

If validation is successful, the :py:meth:`~.DataArrayModel.validate` method
returns ``None``. If validation fails, the :py:meth:`~.DataArrayModel.validate`
method raises a :py:class:`jsonschema.ValidationError` exception.

..  doctest::

    >>> da_model.validate(da.astype(np.int32)

Validating Datasets
=====================

The :py:class:`~.DatasetModel` enables the specification of schema that
verify the key properties of an :py:class:`xarray.Dataset` object. The
:py:class:`~.DatasetModel` object consists of a set of optional components
that specify the validation rules for the dataset's key properties.

Currently, the following components are available:

- :py:class:`~.Attrs`
- :py:class:`~.Coords`
- :py:class:`~.DataVars`

You can create a :py:class:`~.DatasetModel` object by specifying the
components you want to include:

..  doctest::

    >>> data_vars = xrm.DataVars(data_vars=[da_model])
    >>> ds_model = xrm.DatasetModel(data_vars=data_vars)

You can then use the :py:meth:`~.DatasetModel.validate` method to validate
an :py:class:`xarray.DataArray` object:

..  doctest:: data_array_model

    >>> ds = da.to_dataset()
    >>> ds_model.validate(ds)

If validation is successful, the :py:meth:`~.DatasetModel.validate` method
returns ``None``. If validation fails, the :py:meth:`~.DatasetModel.validate`
method raises a :py:class:`jsonschema.ValidationError` exception.

Next steps
==========
The different components of the :py:class:`~.DataArrayModel` and
:py:class:`~.DatasetModel` objects have a range of options to specify more
specific or generic validation schema. These components can be combined to tailor
the validation rules to your needs. You can read about the different components
in the :ref:`user-guide-label` section.
