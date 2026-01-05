# xarray-jsonschema

> [!WARNING]  
> This package is no longer being developed. Take a look at [xarray-specs](https://github.com/mikeblackett/xarray-specs) as an alternative.

JSON Schema validation for Xarray objects.

`xarray-jsonschema` is a [JSON Schema](https://json-schema.org/)-powered validation library for [xarray](https://xarray.dev/) objects.

## Motivation

I needed to validate xarray objects produced in an ETL pipeline, but none of the existing tools had all the features that I required. I took inspiration from the [xarray-schema](https://github.com/xarray-contrib/xarray-schema) package and created `xarray-jsonschema`.

## Installation

`xarray-jsonschema` is still in development.

You can install it from source:

```bash
pip install git+https://github.com/mikeblackett/xarray-jsonschema
```

Or for development, using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mikeblackett/xarray-jsonschema
uv sync --dev
```

## Quick start

```python
import numpy as np
import xarray as xr
from xarray_jsonschema import DataArrayModel

da = xr.DataArray(
    np.random.random(10),
    dims=['x'],
    name='foo',
)
model = DataArrayModel(
    dtype='float64',
    dims=['x'],
    name='foo',
)
model.validate(da)
```

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [docs](https://xarray-jsonschema.readthedocs.io/en/latest/).
