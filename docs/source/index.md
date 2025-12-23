# {{project}}

[JSON Schema](https://json-schema.org/) validation for [Xarray](https://xarray.dev/) objects.

**Version:** {{version}}

:::{warning}
This project is under active development. Frequent and breaking changes are expected.
:::

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

user_guide
api
:::

## Contents

- [Overview](#overview)
- [Features](#features)
- [Motivation](#motivation)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Next steps](#next-steps)

## Overview

`xarray-jsonschema` provides a declarative API for defining [JSON Schema](https://json-schema.org/) and validating [xarray](https://xarray.dev/) objects.

## Features

- Define models using Python classes;
- Translate models to JSON Schemas;
- Validate xarray objects with JSON Schema;

## Motivation

I needed to validate xarray objects produced in an ETL pipeline, but none of the existing tools had all the features that I required. I took inspiration from the [xarray-schema](https://github.com/xarray-contrib/xarray-schema) and [xarray-validate](https://github.com/leroyvn/xarray-validate) packages and created `xarray-jsonschema`.

## Installation

`xarray-jsonschema` is still in development.

You can install it from source:

```bash
pip install git+https://github.com/mikeblackett/xarray-jsonschema
```

Or for development:

```bash
mkdir xarray-jsonschema
cd xarray-jsonschema
git clone https://github.com/mikeblackett/xarray-jsonschema
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
import xarray as xr

from xarray_jsonschema import DataArrayModel

da = xr.DataArray(np.ones(4, dtype='i4'), dims=['x'], name='foo')

model = DataArrayModel(dtype=np.int32, name='foo', shape=(4,), dims=['x'])

model.validate(da)
```

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [User guide](user_guide).
