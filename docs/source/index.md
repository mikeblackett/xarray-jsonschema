# {{project}}

JSON Schema validation for [Xarray](https://xarray.dev/) objects.

**Version:** {{version}}

`xarray-jsonschema` is a re-implementation of [xarray-schema](https://xarray-schema.readthedocs.io/en/latest) that uses [JSON Schema](https://json-schema.org/) to validate [xarray](https://docs.xarray.dev/en/stable/) objects.

:::{warning}
This project is under active development. Frequent and breaking changes are expected.
:::

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

user_guide
examples/index
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

`xarray-jsonschema`'s API is modelled after [xarray-schema](https://xarray-schema.readthedocs.io/en/latest/).

> Xarray-schema provides a simple class-based API for defining schemas and validating Xarray objects (and their components).  
> -- [xarray-schema documentation](https://xarray-schema.readthedocs.io/en/latest/quickstart.html)

The main change is that the validation engine has been replaced with a [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/) Python implementation of the [JSON Schema](https://json-schema.org/) specification.

## Features

- Define schemas using Python classes;
- Translate Python schemas to JSON Schemas;
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
from xarray_jsonschema import DataArraySchema

da = xr.DataArray(np.ones(4, dtype='i4'), dims=['x'], name='foo')

schema = DataArraySchema(dtype=np.integer, name='foo', shape=(4, ), dims=['x'])

schema.validate(da)
```

## Contributing

Contributions are encouraged! Please feel free to submit a Pull Request.

## Next steps

Check out the [User guide](user_guide).
