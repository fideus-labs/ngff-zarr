<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ⚕️ Insight Toolkit (ITK)

Interoperability is available with the [Insight Toolkit (ITK)](https://itk.org).

Bidirectional type conversion that preserves spatial metadata is available with
`itk_image_to_ngff_image` and `ngff_image_to_itk_image`.

Once represented as an `NgffImage`, a multiscale representation can be generated
with `to_multiscales`. And an OME-Zarr can be generated from the multiscales
with `to_ngff_zarr`. For more information, see the
[Python interface documentation](./python.md).

## ITK Python

An example with
[ITK Python](https://docs.itk.org/en/latest/learn/python_quick_start.html):

```python
>>> import itk
>>> import ngff_zarr as nz
>>>
>>> itk_image = itk.imread('cthead1.png')
>>>
>>> ngff_image = nz.itk_image_to_ngff_image(itk_image)
>>>
>>> # Back again
>>> itk_image = nz.ngff_image_to_itk_image(ngff_image)
```

## ITK-Wasm Python

An example with [ITK-Wasm](https://wasm.itk.org). ITK-Wasm's `Image` is a simple
Python dataclass like `NgffImage`.

```python
>>> from itkwasm_image_io import imread
>>> import ngff_zarr as nz
>>>
>>> itk_wasm_image = imread('cthead1.png')
>>>
>>> ngff_image = nz.itk_image_to_ngff_image(itk_wasm_image)
>>>
>>> # Back again
>>> itk_wasm_image = nz.ngff_image_to_itk_image(ngff_image, wasm=True)
```
