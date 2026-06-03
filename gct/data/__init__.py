"""Data loading modules.

Import `gct.data.torch_dataset` directly when PyTorch is available. Keeping this
package init light lets CLI help and ETL config checks run before GPU
dependencies are installed.
"""

__all__: list[str] = []
