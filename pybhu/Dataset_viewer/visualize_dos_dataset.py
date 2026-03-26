"""Compatibility wrapper for the general PyBHU dataset viewer."""

from __future__ import annotations

import os
import sys

from pybhu.Dataset_viewer import dataset_viewer


DEFAULT_DATASET = "dataset_dos20260302.pt"


def main(argv: list[str] | None = None):
    args = sys.argv[1:] if argv is None else argv
    if args:
        data = args[0]
    elif os.path.exists(DEFAULT_DATASET):
        print(f"No path given — using default dataset: {DEFAULT_DATASET}")
        data = DEFAULT_DATASET
    else:
        data = None
    return dataset_viewer(data, block=True)


if __name__ == "__main__":
    main()
