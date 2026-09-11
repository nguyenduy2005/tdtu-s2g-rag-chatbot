"""Validate full-PDF coverage and write the derived deployment manifest."""

from __future__ import annotations

import json

from .config import ProductConfig
from .corpus import FullCorpusCatalog


def main() -> int:
    config = ProductConfig.load()
    catalog = FullCorpusCatalog.load(config)
    result = catalog.write_manifest(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
