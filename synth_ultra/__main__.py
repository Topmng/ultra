"""python -m synth_ultra  — same as ``python test.py``."""

from pathlib import Path
import importlib.util

_CLI = Path(__file__).resolve().parent.parent / "test.py"
_spec = importlib.util.spec_from_file_location("synth_ultra_test_cli", _CLI)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_CLI}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == "__main__":
    raise SystemExit(main())
