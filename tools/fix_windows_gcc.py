"""Repair unquoted Conda GCC specs when the Pixi prefix contains spaces."""

import ctypes
import sys
from pathlib import Path


def main() -> None:
    """Replace this environment's long prefix with its Windows short path."""
    prefix = Path(sys.prefix)
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(prefix), buffer, len(buffer)):
        raise OSError("Could not obtain a short path for the Pixi environment")
    short = buffer.value.replace("\\", "/")
    for path in (prefix / "Library/lib/gcc").rglob("*.specs"):
        original = path.read_text()
        updated = original.replace(prefix.as_posix(), short).replace(str(prefix), short)
        if updated != original:
            path.write_text(updated)
            print(f"[Setup Acados] Repaired GCC specs: {path.name}")


if __name__ == "__main__":
    main()
