"""Enable ``python -m orfedit [file.orf]``."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
