#!/usr/bin/env python3
"""Convenience launcher for the ORF Photo Editor.

Run ``python main.py [path/to/photo.orf]`` to start the GUI.
"""

from orfedit.app import main

if __name__ == "__main__":
    raise SystemExit(main())
