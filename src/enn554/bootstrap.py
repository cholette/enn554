"""
Notebook bootstrap utilities for ENN554.

Provides a single function to ensure the package is installed when
running in Google Colab.
"""

from __future__ import annotations
import importlib.util
import sys
import subprocess


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _package_installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def ensure_installed() -> None:
    """
    Install the package from GitHub if running in Google Colab and the
    package is not already available.
    Safe to call multiple times.
    """

    if _in_colab() and not _package_installed("enn554"):
        print("Installing ENN554 package from GitHub...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "enn554[notebooks] @ git+https://github.com/cholette/enn554",
            ]
        )
