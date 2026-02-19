# Install ENN554 if needed (Colab safe)
import sys, subprocess, pkgutil

if "google.colab" in sys.modules:
    if pkgutil.find_loader("enn554") is None:
        print("Installing ENN554...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "enn554[notebooks] @ git+https://github.com/cholette/enn554"
        ])
