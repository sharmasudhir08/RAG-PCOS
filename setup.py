#!/usr/bin/env python3
"""
One-command setup for RAG-PCOS.

Usage:
    python setup.py

Creates a virtual environment, installs dependencies, downloads the two
pretrained models into models/, and creates an empty Data/ folder with a
reminder to populate it (see PAPERS.md). Safe to re-run -- skips steps
that are already done.
"""
import os
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "Data")

IS_WINDOWS = os.name == "nt"
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin", "python.exe" if IS_WINDOWS else "python")


def step(msg):
    print(f"\n==> {msg}")


def run(cmd, **kwargs):
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    step("1/5 Creating virtual environment (venv/)")
    if os.path.isdir(VENV_DIR):
        print("    Already exists, skipping.")
    else:
        venv.create(VENV_DIR, with_pip=True)

    step("2/5 Upgrading pip")
    run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"])

    step("3/5 Installing dependencies from requirements.txt")
    run([VENV_PYTHON, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])

    step("4/5 Downloading pretrained models into models/ (see NOTICE for licenses)")
    run([VENV_PYTHON, "-m", "pip", "install", "--quiet", "huggingface_hub[cli]"])
    minilm_dir = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
    tinyllama_dir = os.path.join(MODELS_DIR, "TinyLlama-1.1B-Chat-v1.0")
    if os.path.isdir(minilm_dir) and os.listdir(minilm_dir):
        print(f"    {minilm_dir} already populated, skipping.")
    else:
        run([VENV_PYTHON, "-m", "huggingface_hub.commands.huggingface_cli", "download",
             "sentence-transformers/all-MiniLM-L6-v2", "--local-dir", minilm_dir])
    if os.path.isdir(tinyllama_dir) and os.listdir(tinyllama_dir):
        print(f"    {tinyllama_dir} already populated, skipping.")
    else:
        run([VENV_PYTHON, "-m", "huggingface_hub.commands.huggingface_cli", "download",
             "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--local-dir", tinyllama_dir])

    step("5/5 Creating Data/ folder")
    os.makedirs(DATA_DIR, exist_ok=True)
    readme_path = os.path.join(DATA_DIR, "README.txt")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "Put the PCOS research-paper PDFs here (one file per paper).\n"
                "See ../PAPERS.md for the list of 30 papers used in the published study.\n"
            )

    print("\nSetup complete.")
    print(f"Activate the environment with:")
    if IS_WINDOWS:
        print(r"    venv\Scripts\activate")
    else:
        print("    source venv/bin/activate")
    print("Then populate Data/ with PDFs (see PAPERS.md) and run:")
    print("    python main.py --retrieval_k 5")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nSetup failed at: {' '.join(e.cmd)}", file=sys.stderr)
        sys.exit(1)
