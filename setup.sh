#!/usr/bin/env bash
set -Eeuo pipefail

echo
echo "setup.sh options:"
echo "  USE_CONDA=1        use conda instead of venv/uv"
echo "  INSTALL_KERNEL=1   also install a Jupyter kernel"
echo

# config (override with env vars if you want)
ENV_DIR="${ENV_DIR:-$PWD/venv}"
KERNEL_NAME="${KERNEL_NAME:-bacpipe}"
KERNEL_LABEL="${KERNEL_LABEL:-Python ($KERNEL_NAME)}"

say() { printf '\n>>> %s\n' "$*"; }

# Ensure python3.11 is available
if ! command -v python3.11 >/dev/null 2>&1; then
  say "Python 3.11 not found."
  if [ "${USE_CONDA:-0}" = "1" ]; then
    say "Please install Python 3.11 using conda: conda create -n py311 python=3.11"
    exit 1
  else
    read -p "Python 3.11 is required. Would you like to install it now? [Y/n] " yn
    case $yn in
      [Yy]*)
        if [[ "$OSTYPE" == "darwin"* ]]; then
          if command -v brew >/dev/null 2>&1; then
            say "Installing python@3.11 with Homebrew..."
            brew install python@3.11
          else
            say "Homebrew not found. Please install Homebrew first: https://brew.sh/"
            exit 1
          fi
        elif [[ "$OSTYPE" == "linux"* ]]; then
          say "Installing python3.11 with apt-get (sudo required)..."
          sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv
        else
          say "Please install Python 3.11 manually for your OS."
          exit 1
        fi
        ;;
      *)
        say "Python 3.11 is required. Exiting."
        exit 1
        ;;
    esac
  fi
fi


# Prompt to install uv if not available
if ! command -v uv >/dev/null 2>&1; then
  read -p "The 'uv' tool is not installed. Would you like to install it for faster and more reliable Python dependency management? Otherwise, we can proceed with 'venv' [Y/n] " uvyn
  case $uvyn in
    [Yy]*)
      say "Installing uv..."
      python3.11 -m pip install uv
      ;;
    *)
      say "Proceeding without uv."
      ;;
  esac
fi

# Create venv (idempotent)
if command -v uv >/dev/null 2>&1; then
say "Creating venv with uv (Python 3.11)"
python3.11 -m uv venv "$ENV_DIR"
elif command -v venv >/dev/null 2>&1; then
say "Creating venv with venv (Python 3.11)"
python3.11 -m venv "$ENV_DIR"
elif [ "${USE_CONDA:-0}" = "1" ]; then
say "Creating venv with conda (Python 3.11)"
PY_BOOT_ENV="${PY_BOOT_ENV:-py311}"
conda run -n "$PY_BOOT_ENV" python -V >/dev/null 2>&1 || conda create -y -n "$PY_BOOT_ENV" python=3.11
PY311=$(conda run -n "$PY_BOOT_ENV" python -c "import sys,shlex; print(shlex.quote(sys.executable))")
conda run -n "$PY_BOOT_ENV" "$PY311" -m venv "$ENV_DIR"
else
say "Creating venv with python3.11 built-in venv"
python3.11 -m venv "$ENV_DIR"
fi

VENV_PY="$ENV_DIR/bin/python"

# check for pip
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    say "no pip found in venv, bootstrapping with ensurepip"
    "$VENV_PY" -m ensurepip --upgrade
fi

say "seed build tools + numpy in venv"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel


say "install root project deps into venv"

# Detect OS and select requirements file, or allow override
REQS_FILE_ARG="${REQS_FILE:-}"
if [ -n "$REQS_FILE_ARG" ]; then
  REQS_FILE="$REQS_FILE_ARG"
else
  UNAME_OUT="$(uname -s)"
  case "$UNAME_OUT" in
    Linux*)   REQS_FILE="requirements.txt";;
    Darwin*)  REQS_FILE="requirements_mac.txt";;
    *)        REQS_FILE="requirements.txt";;
  esac
fi


# Install dependencies using uv, pip, or conda
say "install root project deps into venv"
if [ -f "$REQS_FILE" ]; then
  if command -v uv >/dev/null 2>&1; then
    say "Using uv to install dependencies"
    uv pip install -r "$REQS_FILE" --python "$VENV_PY"
  elif [ "${USE_CONDA:-0}" = "1" ]; then
    say "Using conda to install dependencies"
    PY_BOOT_ENV="${PY_BOOT_ENV:-py311}"
    conda install --file "$REQS_FILE" -n "$PY_BOOT_ENV"
  else
    say "Using pip to install dependencies"
    "$VENV_PY" -m pip install -r "$REQS_FILE"
  fi
fi


say "install bacpipe"
"$VENV_PY" -m pip install -e . --no-deps

say "install pytest"
"$VENV_PY" -m pip install pytest

# Optionally install Jupyter kernel
if [ "${INSTALL_KERNEL:-0}" = "1" ]; then
  say "install Jupyter kernel"
  "$VENV_PY" -m pip install ipykernel
  "$VENV_PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL"
fi

say "quick import check"
"$VENV_PY" - <<'PY'
import sys
try:
    import bacpipe
    print("import bacpipe: ok")
except Exception as e:
    print("import bacpipe: failed:", e, file=sys.stderr)
    sys.exit(1)
PY

echo
if [ "${INSTALL_KERNEL:-0}" = "1" ]; then
  echo "done. kernel installed: $KERNEL_LABEL"
else
  echo "done. Jupyter kernel not installed (set INSTALL_KERNEL=1 to enable)"
fi
echo
echo "to test:"
echo "  source venv/bin/activate"
echo "  pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py"
