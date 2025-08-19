#!/usr/bin/env bash
set -Eeuo pipefail

# config (override with env vars if you want)
PY_BOOT_ENV="${PY_BOOT_ENV:-py311}"                     # conda env with python 3.11
ENV_DIR="${ENV_DIR:-$PWD/venv}"                         # venv target
KERNEL_NAME="${KERNEL_NAME:-bioacoustic-embedding-eval}"
KERNEL_LABEL="${KERNEL_LABEL:-Python ($KERNEL_NAME)}"

say() { printf '\n>>> %s\n' "$*"; }

say "ensure bootstrap env exists (python 3.11)"
conda run -n "$PY_BOOT_ENV" python -V >/dev/null 2>&1 || conda create -y -n "$PY_BOOT_ENV" python=3.11

say "locate py311 interpreter"
PY311=$(conda run -n "$PY_BOOT_ENV" python -c "import sys,shlex; print(shlex.quote(sys.executable))")

say "create venv (idempotent)"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  conda run -n "$PY_BOOT_ENV" "$PY311" -m venv "$ENV_DIR"
fi

VENV_PY="$ENV_DIR/bin/python"

say "seed build tools + numpy in venv"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
# "$VENV_PY" -m pip install "numpy<2"


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

say "install root project deps into venv"
if [ -f "$REQS_FILE" ]; then
  "$VENV_PY" -m pip install -r "$REQS_FILE"
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
echo "to test like your manual run:"
echo "  pytest -v --disable-warnings bacpipe/tests/test_embedding_creation.py"
