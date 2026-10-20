#!/usr/bin/env bash
# Reuse the installed project venv; no dependency on the Spack python-venv module.

tf_activate_existing_venv() {
  local tf_venv_path="${TF_VENV:-$HOME/venvs/tri-fair-v8}"
  local tf_python_module="${TF_PYTHON_MODULE:-python/3.11.14}"
  module purge || return 1
  if ! module --ignore_cache load "$tf_python_module"; then
    printf 'Cannot load Python base module %s. Inspect module spider python; set TF_PYTHON_MODULE explicitly.\n' "$tf_python_module" >&2
    return 1
  fi
  if [[ ! -r "$tf_venv_path/bin/activate" || ! -x "$tf_venv_path/bin/python" ]]; then
    printf 'Project venv is missing or its Python executable is broken: %s\n' "$tf_venv_path" >&2
    return 1
  fi
  source "$tf_venv_path/bin/activate" || return 1
  python - <<'PY'
import json
import os
from pathlib import Path
import sys

expected = Path(os.environ["VIRTUAL_ENV"]).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit("Project venv activation failed; refusing to use system Python")
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("Tri-Fair requires Python 3.11 or 3.12; restore the existing project venv")
print(json.dumps(dict(
    stage="existing_venv_activated", executable=sys.executable,
    python=sys.version.split()[0], prefix=sys.prefix, base_prefix=sys.base_prefix,
    python_venv_module_required=False,
), indent=2), flush=True)
PY
}
