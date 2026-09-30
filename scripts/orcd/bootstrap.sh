#!/usr/bin/env bash
# Bootstrap an already cloned checkout on an ORCD login node.
# This script uses only public package and repository endpoints.
set -euo pipefail

readonly JSPACE_ROOT="${JSPACE_ROOT:-${HOME}/scratch/jspace-validity}"
readonly COMPANION_URL="https://github.com/anthropics/jacobian-lens.git"
readonly COMPANION_REVISION="581d398613e5602a5af361e1c34d3a92ea82ba8e"
readonly CONFIG_SHA256="18581126da8946d50a7e5cef4d60cb2cc95af7807e25f3fa86b9422889a59b5f"
readonly UV_VERSION="0.12.1"

if [[ ! -d "$JSPACE_ROOT/.git" ]]; then
  echo "clone the reviewed repository before bootstrapping" >&2
  exit 1
fi
cd "$JSPACE_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -n "${JSPACE_EXPECTED_COMMIT:-}" && "$ACTUAL_COMMIT" != "$JSPACE_EXPECTED_COMMIT" ]]; then
  echo "deployed commit does not match the requested commit" >&2
  exit 1
fi
if ! git diff --quiet HEAD --; then
  echo "bootstrap requires a clean tracked checkout" >&2
  exit 1
fi

if ! module load miniforge/24.3.0-0 2>/dev/null; then
  module load miniforge
fi

UV_ENV="$JSPACE_ROOT/.orcd-bootstrap"
python -m venv "$UV_ENV"
"$UV_ENV/bin/python" -m pip install --disable-pip-version-check --quiet "uv==$UV_VERSION"
UV_BIN="$UV_ENV/bin/uv"

mkdir -p "$JSPACE_ROOT/vendor"
COMPANION_DIR="$JSPACE_ROOT/vendor/jacobian-lens"
if [[ -d "$COMPANION_DIR/.git" ]]; then
  if ! git -C "$COMPANION_DIR" diff --quiet HEAD --; then
    echo "companion checkout has modified tracked files" >&2
    exit 1
  fi
  git -C "$COMPANION_DIR" remote set-url origin "$COMPANION_URL"
  git -C "$COMPANION_DIR" fetch origin "$COMPANION_REVISION"
else
  git clone "$COMPANION_URL" "$COMPANION_DIR"
fi
git -C "$COMPANION_DIR" checkout --detach "$COMPANION_REVISION"

"$UV_BIN" sync --frozen --dev
mkdir -p logs/orcd .hf .scale-cache

.venv/bin/python - "$CONFIG_SHA256" <<'PY'
import json
import sys

from jspace.scale_config import config_sha256
from jspace.scale_runtime import verify_companion_checkout

expected_config = sys.argv[1]
observed_config = config_sha256()
if observed_config != expected_config:
    raise SystemExit(
        f"configuration digest mismatch: observed={observed_config}, expected={expected_config}"
    )
print(
    json.dumps(
        {
            "config_sha256": observed_config,
            "companion": verify_companion_checkout(),
        },
        indent=2,
    )
)
PY

printf 'bootstrap complete\n'
printf '  code: %s\n' "$ACTUAL_COMMIT"
printf '  uv: %s\n' "$("$UV_BIN" --version)"
printf '  companion: %s\n' "$(git -C "$COMPANION_DIR" rev-parse HEAD)"
