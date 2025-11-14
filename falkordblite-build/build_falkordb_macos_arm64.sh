#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="$SCRIPT_DIR/test_falkordblite.py"
BACKUP_SO="$SCRIPT_DIR/falkordb.so"

if ! uv pip show falkordblite >/dev/null 2>&1; then
  echo "falkordblite must be installed in the active uv environment before running this script." >&2
  exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/falkordb-build-XXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "Using build workspace: $WORKDIR"
git clone --depth=1 --recurse-submodules https://github.com/FalkorDB/FalkorDB.git "$WORKDIR/FalkorDB" >/dev/null

pushd "$WORKDIR/FalkorDB" >/dev/null

python - <<'PY'
from pathlib import Path

def inject(path: Path, sentinel: str, block: str, *, prepend: bool = False) -> None:
    text = path.read_text()
    if block.strip() in text:
        return
    if sentinel not in text:
        raise SystemExit(f"Sentinel '{sentinel}' not found in {path}")
    if prepend:
        path.write_text(text.replace(sentinel, block + "\n" + sentinel))
    else:
        path.write_text(text.replace(sentinel, sentinel + "\n" + block + "\n"))

vecsim = Path("deps/RediSearch/deps/VectorSimilarity/src/VecSim/CMakeLists.txt")
spaces = Path("deps/RediSearch/deps/VectorSimilarity/src/VecSim/spaces/CMakeLists.txt")

inject(
    vecsim,
    "target_link_libraries(VectorSimilarity VectorSimilaritySpaces)",
    'if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")\n    target_compile_options(VectorSimilarity PRIVATE -Wno-vla-cxx-extension)\nendif()',
)

spaces_text = spaces.read_text()
spaces_block = (
    "if(CMAKE_CXX_COMPILER_ID MATCHES \"Clang\")\n"
    "    target_compile_options(VectorSimilaritySpaces_no_optimization PRIVATE -Wno-vla-cxx-extension)\n"
    "    target_compile_options(VectorSimilaritySpaces PRIVATE -Wno-vla-cxx-extension)\nendif()\n"
)
if "target_compile_options(VectorSimilaritySpaces_no_optimization" not in spaces_text:
    spaces.write_text(spaces_text.rstrip() + "\n" + spaces_block)

macos_defs = Path("deps/readies/mk/macos.defs")
text = macos_defs.read_text()
needle = "CMAKE_FLAGS += -DCMAKE_OSX_DEPLOYMENT_TARGET=$(OSX_MIN_SDK_VER)"
replacement = "CMAKE_DEFS += CMAKE_OSX_DEPLOYMENT_TARGET=$(OSX_MIN_SDK_VER)\nCMAKE_DEFS += CMAKE_POLICY_VERSION_MINIMUM=3.5"
if needle not in text and replacement.split('\n')[0] not in text:
    raise SystemExit(f"Unable to find expected line in {macos_defs}")
if needle in text:
    macos_defs.write_text(text.replace(needle, replacement))
elif replacement.split('\n')[0] not in text:
    macos_defs.write_text(text + "\n" + replacement + "\n")
PY

echo "Building FalkorDB module with clang…"
rm -rf bin
PATH=/opt/homebrew/opt/make/libexec/gnubin:$PATH \
  gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++ >/dev/null

BUILT_SO="$WORKDIR/FalkorDB/bin/macos-arm64v8-release/src/falkordb.so"
if [[ ! -f "$BUILT_SO" ]]; then
  echo "Build completed but $BUILT_SO was not produced." >&2
  exit 1
fi

popd >/dev/null

echo "Updating backup copy at $BACKUP_SO"
cp "$BUILT_SO" "$BACKUP_SO"
chmod 755 "$BACKUP_SO"

TARGET_SO="$(python - <<'PY'
import redislite
from pathlib import Path
print(Path(redislite.__file__).with_name("bin") / "falkordb.so")
PY
)"

if [[ ! -f "$TARGET_SO" ]]; then
  echo "redislite module binary path not found at $TARGET_SO" >&2
  exit 1
fi

echo "Installing module into virtualenv: $TARGET_SO"
cp "$BUILT_SO" "$TARGET_SO"
chmod 755 "$TARGET_SO"

echo "Running verification test…"
python "$TEST_FILE"

echo "FalkorDBLite macOS arm64 setup completed successfully."
