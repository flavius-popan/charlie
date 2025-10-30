# FalkorDBLite on macOS arm64

This note documents how we build and verify a native `falkordb.so` for Apple Silicon, then drop it into the `falkordblite` virtualenv. Everything lives in `falkordblite-build/`.

## Prerequisites

- Xcode Command Line Tools installed (`xcode-select --install`)
- Homebrew `make` (`/opt/homebrew/opt/make/libexec/gnubin/gmake`)
- System clang/clang++ (via Xcode CLT)
- `uv pip install falkordblite` has already been run in the active environment

## Automated build

Run the helper script from this directory:

```bash
./build_falkordb_macos_arm64.sh
```

The script will:

1. Clone `FalkorDB` into a temporary workspace.
2. Apply clang-specific CMake guards to tolerate VLAs in VectorSimilarity.
3. Build with `gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++`.
4. Copy the resulting `falkordb.so` to:
   - `falkordblite-build/falkordb.so` (local backup)
   - `.venv/lib/python3.13/site-packages/redislite/bin/falkordb.so`
5. Run `python test_falkordblite.py`.
6. Clean up the temporary workspace.

The script exits on any failure, so a successful run means the embedded server is ready to use.

## Manual reference (if needed)

If the automation ever breaks, these are the key manual steps.

### Apply clang CMake guards

- `deps/RediSearch/deps/VectorSimilarity/src/VecSim/CMakeLists.txt`

  ```cmake
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(VectorSimilarity PRIVATE -Wno-vla-cxx-extension)
  endif()
  ```

- `deps/RediSearch/deps/VectorSimilarity/src/VecSim/spaces/CMakeLists.txt`

  ```cmake
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(VectorSimilaritySpaces_no_optimization PRIVATE -Wno-vla-cxx-extension)
      target_compile_options(VectorSimilaritySpaces PRIVATE -Wno-vla-cxx-extension)
  endif()
  ```

These blocks are what the script injects before compiling.

### Build the module manually

```bash
git clone https://github.com/FalkorDB/FalkorDB.git
cd FalkorDB
# ensure the guards above exist if upstream removed them
rm -rf bin
PATH=/opt/homebrew/opt/make/libexec/gnubin:$PATH \
  gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++
```

The Mach-O appears at `FalkorDB/bin/macos-arm64v8-release/src/falkordb.so`.

### Install the module

```bash
cp FalkorDB/bin/macos-arm64v8-release/src/falkordb.so \
   ../.venv/lib/python3.13/site-packages/redislite/bin/falkordb.so
chmod 755 ../.venv/lib/python3.13/site-packages/redislite/bin/falkordb.so
```

### Verify

```bash
cd ../falkordblite-build
python test_falkordblite.py
```

Expect to see `✓ All tests passed (2/2)`.

## Upstream follow-up

- When sending a PR upstream, carry the `-Wno-vla-cxx-extension` guards and mention the `gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++` invocation for Apple Silicon.
- Keep `falkordblite-build/falkordb.so` backed up until the official wheel ships an arm64 binary.
- Rerun the script whenever `falkordblite` or `FalkorDB` updates.
