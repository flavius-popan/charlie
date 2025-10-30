# FalkorDBLite on macOS arm64

This note captures the exact steps required to build a native `falkordb.so` for Apple Silicon and wire it into the `falkordblite` Python package so that the embedded server works locally. Keep this file in sync with any future upstream changes.

`falkordb.so` is currently backed up in `backup/` for safekeeping until a fix PR is submitted upstream.

## Prerequisites

- Xcode Command Line Tools installed: `xcode-select --install`
- Homebrew `make` (provides `gmake`) already available at `/opt/homebrew/opt/make/libexec/gnubin`
- Homebrew toolchain already provides a working `clang`/`clang++` at `/usr/bin/clang{,++}`
- Repo layout assumed to be checked out at `research/FalkorDB` under this project root.

## One-time build system tweaks

The stock VectorSimilarity CMake builds treat variable-length arrays as errors under clang. We added a clang-specific relaxation so the code compiles cleanly on macOS arm64:

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

These guards are already checked in. No further edits are required unless upstream changes the build.

## Building `falkordb.so`

From the project root (`/Users/flavius/repos/charlie`):

```bash
cd research/FalkorDB
rm -rf bin
PATH=/opt/homebrew/opt/make/libexec/gnubin:$PATH \
  gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++
```

On success, the arm64 Mach-O shared object is emitted at:

```
research/FalkorDB/bin/macos-arm64v8-release/src/falkordb.so
```

Keep this file around; upstream wheels currently ship Linux/x86_64 binaries.

## Installing `falkordblite` and swapping in the native module

```bash
# Install from local source (ensures we control the wheel contents)
uv pip install ./research/falkordblite

# Overwrite the packaged module with the freshly built Mach-O (keep permissions executable)
cp research/FalkorDB/bin/macos-arm64v8-release/src/falkordb.so \
   .venv/lib/python3.13/site-packages/redislite/bin/falkordb.so
chmod 755 .venv/lib/python3.13/site-packages/redislite/bin/falkordb.so
```

> `redislite/bin/redis-server` produced by the install is already a Mach-O arm64 binary, so no extra work is needed on the server side.

## Verification

Run the smoke test we keep in the repo:

```bash
python test_falkordblite.py
```

Expected output:

```
==================================================
FalkorDBLite Installation Verification
==================================================
...
✓ All tests passed (2/2)
==================================================
```

Once this passes, `falkordblite` is able to start the embedded Redis + FalkorDB module on macOS arm64.

## Next steps / hygiene

- When preparing a PR upstream, include the CMake `-Wno-vla-cxx-extension` guards plus a note about building with `gmake OSNICK=sonoma CLANG=1 CC=clang CXX=clang++` on Apple Silicon. These changes live in `deps/RediSearch/deps/VectorSimilarity/src/VecSim/{CMakeLists.txt,spaces/CMakeLists.txt}` and can be cherry-picked into a personal fork before opening the upstream PR.
- Archive `bin/macos-arm64v8-release/src/falkordb.so` somewhere stable (artifact storage, release tarball, etc.) until the upstream wheel publishes arm64 binaries.
- Whenever we refresh `falkordblite`, rerun the build + copy sequence to keep the module in sync.
- Keep an eye on upstream CMake changes; re-apply the `-Wno-vla-cxx-extension` workaround if their build system is reset.
