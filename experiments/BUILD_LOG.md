# C++ Build Fix Log

## Goal
Fix CMake CUDA detection to rebuild TVM with pruning code.

## Attempts

### 1. Set CUDAARCHS env var (2026-07-16)
```bash
CUDAARCHS=86 cmake .. -G "Visual Studio 18 2026" -A x64
```
Result: FAIL - CMake still uses arch 5.2 in compiler test

### 2. CMAKE_CUDA_ARCHITECTURES=86 in config.cmake
```bash
echo 'set(CMAKE_CUDA_ARCHITECTURES "86")' >> build/config.cmake
```
Result: FAIL - config loaded before enable_language but compiler test ignores it

### 3. CMAKE_CUDA_FLAGS with -arch
```bash
-DCMAKE_CUDA_FLAGS="-arch=sm_86 -allow-unsupported-compiler"
```
Result: FAIL - test still adds its own -gencode

### 4. Explicit CUDA compiler path
```bash
-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
```
Result: FAIL - detection still runs and fails

## Root Cause
CMake 4.3's CUDA compiler ID detection test hardcodes `compute_52` which CUDA 12.6 doesn't support. No user-settable variable can override this.

### 5. FIX: Use py3-none-win_amd64 subdirectory (2026-07-16)
The REAL build tree is at `build/py3-none-win_amd64/` (Python wheel build).
Its CMakeCache.txt is intact. Compile from there:
```bash
cd F:/tvm/build/py3-none-win_amd64 && cmake --build . --config Release --target tvm_runtime -j8
```
Result: **SUCCESS** — tvm_runtime.dll rebuilt.
Also compiled with pruning code in utils.h → no errors.

## C++ Pruning Implementation
File: `src/s_tir/meta_schedule/utils.h`, line 342-366
Inserted between trace replay and EnterPostproc:
- Access `sch->trace()->decisions`
- Find `SamplePerfectTile` with n_splits==3 (k-loop)
- Check middle factor != 1
- If vk==1 → return nullopt (reject)

Compiled successfully. Need to test at runtime.
