@echo off
REM Configures and builds stable-diffusion.cpp (~/ai/stable-diffusion.cpp, cloned alongside
REM llama.cpp) with the CUDA backend. Not part of the image-generation feature yet — see
REM todo.md's "Migrate to stable-diffusion.cpp" and ADR-0013.
REM
REM Uses the VS2022 (14.44) MSVC toolset explicitly, not whatever VS version is newest on this
REM machine: CUDA 13.0's nvcc only officially supports MSVC up to VS2022, and actually crashes
REM (cudafe++ access violation) when pointed at a VS2026 toolset even with
REM -allow-unsupported-compiler. If this machine's VS install ever drops the 14.44 toolset,
REM reinstall it via the Visual Studio Installer's "Individual Components" ->
REM "MSVC v143 - VS 2022 C++ x64/x86 build tools" (side-by-side with newer toolsets is fine).
REM /bigobj is needed for src\stable-diffusion.cpp specifically: it exceeds the default COFF
REM section-count limit (heavy templating) without it — fatal error C1128 otherwise.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
cd /d "%USERPROFILE%\ai\stable-diffusion.cpp"
cmake -B build -G Ninja -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=/bigobj
if errorlevel 1 exit /b 1
cmake --build build --config Release
