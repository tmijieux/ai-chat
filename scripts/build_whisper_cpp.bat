@echo off
REM Configures and builds whisper.cpp (~/ai/whisper.cpp, cloned alongside llama.cpp and
REM stable-diffusion.cpp) with the Vulkan backend. Not part of the voice-dictation feature
REM yet — see todo.md's whisper.cpp migration item and the 2026-08-23 handoff.
REM
REM Deliberately Vulkan, not CUDA: this project runs Whisper on the Intel Arc iGPU so it never
REM competes with the RTX 4070 for VRAM (the LLM and image-gen already fight over that 8GB).
REM GGML_VULKAN accelerates the whole ggml graph (encoder + decoder) across any Vulkan-capable
REM GPU, unlike WHISPER_OPENVINO which only accelerates the encoder.
REM
REM Uses the VS2022 (14.44) MSVC toolset for consistency with the other GGML-based builds on
REM this machine (see build_stable_diffusion_cpp.bat) — not required by Vulkan itself, just the
REM known-working toolset here.
set "VULKAN_SDK=C:\VulkanSDK\1.4.357.0"
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
cd /d "%USERPROFILE%\ai\whisper.cpp"
cmake -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build --config Release
