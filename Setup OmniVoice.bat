@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ===========================================================================
rem  OmniVoice control panel - setup for a fresh install
rem
rem  Double-click this file. It installs everything into this folder:
rem    uv (Python manager, if missing)  ->  OmniVoice source  ->  Python 3.12 env
rem    ->  PyTorch (CUDA 12.8 or CPU)  ->  OmniVoice + Gradio  ->  model download
rem  Safe to run again: it repairs and updates, and never deletes your outputs,
rem  voices or settings.
rem
rem  Options:  /auto      no questions (downloads the model)
rem            /nomodel   skip the model download
rem            /cpu       force a CPU-only install
rem ===========================================================================
cd /d "%~dp0"
title OmniVoice setup

set "OV_COMMIT=08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
set "OV_MODEL=k2-fsa/OmniVoice"
set "TORCH_VER=2.8.0"
set "LOG=%~dp0setup.log"
rem uv cache (C:) and this folder may be on different drives: copy instead of hardlink, quietly.
set "UV_LINK_MODE=copy"
set "AUTO=0"
set "NOMODEL=0"
set "FORCECPU=0"
for %%A in (%*) do (
  if /i "%%~A"=="/auto" set "AUTO=1"
  if /i "%%~A"=="/nomodel" set "NOMODEL=1"
  if /i "%%~A"=="/cpu" set "FORCECPU=1"
)

echo. > "%LOG%"
call :banner
call :log "Setup started in %CD%"

rem ---------------------------------------------------------------- 1. uv
call :step "1/7" "Checking for uv"
set "UV="
where uv >nul 2>&1 && for /f "delims=" %%U in ('where uv') do if not defined UV set "UV=%%U"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV=%USERPROFILE%\.cargo\bin\uv.exe"
if not defined UV (
  echo     uv not found, installing it from astral.sh ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >> "%LOG%" 2>&1
  if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV (
  call :fail "Could not install uv. Check your internet connection, or install it from https://docs.astral.sh/uv/ and run this again."
  exit /b 1
)
for /f "delims=" %%V in ('"!UV!" --version') do echo     %%V
call :log "uv: !UV!"

rem ---------------------------------------------------------------- 2. source
call :step "2/7" "Getting the OmniVoice source"
if exist "repo\pyproject.toml" (
  echo     Already here, keeping it.
) else (
  echo     Downloading k2-fsa/OmniVoice at the tested version ...
  rem Extract inside this folder: "move" cannot move a folder across drives (TEMP may be on C:).
  set "ZIP=%CD%\_omnivoice-src.zip"
  set "UNZ=%CD%\_omnivoice-src"
  if exist "!UNZ!" rd /s /q "!UNZ!"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://github.com/k2-fsa/OmniVoice/archive/%OV_COMMIT%.zip' -OutFile '!ZIP!'; Expand-Archive -Force '!ZIP!' '!UNZ!'" >> "%LOG%" 2>&1
  if not exist "!UNZ!\OmniVoice-%OV_COMMIT%\pyproject.toml" (
    call :fail "Downloading the OmniVoice source failed. Check your internet connection."
    exit /b 1
  )
  move "!UNZ!\OmniVoice-%OV_COMMIT%" "%CD%\repo" >> "%LOG%" 2>&1
  rd /s /q "!UNZ!" 2>nul
  del "!ZIP!" 2>nul
  if not exist "repo\pyproject.toml" (
    call :fail "Could not unpack the OmniVoice source into the repo folder. See setup.log."
    exit /b 1
  )
  echo     Done.
)

rem ---------------------------------------------------------------- 3. GPU
call :step "3/7" "Checking the graphics card"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_TAG=+cu128"
set "DEVICE=cuda"
set "GPU="
if "%FORCECPU%"=="1" goto :cpu
where nvidia-smi >nul 2>&1 || goto :cpu
for /f "delims=" %%G in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader 2^>nul') do if not defined GPU set "GPU=%%G"
if not defined GPU goto :cpu
echo     NVIDIA GPU: !GPU!
echo     Installing the CUDA 12.8 build of PyTorch.
goto :gpu_done
:cpu
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_TAG=+cpu"
set "DEVICE=cpu"
echo     No NVIDIA GPU found ^(or /cpu was given^). Installing the CPU build.
echo     It works, but generation is much slower than on a GPU.
:gpu_done
call :log "device=%DEVICE% gpu=!GPU!"

rem ---------------------------------------------------------------- 4. venv
call :step "4/7" "Creating the Python 3.12 environment"
if exist ".venv\Scripts\python.exe" (
  echo     .venv already exists, reusing it.
) else (
  "!UV!" venv --python 3.12 .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    call :fail "Could not create the Python environment. See setup.log."
    exit /b 1
  )
  echo     Created .venv
)
set "PY=%CD%\.venv\Scripts\python.exe"

rem ---------------------------------------------------------------- 5. torch
call :step "5/7" "Installing PyTorch %TORCH_VER% (about 3 GB on the first run)"
"!UV!" pip install --python "!PY!" "torch==%TORCH_VER%%TORCH_TAG%" "torchaudio==%TORCH_VER%%TORCH_TAG%" --extra-index-url %TORCH_INDEX% --index-strategy unsafe-best-match >> "%LOG%" 2>&1
if errorlevel 1 (
  call :fail "Installing PyTorch failed. See setup.log."
  exit /b 1
)
echo     Done.

rem ---------------------------------------------------------------- 6. omnivoice
call :step "6/7" "Installing OmniVoice, Gradio and the rest"
rem The upstream [tn] extra needs pynini, which has no Windows build; the app
rem falls back to num2words for number spelling instead.
"!UV!" pip install --python "!PY!" -e "./repo[lora]" num2words --extra-index-url %TORCH_INDEX% --index-strategy unsafe-best-match >> "%LOG%" 2>&1
if errorlevel 1 (
  call :fail "Installing OmniVoice failed. See setup.log."
  exit /b 1
)
"!PY!" -c "import torch, gradio, omnivoice; print('    torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| gradio', gradio.__version__, '| omnivoice', omnivoice.__version__)"
if errorlevel 1 (
  call :fail "The packages installed but do not import. See setup.log."
  exit /b 1
)
if not exist "outputs" mkdir "outputs"
if not exist "voices" mkdir "voices"
if not exist "settings.json" (
  > "settings.json" echo {"model": "%OV_MODEL%", "device": "%DEVICE%", "dtype": "float16", "asr_model": "openai/whisper-large-v3-turbo", "lora_adapter": ""}
  echo     Wrote settings.json for %DEVICE%.
)

rem ---------------------------------------------------------------- 7. model
call :step "7/7" "Model weights (about 3.3 GB)"
if "%NOMODEL%"=="1" (
  echo     Skipped. The model downloads the first time you generate.
  goto :done
)
if "%AUTO%"=="0" (
  choice /c YN /n /m "    Download the model now? [Y/N] "
  if errorlevel 2 (
    echo     Skipped. The model downloads the first time you generate.
    goto :done
  )
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"!PY!" -c "from huggingface_hub import snapshot_download as d; p=d('%OV_MODEL%'); print('    Model ready in', p)"
if errorlevel 1 (
  echo     The model download did not finish. It will retry the first time you generate.
  call :log "model download failed"
)

:done
echo.
echo  ======================================================================
echo   OmniVoice is installed.
echo   Start it with "Start OmniVoice.bat", then open http://127.0.0.1:7870
echo  ======================================================================
call :log "Setup finished OK"
if "%AUTO%"=="0" pause
exit /b 0

rem ---------------------------------------------------------------- helpers
:banner
echo.
echo  AIINFLUENCE / VOICE LAB
echo  OmniVoice setup
echo  ----------------------------------------------------------------------
exit /b 0

:step
echo.
echo  [%~1] %~2
call :log "[%~1] %~2"
exit /b 0

:log
>> "%LOG%" echo %date% %time%  %~1
exit /b 0

:fail
echo.
echo  !! %~1
call :log "FAILED: %~1"
if "%AUTO%"=="0" pause
exit /b 1
