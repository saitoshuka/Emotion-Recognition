@echo off
setlocal

set "ROOT=%~dp0"
set "EEGNET_DIR=%ROOT%"
set "MORPHCAST_DIR=%ROOT%..\MorphCastProject_Suda"
set "MODEL_REL=..\realtime_eegnet_regression_deap\eegnet_regressor_best.pt"
set "SCALER_REL=..\realtime_eegnet_regression_deap\eegnet_regressor_scaler.npz"
set "PYTHON_EXE=C:\Users\xinji\.conda\envs\bci\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] bci python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%EEGNET_DIR%%MODEL_REL%" (
  echo [ERROR] model not found: %EEGNET_DIR%%MODEL_REL%
  pause
  exit /b 1
)

if not exist "%EEGNET_DIR%%SCALER_REL%" (
  echo [ERROR] scaler not found: %EEGNET_DIR%%SCALER_REL%
  pause
  exit /b 1
)

if not exist "%MORPHCAST_DIR%\index.html" (
  echo [ERROR] MorphCast index not found: %MORPHCAST_DIR%\index.html
  pause
  exit /b 1
)

echo Using python: %PYTHON_EXE%
echo Mode: DEMO OUTPUT

echo [1/4] starting websocket server...
start "WS Server" cmd /k "cd /d ""%EEGNET_DIR%"" && %PYTHON_EXE% websocket_server_compare_deap.py"

echo [2/4] starting dummy LSL EEG stream...
start "Dummy LSL" cmd /k "cd /d ""%EEGNET_DIR%"" && %PYTHON_EXE% dummy_lsl_eeg_stream_deap.py --stream-name DummyEEG --n-channels 32 --fs 100"

echo [3/4] starting realtime EEG inference (demo)...
start "EEG Realtime" cmd /k "cd /d ""%EEGNET_DIR%"" && set KMP_DUPLICATE_LIB_OK=TRUE && set OMP_NUM_THREADS=1 && %PYTHON_EXE% realtime_eegnet_circumplex_ws_deap.py --model-path %MODEL_REL% --scaler-path %SCALER_REL% --stream-name DummyEEG --fs 100 --ws-uri ws://localhost:8767 --step-sec 0.20 --demo-gain 2.8 --center-sec 50 --norm-sec 50 --demo-smooth-alpha 0.06"

echo [4/4] starting MorphCast web server...
start "MorphCast Web" cmd /k "cd /d ""%MORPHCAST_DIR%"" && %PYTHON_EXE% -m http.server 8000"

timeout /t 2 >nul
start "" http://localhost:8000/index.html

echo.
echo All services started. Close each window to stop.
endlocal
