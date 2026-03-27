@echo off
cd /d D:\Repository\parameter-golf

echo [%date% %time%] Starting plan2 training...
echo Config: FP16 Embed + SWA(0.4, every=50) + Warmdown=3000
echo Muon WD=0.0, Grad Clip=0.0 (observing grad_norm this run)
echo.

python plan2.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath 'logs\plan2.txt'"

echo.
echo [%date% %time%] Training finished.
pause
