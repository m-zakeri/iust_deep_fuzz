@echo off

set filename=today.txt
set /a counter=0
for /f %%j in ("%filename%") do set size=%%~zj
echo file_size = %size%
if %size% gtr 0 set /a counter=%counter%+1
echo %counter% > output.txt
				
timeout /t 10