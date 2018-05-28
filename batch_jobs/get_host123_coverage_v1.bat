REM get host123 coverage (i.e. host1_max + host2_min + host3_avg)
REM measure code coverage for each pdf file seperatly (i.e 6160 items)
REM test mupdf pdf viewer on windows (c++ win32)
REM @ECHO off
SETLOCAL enabledelayedexpansion
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
REM SET PATH=%PATH%;"D:\afl\mupdf\platform\win32\Release"
REM C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\PrivateAssemblies
D:
cd D:\afl\mupdf\platform\win32\Release
CLS

START VSPerfMon /coverage /output:./coverage_single/host123.coverage
timeout /t 3

START mupdf D:\AnacondaProjects\iust_deep_fuzz\incremental_update\hosts\rawhost_new\host1_max.pdf
timeout /t 2
taskkill /IM mupdf.exe /F
timeout /t 1

START mupdf D:\AnacondaProjects\iust_deep_fuzz\incremental_update\hosts\rawhost_new\host2_min.pdf
timeout /t 2
taskkill /IM mupdf.exe /F
timeout /t 1

START mupdf D:\AnacondaProjects\iust_deep_fuzz\incremental_update\hosts\rawhost_new\host3_avg.pdf
timeout /t 2
taskkill /IM mupdf.exe /F
timeout /t 1

VSPerfCmd /shutdown
timeout /t 3
	

