REM get_single_pdf_coverage (just single file for test)
REM e.g. host1_max.pdf, host2_min.pdf, host3_avg.pdf
REM test mupdf pdf viewer on windows (c++ win32)
REM @ECHO off
SETLOCAL enabledelayedexpansion
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
REM SET PATH=%PATH%;"D:\afl\mupdf\platform\win32\Release"
REM C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\PrivateAssemblies
D:
CD D:\afl\mupdf\platform\win32\Release
CLS

START VSPerfMon /coverage /output:./coverage_single/host3_avg.pdf
timeout /t 5
START mupdf D:\AnacondaProjects\iust_deep_fuzz\incremental_update\hosts\rawhost_new\host3_avg.pdf
timeout /t 3
taskkill /IM mupdf.exe /F
timeout /t 2
VSPerfCmd /shutdown
	

