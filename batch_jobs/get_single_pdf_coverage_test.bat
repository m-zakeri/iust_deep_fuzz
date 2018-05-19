REM get_single_pdf_coverage (just single file for test)
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

START VSPerfMon /coverage /output:./coverage6160/nmt2016_auto55
timeout /t 3
START mupdf nmt2016.pdf
timeout /t 55
taskkill /IM mupdf.exe /F
timeout /t 2
VSPerfCmd /shutdown
timeout /t 2
	

