REM get_single_pdf_coverage from corpus
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
SET /A counter=0
SET filename=out.pdf
SET "prefix=cov_"
SET "extension=.coverage"

FOR %%i IN (D:\iust_pdf_corpus\corpus_merged_2\*.pdf) DO (
	SET /A counter=!counter!+1
	START VSPerfMon /coverage /output:./drive_deh_10_100kb/!counter!
	timeout /t 2
	START mupdf %%i
	timeout /t 3
	taskkill /IM mupdf.exe /F
	timeout /t 1
	VSPerfCmd /shutdown
	timeout /t 1
	)
	

