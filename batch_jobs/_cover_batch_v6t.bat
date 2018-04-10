REM _cover_batch version 6.1
REM measure code coverage for corpus_merged include all of our pdfs (i.e 6160 items)
REM test mupdf pdf viewer on windows (c++ win32)
REM @ECHO off
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
REM SET PATH=%PATH%;"D:\afl\mupdf\platform\win32\Release"
D:
cd D:\afl\mupdf\platform\win32\Release
CLS
SET /A counter=0
SET filename=out.pdf

START VSPerfMon /coverage /output:_cover_batch_v6t3.coverage
REM ping google.com
timeout /t 5
 
FOR %%i IN (D:\AnacondaProjects\iust_deep_fuzz\incremental_update\new_pdfs\host1\host1_date_2018-04-05_17-59-59\*.pdf) DO ( 
	REM mutool clean -difa %%i	
	REM mutool clean -d %%i
	REM mutool clean -a
	REM mutool clean %%i
	REM mutool show -b %%i
	REM timeout /t 1
	start windbg -g -G mupdf %%i
	timeout /t 4
	taskkill /IM mupdf.exe
	timeout /t 1
	taskkill /IM windbg.exe
	timeout /t 1
	
	)

timeout /t 5
VSPerfCmd /shutdown


