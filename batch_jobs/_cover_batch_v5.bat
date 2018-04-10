REM _cover_batch version 5.0
REM measure code coverage for corpus_merged include all of our pdfs (i.e 6160 items)
REM test mupdf pdf viewer on windows (c++ win32)
REM @ECHO off
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
CLS
SET /A counter=0
SET filename=out.pdf

START VSPerfMon /coverage /output:_cover_batch_v5.coverage
REM ping google.com
timeout /t 5
 
FOR %%i IN (D:\iust_pdf_corpus\corpus_merged\*.pdf) DO ( 
	REM mutool clean -difa %%i	
	REM mutool clean -d %%i
	REM mutool clean -a
	REM mutool clean %%i
	REM mutool show -b %%i
	REM timeout /t 1
	start mupdf %%i
	timeout /t 1
	taskkill /IM mupdf.exe /F
	
	)

timeout /t 5
VSPerfCmd /shutdown


