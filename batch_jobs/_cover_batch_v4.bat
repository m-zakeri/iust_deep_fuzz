REM _cover_batch version 4.0
REM measure code coverage for corpus_merged include all of our pdfs (i.e 6160 items)
REM also test another mutool clean options
REM @ECHO off
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
CLS
SET /A counter=0
SET filename=out.pdf

start VSPerfMon /coverage /output:_cover_batch_v4.coverage
REM ping google.com
timeout /t 5

FOR %%i IN (D:\iust_pdf_corpus\corpus_merged\*.pdf) DO ( 
	mutool clean -difa %%i	
	mutool clean -d %%i
	mutool clean -a
	mutool clean %%i
	mutool show -b %%i
	REM timeout /t 1
	)

timeout /t 5
VSPerfCmd /shutdown


