REM _pass_rate_batch version 1.0
REM measure code coverage for corpus_merged include all of our pdfs (i.e 6160 items)
REM also measure pass rate for all of this corpus.
@ECHO off
REM SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
CLS
SETLOCAL EnableDelayedExpansion
SET counter=0
SET filename=out.pdf
SET size=0

REM ping google.com
timeout /t 5

FOR %%i IN (D:\iust_pdf_corpus\corpus_merged\*.pdf) DO (
	
	ECHO %%i
	mutool clean -difa %%i
	timeout /t 2	
	FOR /F %%j IN (out.pdf) DO SET size=%%~zj 
	timeout /t 2
	ECHO file_size: 
	ECHO !%size%!
	ECHO counter:
	ECHO !%counter%!
	timeout /t 2
	IF !%size%! gtr 0 SET /A counter=!%counter%!+1

	)

timeout /t 5
ECHO %counter% > output_pass_rate.txt
ENDLOCAL