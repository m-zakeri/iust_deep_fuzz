REM _cover_batch version 3.0
REM measure code coverage for corpus_merged include all of our pdfs (i.e 6160 items)
set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
cls
start VSPerfMon /coverage /output:_cover_batch_v3.coverage
REM ping google.com
timeout /t 5
FOR %%i IN (D:\iust_pdf_corpus\corpus_merged\*.pdf) DO ( 
	mutool clean -difa %%i
	REM timeout /t 1
	)
timeout /t 10
VSPerfCmd /shutdown
