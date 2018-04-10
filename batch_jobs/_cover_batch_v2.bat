REM _cover_batch version 2.0
set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
cls
start VSPerfMon /coverage /output:_cover_batch_v2.coverage
REM ping google.com
timeout /t 5
FOR %%i IN (D:\iust_pdf_corpus\corpus_garbage\all_00_10kb\*.pdf) DO ( 
	mutool clean -difa %%i
	timeout /t 1
	)
timeout /t 10
VSPerfCmd /shutdown
