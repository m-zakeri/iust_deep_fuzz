set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
cls
start VSPerfMon /coverage /output:_cover_batch_v1.coverage
REM ping google.com
timeout /t 5
FOR %%i IN (D:\iust_pdf_corpus\corpus_garbage\all_00_10kb\*.pdf) DO ( 
	mutool clean -difa %%i
	REM timeout /t 1
	)
timeout /t 5
VSPerfCmd /shutdown
