REM Get model7 code coverage for all host, models, divs. Test suite of 72000 pdf
REM Mode = Single Object Update (SOU)
REM test mupdf pdf viewer on windows (c++ win32)
REM @ECHO off
SET PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
CLS
D:
CD D:\afl\mupdf\platform\win32\Release
CLS

START VSPerfMon /coverage /output:./coverage_single/host2_min_model8_div_0.5_mou_date_2018-06-12_17-56-57.coverage
REM ping google.com
timeout /t 5
 
FOR %%i IN (D:\AnacondaProjects\iust_deep_fuzz\incremental_update\new_pdfs\host2_min\host2_min_model8_div_0.5_mou_date_2018-06-12_17-56-57\*.pdf) DO ( 
	REM mutool clean -difa %%i	
	REM mutool clean -d %%i
	REM mutool clean -a
	REM mutool clean %%i
	REM mutool show -b %%i
	REM timeout /t 1
	START mupdf %%i
	timeout /t 3
	taskkill /IM mupdf.exe
	timeout /t 2
	)

timeout /t 5
VSPerfCmd /shutdown

REM With above config 1000 pdf file take about 83.33 minutes to be processed
REM just to get code coverage data 



