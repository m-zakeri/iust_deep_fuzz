REM Compute model7 code coverage for all host, models, divs. Test suite of 72000 pdf
REM Mode = Single Object Update (SOU)
REM test mupdf pdf viewer on windows (c++ win32)
@ECHO off
REM SET PATH=%PATH%;"D:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools\x64"
SET PATH=%PATH%;"D:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\"
CLS
D:
CD D:\afl\mupdf\platform\win32\Release
CLS

FOR /F "usebackq tokens=1,2 delims==" %%i IN (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' SET ldt=%%j
SET ldt=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2% %ldt:~8,2%:%ldt:~10,2%:%ldt:~12,6%
ECHO Start_datetime is [%ldt%] > log_time.txt

ECHO "Test Cases" > log_file_names.txt

START VSPerfMon /coverage /output:./coverage_ift/host1_max_model7_div_1.0_mou_date_2019-03-16_13-53-57_t03.coverage
REM ping google.com
timeout /t 8
 
FOR %%i IN (D:\AnacondaProjects\iust_deep_fuzz\incremental_update\new_pdfs\host1_max\host1_max_model7_div_1.0_mou_date_2018-06-10_20-37-55\*.pdf) DO ( 
	timeout /t 1
	REM START mupdf %%i
	mupdf %%i
	timeout /t 1
	ECHO %%i >> log_file_names.txt
	REM taskkill /IM mupdf.exe
	REM timeout /t 2
	)

timeout /t 5
VSPerfCmd.exe /shutdown

FOR /F "usebackq tokens=1,2 delims==" %%i IN (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' SET ldt=%%j
SET ldt=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2% %ldt:~8,2%:%ldt:~10,2%:%ldt:~12,6%
ECHO End_datetime is [%ldt%] >> log_time.txt

REM With above config 1000 pdf file take about ... minutes to be processed
REM just to get code coverage data 



