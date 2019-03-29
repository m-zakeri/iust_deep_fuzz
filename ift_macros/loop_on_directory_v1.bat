
@ECHO off

SET PATH=%PATH%;"D:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
CLS
D:
CD D:\afl\mupdf\platform\win32\Release
CLS

for /F "usebackq tokens=1,2 delims==" %%i in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' set ldt=%%j
set ldt=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2% %ldt:~8,2%:%ldt:~10,2%:%ldt:~12,6%
ECHO Start_datetime is [%ldt%] > log_time.txt

ECHO "Test Cases" > log_file_names.txt
timeout /t 1
FOR %%i IN (D:\AnacondaProjects\iust_deep_fuzz\incremental_update\new_pdfs\host1_max\host1_max_model7_div_1.0_mou_date_2018-06-10_20-37-55\*.pdf) DO (
	mupdf %%i
	timeout /t 1
	ECHO %%i >> log_file_names.txt
	REM taskkill /IM mupdf.exe
	REM timeout /t 2
	)

for /F "usebackq tokens=1,2 delims==" %%i in (`wmic os get LocalDateTime /VALUE 2^>NUL`) do if '.%%i.'=='.LocalDateTime.' set ldt=%%j
set ldt=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2% %ldt:~8,2%:%ldt:~10,2%:%ldt:~12,6%
ECHO End_datetime is [%ldt%] >> log_time.txt
