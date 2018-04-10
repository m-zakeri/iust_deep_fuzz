set PATH=%PATH%;"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools"
cls
start VSPerfMon /coverage /output:_cover_batch_v1l.coverage
REM ping google.com
timeout /t 30
mutool clean D:\iust_pdf_corpus\corpus_garbage\all_00_10kb\pdftc_010k_2090.pdf
timeout /t 5
VSPerfCmd /shutdown
