@echo off
setlocal enabledelayedexpansion

REM Setting up the output file
set "OUTPUT_FILE=Kaggle_Classification_Kernels.csv"

REM Check if output file exists and delete it if it does
if exist "!OUTPUT_FILE!" del "!OUTPUT_FILE!"

REM Header for the CSV
echo ref,title,author,lastRunTime,totalVotes > "!OUTPUT_FILE!"

REM Looping through pages 1 to 12
for /l %%p in (1,1,12) do (
    echo Fetching page %%p
    kaggle kernels list -s "classification" --page-size 100 --page %%p >> "!OUTPUT_FILE!"
)

echo Done fetching data.
endlocal
