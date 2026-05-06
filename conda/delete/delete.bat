@echo off
echo Removing Conda environment 'genai_project'...
call conda env remove -n genai_project
echo Deletion complete.
pause