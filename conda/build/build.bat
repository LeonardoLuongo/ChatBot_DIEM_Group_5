@echo on
echo Building Conda environment 'nlp_project'...
call conda env create -f conda\build\requirements.yaml
call conda activate nlp_project
echo Build complete.
pause