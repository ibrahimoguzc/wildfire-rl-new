@echo off
cd ../..
powershell -command "pip install --upgrade pip; py -3.10 -m venv .env; .env/Scripts/activate; pip install -e .[dev,docs,cicd,geo]; git lfs install; pre-commit install; python examples.wildfire.main;"
pause
