@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)

if "%GRAPHVIZ_PATH%" == "" (
    set GRAPHVIZ_PATH=C:\Graphviz\bin\
)

REM The below are commands that only affect the current CMD environment!
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=MMG

REM Necessary for rendering of PlantUML and Inheritence Diagrams
set PATH=%PATH%;%GRAPHVIZ_PATH%
set GRAPHVIZ_DOT=%GRAPHVIZ_PATH%dot.exe

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
