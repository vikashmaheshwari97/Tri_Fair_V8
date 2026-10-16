$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3.12 -c "import sys; print(sys.executable); assert sys.version_info[:2] == (3,12)"
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 before running this setup.' }
if (-not (Test-Path '.venv312\Scripts\python.exe')) {
    py -3.12 -m venv .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
}
$TaskPython = Join-Path $PSScriptRoot '.venv312\Scripts\python.exe'
& $TaskPython -c "import sys; assert sys.version_info[:2] == (3,12), 'This environment must use Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv312 uses another Python. Choose a fresh directory.' }
& $TaskPython -m pip install -r requirements-cpu.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $TaskPython -m tri_fair.cli.doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
& $TaskPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Do not submit GPU experiments yet.' }
Write-Host 'Setup passed. Select .venv312\Scripts\python.exe as the PyCharm interpreter.'
