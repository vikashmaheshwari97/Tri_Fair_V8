Tri-Fair V8 Clean: Windows UTF-8 hotfix (8.1.0-hf1-utf8)

Applies only to the Tri_Fair_V8_Clean architecture release, package 8.1.0.
The failure is a test reading UTF-8 Python source using Windows CP1252.
This patch also makes related evaluation/reporting JSON reads and writes UTF-8.
Optimizer behavior, dependencies and package version are unchanged.

INSTALL IN PYCHARM'S POWERSHELL TERMINAL
1. Save the ZIP to your Downloads folder.
2. Close any open editors for the three replaced Python files.
3. Run:

cd C:\Users\maheshwari\PycharmProjects\Tri_Fair_V8_Clean
Expand-Archive -LiteralPath "$HOME\Downloads\Tri_Fair_V8_Clean_UTF8_Hotfix.zip" -DestinationPath . -Force
.\.venv312\Scripts\python.exe -m pytest -q

The ZIP has root-relative paths. Extract into the project root, where
pyproject.toml and tests are located. Do not extract into src or .venv312.
Back up these files first if you have edited them yourself:
tests/test_architecture.py
src/tri_fair/cli/evaluate.py
src/tri_fair/reporting/summarize.py
RELEASE_MANIFEST.json
RELEASE_SHA256SUMS.txt

No pip reinstall or virtual environment recreation is needed for the existing
editable installation. Expected test result after applying: 74 passed.

VALIDATION
74 CPU tests passed on Python 3.12, Linux.
Includes two regression cases scanning the full package with simulated
CP1252 and ASCII default file encodings. Native Windows execution was not
available. Ruff checks passed. This is not GPU or experimental validation.
The original release validation reports are retained as historical records.
The refreshed release checksums cover the original 116 release paths;
this explanatory hotfix README is additional and is not in that manifest.

Optional integrity check (requires original release files otherwise unchanged):
.\.venv312\Scripts\python.exe scripts\verify_release.py
