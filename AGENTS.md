# Python environment

This project uses the virtual environment located at `.venv`.

On Windows, always use:

- `.venv\Scripts\python.exe`
- `.venv\Scripts\pip.exe`

Do not use a globally installed Python interpreter.
Do not use another virtual environment.
Do not create a new virtual environment unless `.venv` does not exist.

When running Python commands, tests, formatters, or dependency installation,
invoke the executable from `.venv` explicitly.

Examples:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ruff check .
```

Before executing Python code, verify the interpreter with:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```