



### How Everything Works Together

**Workflow Overview:**

1. You make code changes
2. Stage your files: git add .
3. Commit using commitizen: cz commit (instead of git commit)
4. Pre-commit hooks run automatically:
•  Black formats your code
•  isort organizes imports
•  flake8 checks for issues
•  Basic file cleanup happens
•  Commit message is validated
5. If hooks pass: Commit succeeds ✅
6. If hooks fail: You fix issues and try again ❌


### Key Commands:

**For commits:**

```bash

# Interactive guided commit (recommended)
cz commit

# Quick commit with message
cz commit -m "feat: add user authentication"
```

**For testing hooks:**

```bash
# Run all hooks manually
pre-commit run --all-files

# Run just Black formatting
pre-commit run black

# Run just flake8 linting
pre-commit run flake8
```

**For changelog generation:**

```bash
# Bump version and generate changelog
cz bump

# Just generate changelog
cz changelog
```


### What Happens During Commits:

1. Auto-fixes applied:
•  Code formatting (Black)
•  Import sorting (isort)
•  Trailing whitespace removal
•  End-of-file fixes
2. Checks performed:
•  Code style issues (flake8)
•  Commit message format (commitizen)
•  YAML syntax
•  Large file detection
•  Merge conflict detection
3. Result:
•  Pass: Clean, formatted commit with good message ✅
•  Fail: You fix issues and commit again ❌
