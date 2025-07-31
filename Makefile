# ───────────────────────── Makefile ─────────────────────────
#
# ManilaFolder Build System
#

# Sync configuration (adjust PUBLIC_DIR to your public repo path)
PRIVATE_DIR = .
PUBLIC_DIR = ../Manila

# Define exclusion patterns for sync
# Use dedicated .syncignore file for cleaner management
SYNCIGNORE_EXCLUDE = $(shell test -f .syncignore && echo "--exclude-from=.syncignore" || echo "")

# Use .gitignore patterns if the file exists
GITIGNORE_EXCLUDE = $(shell test -f .gitignore && echo "--exclude-from=.gitignore" || echo "")

# Always exclude .git directory
GIT_EXCLUDE = --exclude=".git/" --exclude="node_modules/"

# Combine all exclusions
ALL_EXCLUDES = $(GIT_EXCLUDE) $(SYNCIGNORE_EXCLUDE) $(GITIGNORE_EXCLUDE)

.PHONY: help install test lint type-check clean app package dev-install sync sync-preview sync-push check-public-repo

# Default target
help:
	@echo "ManilaFolder Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run code linting"
	@echo "  type-check   - Run type checking"
	@echo "  run          - Run desktop application (PySimpleGUI)"
	@echo "  streamlit    - Run web application (Streamlit)"
	@echo "  streamlit-dev- Run Streamlit with auto-reload"
	@echo "  app          - Build standalone application"
	@echo "  package      - Create distribution package"
	@echo "  clean        - Clean build artifacts"
	@echo ""
	@echo "Sync targets (private to public repo):"
	@echo "  sync-preview - Preview what would be synced (safe)"
	@echo "  sync         - Sync files to public repo"
	@echo "  sync-push    - Sync files and push to GitHub"

# Installation targets
install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-cov pytest-mock mypy flake8 pyinstaller build twine

# Development targets
test:
	pytest tests/ -v --cov=src/manilafolder --cov-report=term-missing

lint:
	flake8 src/manilafolder tests/ --max-line-length=127 --exclude=__pycache__,.git,build,dist

type-check:
	mypy src/manilafolder --ignore-missing-imports

run:
	python -m src.manilafolder.app

# Streamlit targets
streamlit:
	streamlit run streamlit_app.py

streamlit-dev:
	streamlit run streamlit_app.py --server.runOnSave true --server.port 8501

streamlit-prod:
	streamlit run streamlit_app.py --server.headless true --server.port 8501

# Build targets
app: clean
	@echo "Building standalone application..."
	pyinstaller --onedir --windowed --name manilafolder \
		--add-data "src/manilafolder:manilafolder" \
		--hidden-import="sentence_transformers" \
		--hidden-import="chromadb" \
		--hidden-import="langchain_community" \
		--exclude-module="matplotlib" \
		--exclude-module="jupyter" \
		--exclude-module="IPython" \
		src/manilafolder/app.py
	@echo "Application built: dist/manilafolder/"

app-simple: clean
	@echo "Building simple standalone application..."
	pyinstaller --onefile --console --name manilafolder \
		src/manilafolder/app.py
	@echo "Simple application built: dist/manilafolder"

package: clean
	@echo "Creating distribution package..."
	python -m build
	@echo "Package created in dist/"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete

# Quality assurance
qa: lint type-check test
	@echo "Quality assurance checks completed"

# Development workflow
dev: dev-install qa
	@echo "Development environment ready"

# ───────────────────────── Sync Targets ─────────────────────────

# Check if public repo exists and is a git repo
check-public-repo:
	@test -d $(PUBLIC_DIR) || (echo "❌ Public repo directory not found at $(PUBLIC_DIR)" && echo "💡 Create it with: git clone <your-public-repo-url> $(PUBLIC_DIR)" && exit 1)
	@test -d $(PUBLIC_DIR)/.git || (echo "❌ $(PUBLIC_DIR) is not a git repository" && exit 1)
	@echo "✅ Public repo looks good at $(PUBLIC_DIR)"

# Preview what would be synced without actually doing it
sync-preview: check-public-repo
	@echo "🔍 Preview of what would be synced:"
	@echo "📁 From: $(PRIVATE_DIR)"
	@echo "📁 To:   $(PUBLIC_DIR)"
	@echo ""
	rsync -av --dry-run $(ALL_EXCLUDES) $(PRIVATE_DIR)/ $(PUBLIC_DIR)/
	@echo ""
	@echo "💡 Run 'make sync' to actually perform the sync"

# Sync files to public repo
sync: check-public-repo
	@echo "🔄 Syncing to public repo..."
	@echo "📁 From: $(PRIVATE_DIR)"
	@echo "📁 To:   $(PUBLIC_DIR)"
	@echo ""
	@echo "🔍 Preview of changes:"
	@rsync -av --dry-run $(ALL_EXCLUDES) $(PRIVATE_DIR)/ $(PUBLIC_DIR)/ | head -20
	@echo ""
	@read -p "Continue with sync? (y/N): " confirm && [ "$$confirm" = "y" ] || (echo "❌ Sync cancelled" && exit 1)
	@echo ""
	rsync -av $(ALL_EXCLUDES) $(PRIVATE_DIR)/ $(PUBLIC_DIR)/
	@echo ""
	@echo "📝 Committing changes in public repo..."
	cd $(PUBLIC_DIR) && git add . && git commit -m "Sync from private repo $$(date '+%Y-%m-%d %H:%M')" || echo "ℹ️  No changes to commit"
	@echo "✅ Sync complete!"
	@echo "💡 Run 'make sync-push' next time to auto-push, or 'cd $(PUBLIC_DIR) && git push'"

# Sync and push to GitHub
sync-push: sync
	@echo "🚀 Pushing to GitHub..."
	cd $(PUBLIC_DIR) && git push origin main
	@echo "🎉 Sync and push complete!"
