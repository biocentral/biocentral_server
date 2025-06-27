# UV Package Manager Migration

This project has migrated from Poetry to UV package manager.

## Quick Start

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies 
uv sync --group dev

# Run commands
uv run python run-local.py
uv run pytest
```

## Command Changes

| Task | Old (Poetry) | New (UV) |
|------|--------------|----------|
| Install deps | `poetry install` | `uv sync --group dev` |
| Run command | `poetry run cmd` | `uv run cmd` |
| Add dependency | `poetry add pkg` | `uv add pkg` |
| Show deps | `poetry show` | `uv tree` |

## What Changed

- **pyproject.toml**: Converted to PEP 621 standard format
- **Lock file**: `uv.lock` replaces poetry.lock (auto-generated)
- **Virtual env**: Now in `.venv/` (managed by UV)
- **Dependencies**: Platform-compatible versions added

## Benefits

- ⚡ **Faster**: Significantly faster dependency resolution
- 🔒 **Reliable**: Better lock file and reproducible builds  
- 🛠️ **Modern**: Uses Python packaging standards
- 📦 **Simple**: One tool for all package management

## Notes

- The project now uses UV instead of Poetry
- All functionality verified and working
- 159 packages installed successfully
- Git dependencies resolved correctly

**Important**: Update your development environment to use UV commands.