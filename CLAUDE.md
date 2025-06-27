# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Biocentral Server is a Flask-based Python REST API server that provides bioinformatics functionality for the biocentral frontend. It's a compute-intensive backend supporting protein analysis tasks including embeddings, predictions, protein-protein interactions, model training/evaluation, and Bayesian optimization.

## Development Commands

### Local Development Setup
```bash
# Install dependencies (requires Python 3.11 and UV package manager)
uv sync --group dev

# For local development with workers and dependencies
cp .env.local .env
docker compose -f docker-compose.dev.yml up -d
uv run python run-local.py

# For simple server-only development
uv run python run-biocentral_server.py
```

### Production Setup
```bash
cp .env.example .env
docker compose up -d
```

### Testing
```bash
# Run all tests
uv run pytest

# Security audit
uv run pip-audit

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### UV Package Manager

The project uses UV (not Poetry) for dependency management:
- `uv sync --group dev` - Install all dependencies including dev tools
- `uv add package` - Add new dependency
- `uv run command` - Run command in virtual environment
- `uv tree` - Show dependency tree

### Changesets

The project uses changesets for version management and release notes:
- `npx @changesets/cli add` - Create new changeset for changes
- `npx @changesets/cli version` - Apply changesets and update versions
- `npx @changesets/cli publish` - Publish releases
- Always create changesets using the CLI, not manually
- Use appropriate semver levels: patch, minor, major

## Architecture

### Core Application Structure

- **Entry Point**: `ServerAppState` singleton in `server_entrypoint/server_app_state.py` manages Flask app initialization
- **Modular Design**: Each service module (`embeddings/`, `predict/`, `ppi/`, etc.) has its own blueprint and endpoint handlers
- **Task Management**: Uses Redis Queue (RQ) for background job processing with worker processes
- **Database**: PostgreSQL for embeddings storage, Redis for job queues
- **File Storage**: SeaweedFS for distributed file storage

### Key Service Modules

- `embeddings/`: Protein sequence embedding generation using biotrainer
- `predict/`: Pre-trained model predictions (TMbed, VespaG, etc.)
- `prediction_models/`: Model training and evaluation workflows
- `bayesian_optimization/`: Gaussian process-based optimization
- `ppi/`: Protein-protein interaction analysis
- `plm_eval/`: Protein language model evaluation using autoeval
- `proteins/`: Protein data management with taxonomy support
- `protein_analysis/`: General protein sequence analysis functions

### Server Management

- `server_management/`: Core infrastructure components
- `task_management/`: RQ task interface and management
- `embedding_database/`: PostgreSQL strategy pattern for embeddings
- `file_management/`: Abstracted file storage backends
- `server_initialization/`: Module initialization orchestration
- `user_manager.py`: Request validation and user management

### Configuration

- Environment variables defined in `.env.example` and `.env.local`
- Key paths: `EMBEDDINGS_DATA_DIR`, `FILES_DATA_DIR`, `SERVER_TEMP_DATA_DIR`
- Services: PostgreSQL (embeddings), Redis (jobs), SeaweedFS (files)
- Default server port: 9540

### Dependencies

- Core ML libraries via git dependencies: biotrainer, autoeval, hvi_toolkit, vespag, tmbed, protspace
- Flask ecosystem with gunicorn for production serving
- Task processing via RQ (Redis Queue)
- Database via psycopg for PostgreSQL

### Testing Strategy

- Unit tests for pure functions and server_management classes
- Integration tests required for all endpoints
- Test discovery: `find biocentral_server -name "*_endpoint.py" | xargs grep "@.*_route.route"`
- Run with: `uv run pytest`

### Development Workflow

- Uses UV for dependency management with Python 3.11 requirement
- Git workflow follows modified GitFlow with `main` and `develop` branches
- Branch naming: `<module_name>/feature/description` or `biocentral/feature/description`
- **Always create changesets for changes**: Use `npx @changesets/cli add` (never create manually)
- **PR Requirements**: Use PR templates and complete all checklist items
  - Tests must pass: `uv run pytest`
  - Changeset required for all changes
  - Documentation updates for user-facing changes
  - LLM code review completed
- Server runs multi-process architecture with separate worker processes for compute-intensive tasks

## Documentation References

- Architecture evaluation can be found in `docs/architecture`
- Migration plan can be found in `docs/migrations`

## Commit Best Practices

Focus on WHY, not WHAT:

- Concise and purposeful
- Explain value/impact, not technical details
- Template: `[Action] [What] - [Why needed/benefit]`

## PR Best Practices

Always include the GitHub PR template checklist in PR descriptions and ensure all items are completed before requesting review. Never sign PRs with Claude Code attribution in the description. Focus on why and the high-level overview. Only include necessary details to understand the concepts.
