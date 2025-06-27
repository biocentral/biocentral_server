# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Biocentral Server is a Flask-based Python REST API server that provides bioinformatics functionality for the biocentral frontend. It's a compute-intensive backend supporting protein analysis tasks including embeddings, predictions, protein-protein interactions, model training/evaluation, and Bayesian optimization.

## Development Commands

### Local Development Setup
```bash
# Install dependencies (requires Python 3.11 and Poetry)
poetry install

# For local development with workers and dependencies
cp .env.local .env
docker compose -f docker-compose.dev.yml up -d
poetry run python run-local.py

# For simple server-only development
poetry run python run-biocentral_server.py
```

### Production Setup
```bash
cp .env.example .env
docker compose up -d
```

### Testing
```bash
# Run all tests
pytest

# Security audit
poetry run pip-audit
```

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
- Run with: `pytest`

### Development Notes
- Uses Poetry for dependency management with Python 3.11 requirement
- Git workflow follows modified GitFlow with `main` and `develop` branches
- Branch naming: `<module_name>/feature/description` or `biocentral/feature/description`
- Server runs multi-process architecture with separate worker processes for compute-intensive tasks

## Documentation References

- Architecture evaluation can be found in `docs/architecture`
- Migration plan can be found in `docs/migrations`