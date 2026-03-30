# Contributing to Fraud Detection System

## Development Workflow

1. **Fork** the repository and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style

3. **Run tests** before committing
   ```bash
   # ML pipeline tests
   pytest services/ml-pipeline/ -v

   # FastAPI tests
   pytest services/ml-serving/ -v

   # Frontend tests
   cd services/frontend && npm test
   ```

4. **Commit** using conventional commits
   ```bash
   git commit -m "feat: add new feature"
   ```

5. **Push and create Pull Request**

## Code Style

- **Python:** Follow PEP 8, max line length 120
- **TypeScript:** Follow ESLint + Prettier defaults
- **Commit messages:** Use Conventional Commits format

## Service-Specific Guidelines

### ML Pipeline (`services/ml-pipeline/`)
- Scripts must be idempotent (re-run safe)
- Always log params and metrics to MLflow
- Use relative paths from the script's directory

### API Server (`services/ml-serving/`)
- FastAPI handles both ML inference and REST API
- Use Pydantic models for request/response validation
- Export Prometheus metrics for every endpoint
- Use SQLAlchemy for database operations

### Frontend (`services/frontend/`)
- Use functional components with hooks
- API calls go directly to FastAPI (`/transactions`, `/predict`, etc.)
- Follow dark theme design system

## Docker Guidelines

- Each service has its own `Dockerfile`
- Use multi-stage builds for smaller images
- All services must have health checks
- Use `.env` files for local development

## Questions?

Open an issue or reach out to the team!
