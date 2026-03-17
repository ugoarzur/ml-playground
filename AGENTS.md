# AI Coding Guidelines

> Only contains what an AI agent cannot deduce from repository files.
> Island files (e.g. `backend/AGENTS.md`, `ml/AGENTS.md`) extend this file and override where specified.

---

## Behavior

### Communication

- **Directness over diplomacy.** Correct mistakes explicitly; don't "yes-please" the user.
- **Goal-oriented responses.** Answer the question, add only necessary context, no filler.
- **Concise explanations.** Name the concept, show an example, stop.
- **No affirmations.** Don't spend time validating correctness; verify it.

### Language & Terminology

- **Default: English (US)** for all code, comments, docs, logs, APIs.
- **Domain terms** in other languages are allowed only if listed in `GLOSSARY.md`.
- Treat accepted non-English terms as atomic tokens (`SIRET`, `facture`) but keep surrounding code in English (`factureId`, `parseSiret`).

### Code Philosophy

| Principle | Description |
|-----------|-------------|
| **Explicit > Implicit** | No hidden state, silent mutations, or magic. Make data flow obvious. |
| **CQS (Command-Query Separation)** | Functions either change state (command) or return data (query), not both. |
| **Composition > Duplication** | Shared behaviors in focused helpers. No copy-paste of complex logic. |
| **Fail Fast** | Validate external input early. Return clear errors immediately. |
| **Minimal Surface** | One responsibility per module. Small, well-defined public APIs. |

**CQS Naming:**
- Commands: `create*`, `update*`, `delete*`, `assign*`
- Queries: `get*`, `find*`, `list*`, `is*`, `has*`, `count*`

### Comments & Documentation

- Write comments only when they add information not inferable from code.
- Explain **why** or **when**, not **what**.
- Prefer small, focused comments over documentation blocks.

### Safety & Compliance

- **Error handling:** Prefer returning error types over throwing. Make throwable functions explicit.
- **External boundaries:** Validate all external input (network, files, user input, env).
- **Secrets:** Never leak in logs or errors. Never commit `.env` files.
- **Logs:** Ensure compliance before emission (redact/mask sensitive data).

---

## Conventions

> Rules that config files and code alone don't convey.

### ML Project Structure

Each ML project lives under `ml/projects/<project_name>/` and **must** follow this layout:

```
ml/projects/<project_name>/
  data/            # Raw and processed datasets
  experiments/     # Experiment scripts and results
  features/        # Feature engineering code
  models/          # Model training scripts
  preprocessing/   # Data cleaning and transformation
  trained_models/  # Serialized models (.pkl, .joblib)
```

### Code Sharing (Monorepo)

- Shared utilities go in `ml/shared/` — reusable across all ML projects.
- Jupyter notebooks in `ml/jupyter/` are for **display and execution only** — all logic must live in `ml/projects/` code that notebooks import.

### Model Serialization

- Use `joblib.dump()` / `joblib.load()` for all model persistence.
- Trained models are saved to `ml/projects/<project>/trained_models/`.

---

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/sync all dependencies |
| `uv run fastapi dev backend/main.py` | Start FastAPI dev server |
| `uv run pytest` | Run tests |
| `jupyter lab` | Launch Jupyter Lab for ML experimentation |

---

## Project

### Overview

ML playground and model factory for experimenting with scikit-learn models, with a documented process from data exploration to serving predictions via FastAPI.

### Glossary

See [GLOSSARY.md](GLOSSARY.md) for machine learning terms and definitions used in this project.
