# AI Guidelines — Backend

> Extends root `AGENTS.md`. Rules here take precedence within this scope.

## Scope

FastAPI application that exposes APIs consuming trained ML models for predictions. This layer does **not** contain ML logic — it loads serialized models and serves prediction endpoints.

## Conventions

- Load trained models at app startup using `joblib.load()` from `ml/projects/<project>/trained_models/`.
- Prediction endpoints follow the pattern `/predict/<project_name>`.
- Input validation is done via Pydantic models (FastAPI's built-in).
- ML training, preprocessing, and evaluation logic belongs in `ml/`, not here.
