# ml-playground

This is a monorepo to play with machine learning concepts (mainly in Python).

# Guidelines

- Python 3.12 is a requirement when possible
- Use [Astral "uv" package manager](https://docs.astral.sh/uv/) for dependencies management
- You can use raw code or [Jupyter Labs](https://jupyter.org/install)
- The `AGENTS.md` file ([documentation](https://agents.md/)) will be a guide for local AI assistance

# Launch

Verify uv is installed on your machine - [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```

# Projects

## titanic-visualization

### Scikit-Learn - Titanic dataset manipulation

The project is a dataset manipulation with sciki-learn to visualize informations from the dataset.
