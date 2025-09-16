# ml-playground

This is a monorepo to play with machine learning concepts (mainly in Python).

# Guidelines

- Python `3.12` is a requirement when possible
- Use [Astral "uv" package manager](https://docs.astral.sh/uv/) for dependencies management
- You can use raw code or [Jupyter Labs](https://jupyter.org/install)
- The `AGENTS.md` file ([documentation](https://agents.md/)) will be a guide for local AI assistance

# Launch

Verify uv is installed on your machine - [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
uv run main.py
uv run
```

# Projects

## titanic-visualization

### Titanic dataviz (dataset manipulation)

The project `titanic-dataviz` directory is a dataset manipulation project with [scikit-learn](https://scikit-learn.org/stable/index.html) to visualize informations from the dataset source.
This will use neighbor algorithm and create a plot of survivors from open data.

# Recommendations

- [VSCode Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [Polars](https://pola.rs/)
- [kaggle](https://www.kaggle.com/)
- [Numpy](https://numpy.org/doc/stable/index.html)
