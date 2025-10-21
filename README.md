# Machine Learning Playground

## A word on projects

Projects are splited within `~./ml/projects/<project-name>`.
Each projet will share the venv, a `README.md` file will explain a bit on the project.

- `houses`: a machine learning project to predict houses prices based on a well known and prepared dataset from sklearn.
- `titanic`: a data manipulation project to create a heatmap of titanic catastrophe survivors.

### With the code

This project is using uv as package manager so be sure to have it on your computer.

If you have a preference for jupyter labs there is a support for it too.

```bash
uv sync
jupyter lab
```

## Repository Architecture

```
ml-playground/
├── backend/                     # API to serve models
│   ├── api/
│   │   ├── routes/
│   │   │   ├── titanic.py      # POST /predict/titanic
│   │   │   └── houses.py       # POST /predict/houses
│   │   └── main.py             # FastAPI app
│   └── requirements.txt
├── frontend/                    # Web interface (in the future)
│   ├── src/
│   ├── package.json
│   └── tsconfig.json
├── ml/                          # ML code
│   ├── shared/                  # Shared code between projects
│   │   ├── visualization/       # Plot
│   │   ├── metrics/             # Metrics
│   │   └── preprocessing/       # Preprocessing data
│   ├── projects/
│   │   ├── titanic/
│   │   │   ├── data/            # loaders and stored data (csv, etc)
│   │   │   ├── features/        # PolynomialFeatures
│   │   │   ├── models/          # regression, classification, etc
│   │   │   ├── trained_models/  # .pkl ou .joblib
│   │   │   └── experiments/     # Configs, metrics history, compare
│   │   └── houses/
│   │       └── ...
│   ├── jupyter/                 # Notebooks
│   └── tests/
├── assets/                      # Raw Datasets, images, etc.
└── docker-compose.yml           # Orchestration (optional)
```

## Explanations

### Projects Structure

`data/` Preprocessed data storage and specific loaders

- Loaders to load CSV and datasets
- Raw data

`features/` : Transformation

- Feature engineering (PolynomialFeatures, encoders, scalers)

`models/` : Blueprints, conception

- Architectures definitions (RandomForest, XGBoost, etc.)
- training code and hyperparametization

`trained_models/` : The final product

- Serialized trained models (.pkl, .joblib)
- Ready for production

`experiments/` : comparing step

- Configurations
- metrics history

### Shared Structure

- `metrics/`: metrics for training
- `pipelines`: pipelines for composing models and reproduce them
- `preprocessing`: data manipulation
- `visualization`: ploting

# Resources

| term                                                                                                                                                         | description                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) | create a training set and testing set                         |
| [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)                                                  | test multiple scenarios for best performances                 |
| [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)                                                    | cleaning data for the model (convert to a number from 0 to 1) |
| [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)                                        | classifier implementing the k-nearest neighbors vote.         |
| [matplotlib](https://matplotlib.org/)                                                                                                                        | it's a visualization library for machine learning statistics  |
| [seaborn](https://seaborn.pydata.org/)                                                                                                                       | a plugin for matplotlib                                       |

![Machine learning - Scikit-Learn](https://youtu.be/SW0YGA9d8y8?si=GY8nj5MjE_KFYWMR)
