# Machine Learning Playground

## Projects

- Titanic: Data is coming from Kaggle: [Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)
- Houses pricing: Data is coming straight from `sklearn.datasets.fetch_california_housing()`

## The whole processus

1. Fetch or get a dataset
2. verify for missing values or bad type (models only accept numbers)
3. Prepare data for model: manipulation of the dataFrame with numpy to convert missing data into a value from 0 to 1.
4. Once you have your final dataFrame, if you have don't have _features_ and _target variables_ then make them. Split your DataFrame with `train_test_split()`
5. Train the model on it.
6. Evaluate the model

## How to run project

### With the code

This project is using uv as package manager so be sure to have it on your computer.

If you have a preference for jupyter labs there is a support for it too.

```bash
uv sync
jupyter lab
```

# Architecture

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
│   │   │   ├── data/
│   │   │   ├── features/
│   │   │   ├── models/
│   │   │   └── trained_models/  # .pkl ou .joblib
│   │   └── houses/
│   │       └── ...
│   ├── jupyter/                 # Notebooks
│   └── tests/
├── assets/                      # Datasets, images, etc.
└── docker-compose.yml           # Orchestration (optional)
```

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
