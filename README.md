# Titanic classifier

## Data

Data is coming from Kaggle: [Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

## The whole processus

1. I already provided a zipped archived of the dataset in `/assets/` but you may want to download a new version.
2. Read the csv and get a sum of missing values

You can read the downloaded csv file with `pandas` an sum the null values.
This means we are missing 86 values for `Age`, 327 values for `Cabin` and 2 for `Embarked`.

Two things:

- The types are not only numbers and a machine learning model only knows numbers
- Data are missing in this data frame (`Gender`, `Cabin`, `Embarked`)

So we need to preprocess data first in order to use them.

3. Preparing data
   Manipulation of the dataFrame with numpy to convert missing data into a value from 0 to 1.

## How to run project

### With the code

This project is using uv as package manager so be sure to have it on your computer.

```bash
uv sync
uv run main.py
```

### As Jupyter Lab (still a work in progress)

If you have a preference for jupyter labs there is a support for it too.

```bash
uv sync
jupyter lab
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
