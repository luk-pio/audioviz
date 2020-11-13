```
_____/\\\\\\\\\__________________________/\\\________________________________________________________
 ___/\\\\\\\\\\\\\_______________________\/\\\________________________________________________________
  __/\\\/////////\\\______________________\/\\\___/\\\______________________________/\\\_______________
   _\/\\\_______\/\\\__/\\\____/\\\________\/\\\__\///______/\\\\\_____/\\\____/\\\_\///___/\\\\\\\\\\\_
    _\/\\\\\\\\\\\\\\\_\/\\\___\/\\\___/\\\\\\\\\___/\\\___/\\\///\\\__\//\\\__/\\\___/\\\_\///////\\\/__
     _\/\\\/////////\\\_\/\\\___\/\\\__/\\\////\\\__\/\\\__/\\\__\//\\\__\//\\\/\\\___\/\\\______/\\\/____
      _\/\\\_______\/\\\_\/\\\___\/\\\_\/\\\__\/\\\__\/\\\_\//\\\__/\\\____\//\\\\\____\/\\\____/\\\/______
       _\/\\\_______\/\\\_\//\\\\\\\\\__\//\\\\\\\/\\_\/\\\__\///\\\\\/______\//\\\_____\/\\\__/\\\\\\\\\\\_
        _\///________\///___\/////////____\///////\//__\///_____\/////_________\///______\///__\///////////__
```

Exploring dimensionality reduction and music information retrieval techniques for visualizing large sets of audio data.

## The problem
Find an algorithm for producing visualizations of audio collections.

## Formal problem definition
> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

* **Task (T)**: Reduce the dimensions (to 2 or 3) of a set of audio files.
* **Experience (E)**: A dataset of audio files labeled based on similarity.
* **Performance (P)**: *see scoring below*

## Usage
[//]: <> (Browsing folders = context switching to focused mode from diffuse of creative process)

preprocess: `make data`

extract features: `make features`

## Feature selection


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
