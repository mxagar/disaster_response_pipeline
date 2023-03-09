# Disaster Response Pipeline

This repository contains a Machine Learning (ML) pipeline which predicts the response to messages in disaster situations. An ETL pipeline is also developed and the project is deployed to Heroku using a web app based on Flask; the final version is available under the following link (it might take some time to awaken the app the first time):

LINK

It is precisely during disaster situations, that the response organizations have the least capacity to evaluate and react properly to each message that arrives to them (via direct contact, social media, etc.). In this project, NLP is applied and a classification model trained so that the category of each message can be predicted automatically; then, the messages can be directed to the appropriate relief agencies.

The project is embedded in a Flask web app which visualizes the dataset and facilitates the usage of the classifier with a GUI:

<p style="text-align:center">
  <img src="./assets/disaster_response_app.png" alt="A snapshot of the disaster response app." width=1000px>
</p>

I took the [`starter`](starter) code for this project from the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) and modified it to the present form, which deviates significantly from the original version.

## Table of Contents

- [Disaster Response Pipeline](#disaster-response-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [How to Use This Project](#how-to-use-this-project)
    - [Installing Dependencies for Custom Environments](#installing-dependencies-for-custom-environments)
  - [Notes on the Implementation](#notes-on-the-implementation)
    - [ETL Pipeline](#etl-pipeline)
    - [Machine Learning Training Pipeline](#machine-learning-training-pipeline)
      - [Notes on the Model Evaluation](#notes-on-the-model-evaluation)
    - [Flask Web App](#flask-web-app)
    - [Tests](#tests)
    - [Deployment to Heroku](#deployment-to-heroku)
    - [Docker Container](#docker-container)
    - [Summary of Contents](#summary-of-contents)
  - [Next Steps, Improvements](#next-steps-improvements)
  - [References and Links](#references-and-links)
  - [Authorship](#authorship)

## Dataset

[`data`](data)

## How to Use This Project

The directory of the project consists of the following files:

```
.
├── Instructions.md                             # Original challenge/project instructions
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── assets/
├── data
│   ├── DisasterResponse.db
│   ├── Twitter-sentiment-self-drive-DFE.csv
│   ├── categories.csv
│   └── messages.csv
├── disaster_response
│   ├── __init__.py
│   ├── file_manager.py
│   ├── process_data.py
│   └── train_classifier.py
├── models/
├── logs
│   └── disaster_response_pipeline.log
├── notebooks
│   ├── ETL_Pipeline_Preparation.ipynb
│   └── ML_Pipeline_Preparation.ipynb
├── config.yaml
├── requirements.txt
├── conda.yaml
├── Dockerfile
├── docker-compose.yaml
├── run.sh
├── Procfile
├── runtime.txt
├── setup.py
├── starter/
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_app.py
    └── test_library.py
```

You can run the notebook at leas in two ways:

1. In a custom environment, e.g., locally or on a container. To that end, you can create a [conda](https://docs.conda.io/en/latest/) environment and install the [dependencies](#installing-dependencies-for-custom-environments) as explained below.
2. In Google Colab. For that, simply click on the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mxagar/airbnb_data_analysis/blob/master/00_AirBnB_DataAnalysis_Initial_Tests.ipynb)


### Installing Dependencies for Custom Environments

If you'd like to control where the notebook runs, you need to create a custom environment and install the required dependencies. A quick recipe which sets everything up with [conda](https://docs.conda.io/en/latest/) is the following:

```bash
# Create environment with YAML, incl. packages
conda env create -f conda.yaml
conda activate dis-res

# Alternatively, if you prefer, create your own environment
# and install the dependencies with pip
conda create --name dis-res pip
conda activate dis-res
pip install -r requirements.txt

# To track any changes and versions you have
conda env export > conda_.yaml
pip list --format=freeze > requirements.txt
```

[`conda.yaml`](./conda.yaml)

[`requirements.txt`](./requirements.txt)

## Notes on the Implementation

### ETL Pipeline

[`distaster_response/process_data.py`](./distaster_response/process_data.py)

[`data`](data)

```bash
cd data
# Enter SQLite terminal
sqlite3
# Open a DB
.open DisasterResponse.db
# Show tables
.tables # Student
# Get table info/columns & types
PRAGMA table_info(Message);
# Get first 5 entries
SELECT * FROM Message LIMIT 5;
# ...
# Exit SQLite CLI terminal
.quit
```

### Machine Learning Training Pipeline

[`distaster_response/train_classifier.py`](./distaster_response/train_classifier.py)

[`models`](models)

#### Notes on the Model Evaluation

Imbalanced dataset.

### Flask Web App

[`app/run.py`](./app/run.py)

### Tests

[`tests/conftest.py`](./tests/conftest.py)

[`tests/test_library.py`](./tests/test_library.py)

[`tests/test_app.py`](./tests/test_app.py)


### Deployment to Heroku

[`Procfile`](./Procfile)

[`.slugignore`](.slugignore)

[`runtime.txt`](runtime.txt)

[`requirements.txt`](./requirements.txt)

### Docker Container

[`Dockerfile`](./Dockerfile)

[`.dockerignore`](.dockerignore)

[`docker-compose.yaml`](./docker-compose.yaml)

[`run.sh`](./run.sh)

### Summary of Contents

- [x] ETL Pipeline in which datasets are merged and loaded to a SQLite database.
- [x] ML Pipeline which applies NLP to extract text features and train a random forest classifier.
- [x] Flask Web App
- [x] Tests
- [x] Deployment to Heroku using Continuous Integration and Continuous Delivery (CI/CD).
- [x] Containerization (Docker)

## Next Steps, Improvements

- [x] Add logging.
- [x] Lint with `flake8` and `pylint`.
- [ ] Extend tests; currently, the test package contains very few tests that serve as blueprint for further implementations.
- [ ] Add type hints to `process_data.py` and `train_classifier.py`; currently type hints and `pydantic` are used only in `file_manager.py` to clearly define loading and persistence functionalities and to validate the objects they handle.
- [ ] Add more visualizations.
- [ ] Based on the detected categories, suggest organizations to connect to.
- [ ] Improve the front-end design.
- [ ] Address the imbalanced nature of the dataset.

## References and Links

- A
- B
- C
- Link
- Link

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository useful, you're free to use it, but please link back to the original source.