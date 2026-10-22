# Churner

A production-style machine learning application for customer churn prediction.

The primary objective of this repository is **not** to achieve the highest prediction accuracy. Instead, it is to learn and demonstrate the complete machine learning engineering workflow used in professional software teams.

---

## Project Objectives

- Build an end-to-end machine learning application.
- Learn production-grade ML engineering practices.
- Develop maintainable and modular Python code.
- Apply software engineering principles to machine learning.
- Deploy the application using modern MLOps practices.

---

## Learning Goals

This project focuses on understanding:

- Repository organization
- Data acquisition and preprocessing
- Feature engineering
- Model development
- Model evaluation
- Testing
- Packaging
- API development with FastAPI
- Docker
- CI/CD
- Cloud deployment

---

## Tech Stack

- **Python 3.10** — the version the package targets and the container runs.
- **pandas** and **NumPy** — dataset loading, preparation, and the feature frames the model is given.
- **scikit-learn** — preprocessing, baseline estimators, pipelines, and evaluation metrics.
- **FastAPI and Pydantic** — the API framework and the schemas that validate prediction requests and responses.
- **Uvicorn** — the ASGI server the application is served with.
- **pytest** — the test suite.
- **Docker** — the image that serves the trained model.

Runtime versions are pinned in `requirements.txt`. Test and analysis tooling is pinned in `requirements-dev.txt`.

---

## Repository Structure

```
src/churner/        The installable package; each subpackage owns one step
  api/              FastAPI application, routes, request/response schemas,
                    and the dependency that hands a route the prediction service
  config/           The one place that reads the environment into settings
  evaluation/       Scoring a fitted pipeline, costing its errors, comparing
                    models, and selecting one against a policy
  packaging/        Saving a fitted pipeline to a file and loading it back
  preprocessing/    Model-specific preprocessors and the pipelines that join
                    them to an estimator
  serving/          Prediction from an already-fitted pipeline, one row per customer
  training/         The train/test split and the fitting of the baseline pipelines
  workflow/         End-to-end orchestration of train, evaluate, select, and promote
scripts/            Executable entry points, including the training run
tests/              The pytest suite, mirroring the package layout
models/             The trained artifact, produced locally and excluded from Git
data/               The raw dataset, excluded from the container image
docs/               Architecture and analysis notes written alongside the code
```

---

## Current Status

The end-to-end training path is implemented and runnable:

- Dataset loading — `src/churner/data/load_dataset.py`
- Preprocessing and modeling data preparation — `src/churner/preprocessing`, `src/churner/data/prepare_modeling_data.py`
- Model training and evaluation — `src/churner/training`, `src/churner/evaluation`
- Model selection and promotion — `src/churner/workflow/train_and_promote.py`
- Model persistence — `src/churner/packaging/model.py`
- Executable training entry point — `scripts/train.py`
- Prediction serving with FastAPI — `src/churner/api`, `src/churner/serving`
- Container image serving the trained model — `Dockerfile`

The test suite passes both locally and in GitHub Actions CI, which runs it against Python 3.10.

The Docker image has been built and a container from it has been run and checked: the application starts, the Docker health check reports `healthy`, and the prediction route answers from the artifact baked into the image.

Continuous integration is implemented: GitHub Actions runs the full pytest suite on pushes to `main` and on pull requests targeting `main`, and the initial run succeeded. The rest of the delivery path remains a learning goal — automated Docker image publishing, cloud deployment, and production monitoring are not yet implemented.

---

## Training Workflow

Run the training entry point from the project root:

```bash
python scripts/train.py
```

The script reads the raw dataset from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` and delegates training, evaluation, selection, and promotion to `train_and_promote()` in `src/churner/workflow/train_and_promote.py`. It holds no modeling logic of its own: it resolves the dataset and artifact paths relative to the project root, supplies the selection policy, and reports the outcome.

When a candidate satisfies the selection policy, the selected pipeline is saved to `models/churn_model.joblib` and the script reports the selected model, the selection status, and the artifact path.

When no candidate satisfies the policy, the run is still complete: no artifact is written and none is reported. The decision trace printed alongside the outcome records which gate ended the decision.

Model artifacts are excluded from version control (`*.joblib` in `.gitignore`), so the file is produced locally by running the script.

---

## API

The application is served by Uvicorn and exposes three routes:

| Route | Purpose |
| --- | --- |
| `GET /health/liveness` | Reports that the application process is alive. |
| `GET /health/readiness` | Reports that the prediction service is available to answer requests. |
| `POST /customers/{customer_id}/prediction` | Answers one prediction about the customer the path names. |

Interactive documentation is served at `http://localhost:8000/docs`.

A prediction request body holds a single `features` object carrying all 19 required model features — the three numerical and sixteen categorical columns of the raw dataset, under the dataset's own names. Which customer the request is about is not part of the body: it is the `customer_id` in the URL path, because the address already names the customer the answer will be attributed to.

```bash
curl -X POST http://localhost:8000/customers/7590-VHVEG/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure": 5,
      "MonthlyCharges": 79.85,
      "TotalCharges": 399.25,
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check"
    }
  }'
```

The response reports four fields:

- `customer_id` — the customer the answer is about, taken from the path.
- `churn_prediction` — the class the model predicted, in the `Yes`/`No` vocabulary it was fitted on.
- `churn_probability` — the probability the model assigned to churn, reported as it scored it.
- `model_version` — the version the deployment stamps its answers with, so a stored response stays attributable.

---

## Docker

Running the container requires Docker Desktop with WSL integration on Windows, or an equivalent Docker Engine environment on Linux or macOS.

The image is built from the project root, which is where the `Dockerfile` and the trained artifact are found:

```bash
docker build -t churner:local .
```

Run a container and publish the application port:

```bash
docker run -d --name churner -p 8000:8000 churner:local
```

Inspect the container and its health check:

```bash
docker ps
docker inspect churner --format '{{.State.Health.Status}}'
```

View the application logs:

```bash
docker logs churner
```

Stop and remove the container:

```bash
docker stop churner
docker rm churner
```

### What the image contains

- The runtime dependencies installed from `requirements.txt`, pinned versions included.
- The `churner` package, installed regularly rather than editably, so the code runs from the installed location.
- The existing `models/churn_model.joblib` artifact, copied in as it is.
- The FastAPI application, served by Uvicorn as a non-root user.

No model is trained while the image is built or when the container starts. The artifact has to exist locally beforehand, which is what `python scripts/train.py` produces; because `*.joblib` is excluded from Git, a fresh clone has to run the training script before the image can be built.

The container reads the same two environment variables the application reads anywhere: `CHURNER_MODEL_PATH`, which the image sets to the artifact's path inside the container, and `CHURNER_MODEL_VERSION`, which defaults to `unversioned`. That default is deliberate — the artifact carries no version of its own, so predictions are stamped with a value that says so rather than with one that implies otherwise. A deployment serving a known model should override it:

```bash
docker run -d --name churner -p 8000:8000 \
  -e CHURNER_MODEL_VERSION=2026-10-14-logreg \
  churner:local
```

---

## Tests

The test suite needs the development dependencies, which include `pytest` itself:

```bash
pip install -r requirements-dev.txt
```

Run the suite from the project root. `tests/conftest.py` puts `src/` on `sys.path`, so the tests import the package without it having been installed:

```bash
pytest
```

Useful variations:

```bash
pytest tests/api            # one directory
pytest -k readiness         # tests whose name matches an expression
pytest -q                   # quieter output
```

The suite is run against Python 3.10, the version the package targets. On Python 3.14 one test fails for a reason outside this project: `test_a_model_without_a_yes_class_raises_value_error` matches on the message of the `ValueError` that `list.index` raises, and CPython reworded that message in 3.14. Python 3.10 through 3.13 report the label in the message, as the test expects.

---

## License

This project is released under the MIT License. The full text is in [`LICENSE`](LICENSE), copyright 2026 Giridharan Murugesan.
