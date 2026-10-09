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

The technologies will be added as they are introduced throughout the project.

---

## Repository Structure

Repository structure documentation will be updated as development progresses.

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

The test suite currently passes: 562 tests passed, with 5 warnings.

Docker, CI/CD, and cloud deployment remain learning goals and are not yet implemented.

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

## Tests

Run the test suite from the project root:

```bash
pytest
```

---

## License

This project will use the MIT License.
