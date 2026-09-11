"""The canonical modelling feature schema of the Telco customer churn dataset.

This module answers one question and holds nothing else: of the columns handed
to a model, which are recorded as amounts and which as categories. Preprocessing
treats the two kinds differently, so every stage that needs to make that
distinction reads it from here rather than restating it. A column that moved
between the two collections in one module and not in another would be scaled in
one place and one-hot encoded in another, which is exactly the drift a single
definition prevents.

The split follows how the data dictionary documents each column, not how pandas
happens to have read it. ``SeniorCitizen`` is the case where the two differ: it
is stored as 0 and 1, so it arrives as an integer, but it records which of two
groups a customer belongs to rather than an amount that can be added up or
averaged, and it is listed as a category here for that reason.

The collections name the predictors that ``prepare_modeling_data`` leaves in
the feature frame, so ``customerID`` and ``Churn`` appear in neither: the first
identifies a customer rather than describing one, and the second is the outcome
being predicted.

Nothing here is specific to a model. Which transformations a given model needs
its features to go through is a modelling decision, and it belongs to the
preprocessing stage that reads this schema, not to the schema itself.
"""

# --- Numerical features ---
# Recorded amounts: months of tenure and two currency columns. Each supports
# arithmetic, so a statistic such as a median or a standard deviation means
# something when fitted on it. ``TotalCharges`` belongs here as the amount the
# data dictionary documents, which is what ``prepare_modeling_data`` converts
# the raw text into.
NUMERICAL_FEATURES = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

# --- Categorical features ---
# Recorded categories: each value names a group a customer belongs to, and the
# groups carry no order, so the difference between two of them is not a
# quantity. Ordered by the account, service, and billing groupings the data
# dictionary uses, which is also the order the raw file holds them in.
CATEGORICAL_FEATURES = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
