"""The shape of a prediction request, and of the answer to one.

This module writes down the contract the serving API is spoken to in: the
fields a caller sends, the fields it is sent back, and the type each one
carries. It is the boundary between a JSON payload and the typed objects
the rest of the application already works with, and that boundary is the
whole of it. Nothing here declares a route, reaches a model, predicts,
builds the frame a pipeline is given, or decides what a prediction means.

The feature fields restate the canonical schema in
``churner.schema.features``, name for name and in the order that module
lists them. They are restated rather than generated from those
collections, because a model assembled at import time would carry no field
a type checker or an editor could see, and the contract a caller reads
would have to be run to be known. The cost is that the two are kept in
step by hand: a column added to the canonical schema is a field added
here.

The feature names are the dataset's own -- ``MonthlyCharges`` and
``SeniorCitizen`` included -- rather than the snake_case a payload is
usually written in. The fitted pipeline selects its columns by name from
the frame it is handed, so a renamed field would have to be mapped back to
the dataset's name before anything could be predicted from it, and a
mapping that fell out of step would surface as a missing column at predict
time. The response fields are snake_case, because they name results rather
than columns.

Categorical fields are plain strings, not enumerations of the values the
training data happened to hold. The fitted ``OneHotEncoder`` is configured
with ``handle_unknown="ignore"``, so a value it was not fitted on already
has defined behaviour: every indicator for that column is zero, which is
the encoding the dropped reference category also receives. Refusing such a
value here would overrule that decision from outside the stage that made
it, and would reject a request the model is able to answer. What an unseen
value costs the answer is a modelling question, and preprocessing is where
it is settled.

``SeniorCitizen`` is typed as an integer even though it is a category. The
raw file records it as 0 and 1, so that is the form the encoder learned its
categories in, and the string ``"0"`` is not the integer ``0`` to an
encoder comparing values. A string field here would therefore turn every
request into one holding an unknown category, which ``handle_unknown`` would
absorb in silence rather than report.

``customerID`` is not a feature field, because the canonical schema leaves
it out: it identifies a customer rather than describing one. Nor is it a
body field. The route it is predicted through names the customer in its
path,

    POST /customers/{customer_id}/prediction

so the identifier is already stated by the address the request is sent to,
and a copy of it in the body would be a second statement of the same thing
that a caller could contradict. The request body therefore describes a
customer and says nothing about which one; the response names the customer
because an answer has to be attributable once it is away from the request
that asked for it.
"""

from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    """The canonical churn model features for one customer.

    The 19 fields are the three numerical and sixteen categorical columns
    ``churner.schema.features`` names, which is every column of the raw
    dataset except the identifier and the target.

    Every field is required, since a pipeline selects all of these columns
    from the frame it transforms and a missing one is a missing column
    rather than a missing value. ``TotalCharges`` is the one exception in
    what it may hold: the data dictionary documents it as blank for
    customers with no accumulated amount, and the median imputer the
    pipeline is fitted with is what stands in for that, so the field
    accepts ``None`` as the absence the imputer exists to resolve.

    No field is validated beyond its type. Ranges, category membership,
    and agreement between fields -- a customer with no phone service who
    reports multiple lines, say -- are not checked, because this model
    states what a request looks like and not what a plausible customer
    looks like.
    """

    # --- Numerical features ---
    # Recorded amounts, in the units the data dictionary documents: whole
    # months of tenure and two currency columns.
    tenure: int
    MonthlyCharges: float
    TotalCharges: float | None

    # --- Categorical features ---
    # Each value names a group rather than an amount. Unconstrained by
    # design; see the module docstring for why, and for why one of them is
    # an integer.
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class PredictionRequest(BaseModel):
    """The body of a request for a prediction, holding features alone.

    Which customer is being predicted about is not part of the body. It is
    the ``customer_id`` in the path of ``POST
    /customers/{customer_id}/prediction``, read as a path parameter by the
    route rather than from here, so this model carries only what a model is
    given.

    The features stay nested under ``features`` rather than being lifted to
    the top level, so the body keeps the separation the rest of the project
    holds to between what describes a customer and everything else said
    about the request.
    """

    features: CustomerFeatures


class PredictionResponse(BaseModel):
    """One model's answer about one customer.

    ``churn_prediction`` is the class the model predicted, in the Yes/No
    vocabulary it was fitted on rather than as a boolean, so the response
    reports the label the model produced instead of a reading of it.
    ``churn_probability`` is the probability it assigned to churn, which a
    caller applying its own cut-off reads in place of the class.

    ``model_version`` records which model produced the two, so a stored
    response stays attributable after the deployed model has moved on. No
    field is constrained beyond its type; the probability is reported as
    the model scored it.
    """

    customer_id: str
    churn_prediction: str
    churn_probability: float
    model_version: str
