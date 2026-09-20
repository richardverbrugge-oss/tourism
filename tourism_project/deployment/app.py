"""Streamlit app "Score a Lead": should the sales team contact this lead first?

Context
    Visit with Us is introducing the Wellness Tourism Package. Contacting every customer is expensive,
    so marketing wants to know, BEFORE a customer is contacted, whether that customer belongs to the
    group most likely to buy. This app is the last stage of the MLOps pipeline:

        data_register.py  ->  prep.py  ->  train.py         ->  this app
                                           model on the Hub      (Hugging Face Space)

    A lead is a customer who has not been contacted yet. A user enters the profile of one lead; the app
    returns a ranking score and the contact group the lead falls into (top 5%, 10% or 20% of customers),
    together with how well that group did on test customers the model never saw during training.

Where things come from
    - The trained model, its metadata and its operating points are downloaded from the private
      Hugging Face model repository at the commit given in the MODEL_REVISION environment variable
      (a Space variable). Pinning the commit means a new training run never changes the app silently.
    - model_metadata.json holds everything the app needs to know about the model: feature lists,
      allowed input ranges, category values, thresholds and package versions. The app keeps no copy
      of these, so it can never drift apart from the model.
    - HF_TOKEN (a Space secret with read access to the model repository only) is read from the
      environment and never shown or logged.

Why the score is not called a probability
    The model ranks customers well, but its scores were not calibrated to match real purchase rates.
    The app therefore explains a score through the contact groups and their measured precision, not
    as "x% chance to buy".

Inputs outside the training data
    Every numeric input must lie within the range seen in training (e.g. monthly income 16,009 - 38,291).
    Otherwise the app gives no score and says which value is outside which range: the model has no
    information about such customers, so a score would be a guess presented as an assessment.

Deliberately left out
    No file upload: a public app that accepts customer files invites people to send customer data to
    a public server. Scoring a whole customer list belongs in an internal job, not in this page.
    Inputs are never printed or logged.
"""
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


# =============================================================================
# Settings
# =============================================================================
#
# The container only contains this file, so the project's central Config is not available. The few
# fixed values below describe the model repository layout that train.py writes; everything about the
# model itself comes from model_metadata.json.
#

DEFAULT_MODEL_REPO = "richvrb/tourism-wellness-model"
MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"
OPERATING_POINTS_FILE = "operating_points.csv"

# A score recomputed here may differ from the training environment only by rounding noise.
PARITY_TOLERANCE = 1e-9

# How the operating points are shown to marketing, from most to least selective.
CONTACT_GROUPS = {
    "top_5pct": "Top 5% of customers",
    "top_10pct": "Top 10% of customers",
    "top_20pct": "Top 20% of customers",
}
BALANCED_POINT = ("max_f1", "Best balance of precision and recall")

# Readable labels and the section of the form each feature appears in. Features are grouped the way a
# sales employee thinks about a customer. Any feature not listed here still gets a field, under "Other".
FORM_SECTIONS = {
    "Customer profile": {
        "Age": "Age",
        "Gender": "Gender",
        "MaritalStatus": "Marital status",
        "Occupation": "Occupation",
        "Designation": "Job title",
        "MonthlyIncome": "Gross monthly income",
    },
    "Travel habits": {
        "NumberOfTrips": "Trips per year",
        "PreferredPropertyStar": "Preferred hotel rating (stars)",
        "Passport": "Holds a valid passport",
        "OwnCar": "Owns a car",
    },
    "Planned trip": {
        "NumberOfPersonVisiting": "People travelling",
        "NumberOfChildrenVisiting": "Children under 5 travelling",
    },
    "Contact and location": {
        "TypeofContact": "How the customer got in touch",
        "CityTier": "City tier (1 = most developed)",
    },
}


@dataclass(frozen=True)
class AppSettings:
    """Values the Space provides through environment variables."""

    model_repo: str
    model_revision: str | None
    token: str | None


def read_settings() -> AppSettings:
    """Read the model repository, its pinned revision and the access token from the environment."""
    return AppSettings(
        model_repo=os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO),
        model_revision=os.environ.get("MODEL_REVISION"),
        token=os.environ.get("HF_TOKEN"),  # None locally: then the `hf auth login` token is used
    )


# =============================================================================
# Loading the model
# =============================================================================
#
# st.cache_resource keeps the loaded model in memory for the lifetime of the container. Without it,
# Streamlit would download and load the ~20 MB model again on every click, because it reruns the whole
# script on each interaction.
#

@st.cache_resource(show_spinner="Loading the model from the Hugging Face Hub ...")
def load_bundle(model_repo: str, model_revision: str, _token: str | None) -> tuple:
    """Download model, metadata and operating points at the pinned revision; load them once."""
    # The leading underscore in _token tells Streamlit not to use the token as part of the cache key.
    def download(filename: str) -> str:
        return hf_hub_download(model_repo, filename, revision=model_revision, token=_token)

    model = joblib.load(download(MODEL_FILE))
    metadata = json.loads(Path(download(METADATA_FILE)).read_text(encoding="utf-8"))
    operating_points = pd.read_csv(download(OPERATING_POINTS_FILE))
    return model, metadata, operating_points


def feature_order(metadata: dict) -> list[str]:
    """Model input columns in the order the model was trained with."""
    groups = metadata["feature_columns"]
    return [*groups["numeric"], *groups["binary"], *groups["categorical"]]


# =============================================================================
# Start-up checks
# =============================================================================
#
# A model that loads without an error can still give different scores when the installed package
# versions differ from the training environment. Two checks run before the app shows anything:
#   1. the installed versions of the packages the model depends on must equal those in the metadata;
#   2. the synthetic example profiles stored at training time must get exactly the same score here.
# If either fails the app stops with an explanation: no app is better than an app with wrong scores.
#

def version_mismatches(metadata: dict) -> list[str]:
    """Packages whose installed version differs from the version the model was trained with."""
    return [
        f"{package}: installed {version(package)}, model trained with {trained_version}"
        for package, trained_version in metadata["packages"].items()
        if version(package) != trained_version
    ]


def parity_mismatches(model: object, metadata: dict) -> list[str]:
    """Example profiles whose score here differs from the score at training time."""
    mismatches = []
    for number, example in enumerate(metadata["parity_examples"], start=1):
        row = pd.DataFrame([example["input"]])[feature_order(metadata)]
        score = float(model.predict_proba(row)[:, 1][0])
        if abs(score - example["expected_score"]) > PARITY_TOLERANCE:
            mismatches.append(f"example {number}: expected {example['expected_score']:.6f}, got {score:.6f}")
    return mismatches


# =============================================================================
# Input form
# =============================================================================
#
# The form only offers what the model knows, and checks what it cannot restrict:
#   - categorical fields: exactly the categories the model's encoder learned (category_levels);
#   - yes/no fields: a toggle that becomes 1 or 0;
#   - numeric fields: accept any number, and show the range seen in training (input_bounds) as help.
#
# Numeric fields deliberately have no hard minimum or maximum. Streamlit silently replaces a value
# outside such limits by the previous valid value, so a customer with an income of 40,000 would be
# scored as if the income were the default 22,369, without any notice. Instead, the values are checked
# after submit (out_of_range_fields): if any value lies outside the range in the training data, the app
# shows which field and range, and gives NO score. The model has never seen such customers, so any score
# for them would be a guess presented as an assessment. The same check also catches typing mistakes
# such as "30.000" being read as 30.
#
# The form starts with a "typical" customer (median and most common values), the first parity example.
# The answers are returned as a one-row DataFrame in the model's column order, which is the format the
# model pipeline expects.
#

def section_of_features(metadata: dict) -> dict[str, dict[str, str]]:
    """Form sections with (feature -> label); features not placed in a section go to 'Other'."""
    placed = {feature for fields in FORM_SECTIONS.values() for feature in fields}
    sections = {
        title: {feature: label for feature, label in fields.items() if feature in feature_order(metadata)}
        for title, fields in FORM_SECTIONS.items()
    }
    unplaced = [feature for feature in feature_order(metadata) if feature not in placed]
    if unplaced:
        sections["Other"] = {feature: feature for feature in unplaced}
    return sections


def input_step(bounds: dict) -> float:
    """Step for the +/- buttons, scaled to the range: 1 for age, 100 for monthly income."""
    spread = bounds["max"] - bounds["min"]
    magnitude = 10 ** (len(str(int(spread))) - 3)  # a step of roughly 1/100 to 1/1000 of the range
    return max(1, magnitude) if bounds["integer"] else max(0.01, magnitude)


def feature_label(feature: str) -> str:
    """Readable label of a feature, as shown in the form."""
    for fields in FORM_SECTIONS.values():
        if feature in fields:
            return fields[feature]
    return feature


def out_of_range_fields(customer: pd.DataFrame, metadata: dict) -> list[str]:
    """Messages for numeric inputs outside the range seen in training; empty when all are inside."""
    messages = []
    for feature, bounds in metadata["input_bounds"].items():
        value = float(customer[feature].iloc[0])
        if not bounds["min"] <= value <= bounds["max"]:
            messages.append(
                f"{feature_label(feature)} {value:,.0f} is outside the range in the training data "
                f"({bounds['min']:,.0f} – {bounds['max']:,.0f})."
            )
    return messages


def input_widget(feature: str, label: str, metadata: dict, default: object) -> object:
    """One form field whose type and allowed values follow the model metadata."""
    groups = metadata["feature_columns"]

    if feature in groups["numeric"]:
        bounds = metadata["input_bounds"][feature]
        help_text = f"Range in the training data: {bounds['min']:,.0f} – {bounds['max']:,.0f}"
        step = input_step(bounds)
        # No min_value/max_value on purpose: see the section comment above.
        if bounds["integer"]:
            return st.number_input(label, value=int(round(default)), step=int(step), help=help_text)
        return st.number_input(label, value=float(default), step=step, help=help_text)

    if feature in groups["binary"]:
        return int(st.toggle(label, value=bool(default)))

    options = metadata["category_levels"][feature]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)


def customer_form(metadata: dict) -> pd.DataFrame | None:
    """Show the input form; return the lead as a one-row DataFrame after submit, else None."""
    typical_customer = metadata["parity_examples"][0]["input"]
    answers = {}
    with st.form("customer"):
        for title, fields in section_of_features(metadata).items():
            st.subheader(title)
            columns = st.columns(2)
            for position, (feature, label) in enumerate(fields.items()):
                with columns[position % 2]:
                    answers[feature] = input_widget(feature, label, metadata, typical_customer[feature])
        submitted = st.form_submit_button("Score this lead", type="primary")

    if not submitted:
        return None
    return pd.DataFrame([answers])[feature_order(metadata)]


# =============================================================================
# Score and explanation
# =============================================================================
#
# The model gives a score between 0 and 1; a higher score means the customer resembles past buyers
# more. A score becomes a decision through the thresholds fixed during training:
#   - top_5pct / top_10pct / top_20pct: the score from which a customer belongs to the 5%, 10% or 20%
#     highest-scoring customers, for when marketing has a fixed calling budget;
#   - max_f1: the threshold with the best balance between finding buyers and avoiding wasted calls.
# For each group the app shows how it did on the test set: precision (share of contacted customers
# who bought) and lift (how many times better than contacting customers at random).
#

def score_customer(model: object, customer: pd.DataFrame) -> float:
    """Ranking score of one customer: higher means more similar to customers who bought."""
    return float(model.predict_proba(customer)[:, 1][0])


def test_results(operating_points: pd.DataFrame) -> pd.DataFrame:
    """Test-set precision and lift per operating point, indexed by operating point."""
    return operating_points[operating_points["split"] == "test"].set_index("operating_point")


def operating_point_view(score: float, metadata: dict, operating_points: pd.DataFrame) -> pd.DataFrame:
    """Table: for each contact group, its threshold, whether this lead is in it, and test results."""
    thresholds = metadata["operating_points"]
    results = test_results(operating_points)
    rows = []
    for key, label in [*CONTACT_GROUPS.items(), BALANCED_POINT]:
        rows.append(
            {
                "Contact group": label,
                "Score needed": round(thresholds[key], 3),
                "This lead": "yes" if score >= thresholds[key] else "no",
                "Buyers among contacted (test)": f"{results.loc[key, 'precision']:.0%}",
                "Better than random (test)": f"{results.loc[key, 'lift']:.1f}x",
            }
        )
    return pd.DataFrame(rows)


def show_verdict(score: float, metadata: dict, operating_points: pd.DataFrame) -> None:
    """Headline message: the most selective contact group this lead belongs to."""
    thresholds = metadata["operating_points"]
    results = test_results(operating_points)

    for key, label in CONTACT_GROUPS.items():  # most selective group first
        if score >= thresholds[key]:
            st.success(
                f"**{label}** — contact this lead. In the test set, "
                f"{results.loc[key, 'precision']:.0%} of the customers in this group bought the package, "
                f"{results.loc[key, 'lift']:.1f} times the average rate."
            )
            return

    if score >= thresholds[BALANCED_POINT[0]]:
        st.info(
            "**Outside the top 20%, but above the balanced threshold** — worth contacting when the "
            "calling budget is larger than 20% of customers."
        )
    else:
        st.warning("**Low priority** — outside the top 20% and below the balanced threshold.")


# =============================================================================
# Page
# =============================================================================

def model_info(settings: AppSettings, metadata: dict) -> None:
    """Collapsible section describing which model answers, so results can be traced."""
    with st.expander("About the model"):
        st.markdown(
            f"- Model repository: `{settings.model_repo}` at revision `{settings.model_revision[:8]}`\n"
            f"- Selected in training run: `{metadata['selected_run']}`\n"
            f"- Trained on dataset revision `{metadata['data_revision'][:8]}`; "
            f"{metadata['prevalence']:.0%} of training customers bought the package\n"
            f"- Uses only information known before contact: "
            f"{len(feature_order(metadata))} customer features, no sales-pitch details\n"
            f"- Python {metadata['python_version']}, "
            + ", ".join(f"{package} {v}" for package, v in metadata["packages"].items())
        )


def main() -> None:
    """Load and check the model, show the form, and explain the score of a submitted lead."""
    st.set_page_config(page_title="Score a Lead – Wellness Tourism", page_icon="🧳")
    st.title("Score a Lead")
    st.write(
        "A lead is a customer who has not been contacted yet. Enter the profile of one lead to see whether "
        "they belong to the group most likely to buy the Wellness Tourism Package, so the sales team knows "
        "whom to contact first. Only information known before the first contact is used."
    )

    settings = read_settings()
    if not settings.model_revision:
        st.error(
            "No model version configured. Set the Space variable MODEL_REVISION to a commit of the "
            "model repository (the hosting script does this)."
        )
        st.stop()

    try:
        model, metadata, operating_points = load_bundle(
            settings.model_repo, settings.model_revision, settings.token
        )
    except Exception:  # shown generically on purpose: the details stay in the server log
        st.error(
            "The model could not be loaded. Check that the Space secret HF_TOKEN can read the model "
            "repository and that MODEL_REVISION is an existing commit."
        )
        raise

    problems = version_mismatches(metadata) + parity_mismatches(model, metadata)
    if problems:
        st.error("The app stopped because it would not reproduce the trained model's scores:\n\n- "
                 + "\n- ".join(problems))
        st.stop()

    customer = customer_form(metadata)
    if customer is not None:
        outside = out_of_range_fields(customer, metadata)
        if outside:
            st.error(
                "**No score for this lead.** The model was trained on customers within these ranges "
                "and cannot reliably assess values outside them:\n\n- " + "\n- ".join(outside)
            )
            model_info(settings, metadata)
            st.stop()

        score = score_customer(model, customer)
        st.header("Result")
        show_verdict(score, metadata, operating_points)
        st.metric("Ranking score", f"{score:.3f}", help="Between 0 and 1. Higher means more similar to "
                  "customers who bought. It is a ranking, not a probability of buying.")
        st.dataframe(operating_point_view(score, metadata, operating_points), hide_index=True)

    model_info(settings, metadata)


if __name__ == "__main__":
    main()
