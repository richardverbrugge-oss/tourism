"""Central configuration of the Visit with Us MLOps pipeline.

Context
    Every stage of the pipeline (data registration, data preparation, and later model training,
    deployment and hosting) imports this one Config dataclass. Paths, Hugging Face repository
    names, the target and feature columns, and the constants of each stage are defined here
    once. A change is therefore made in one place and used by every stage, in the notebook as
    well as in GitHub Actions.

    The notebook writes this file with %%writefile; GitHub Actions uses the copy that is
    committed to the repository. Settings are grouped per stage, and each section names the
    script that uses it. The logic that applies a setting lives in that script, not here.
"""
from dataclasses import dataclass, field
from pathlib import Path

# Project root: this file is tourism_project/config.py, so the root is two levels up.
# In Colab that is /content, locally it is the repository folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """All pipeline settings. Frozen, so no script can change a setting by accident."""

    # =========================================================================
    # Paths and Hugging Face Hub (used by all scripts)
    # =========================================================================
    project_root: Path = PROJECT_ROOT

    hf_user: str = "richvrb"
    hf_dataset_name: str = "tourism-wellness-data"  # private: holds customer data

    # File names, identical locally and on the Hub
    raw_data_file: str = "tourism.csv"
    train_file: str = "train.csv"
    test_file: str = "test.csv"

    # =========================================================================
    # Columns (used by all scripts)
    # =========================================================================
    # What the model predicts: 1 = the customer bought a package, 0 = did not. About 19% of
    # the 4,128 customers in the dataset bought one.
    target: str = "ProdTaken"

    # Model features: customer profile information that is known BEFORE the sales team makes
    # contact. The prediction is used to decide whom to contact, so it has to work with what is
    # known at that moment.
    numeric_features: tuple[str, ...] = (
        "Age",
        "CityTier",
        "NumberOfPersonVisiting",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    )
    binary_features: tuple[str, ...] = ("Passport", "OwnCar")
    categorical_features: tuple[str, ...] = (
        "TypeofContact",
        "Occupation",
        "Gender",
        "MaritalStatus",
        "Designation",
    )

    # Identifier: not a feature, only used to check that no customer is in train and test.
    id_column: str = "CustomerID"

    # Columns that describe the sales pitch and therefore only exist AFTER contact. A model
    # trained with them would look good on paper but could not be used before contact, when
    # these values are still unknown (data leakage). They are dropped in prep.py.
    post_contact_columns: tuple[str, ...] = (
        "DurationOfPitch",
        "NumberOfFollowups",
        "PitchSatisfactionScore",
        "ProductPitched",
    )

    # Group id written by prep.py. The dataset contains near-copies of customer records; rows
    # that are copies of each other share a group id, and a group is never split across train
    # and test (or across cross-validation folds), so the model is never evaluated on a copy of
    # a customer it was trained on. It is not a model feature.
    group_column: str = "group_id"

    # =========================================================================
    # Reproducibility (used by all scripts)
    # =========================================================================
    random_state: int = 42  # same seed everywhere, so every run gives the same result

    # =========================================================================
    # Data preparation (used by model_building/prep.py)
    # =========================================================================
    # Cleaning rules (step 2), based on an inspection of the raw data. They are fixed values,
    # not computed from the data, so the test rows cannot influence how training data is cleaned.

    # Gender contains "Fe Male" (155 rows), a spelling variant of "Female".
    gender_replacements: tuple[tuple[str, str], ...] = (("Fe Male", "Female"),)

    # In the raw data every monthly income lies between 16,009 and 38,304, except three values:
    # 1,000, 4,678 and 98,678. The wide limits below catch exactly those entry errors, which
    # become NaN and are filled in later by the model's imputer.
    monthly_income_range: tuple[float, float] = (5_000.0, 50_000.0)

    # Every customer takes 1-8 trips a year, except four values of 19-22: entry errors -> NaN.
    max_number_of_trips: int = 10

    # Finding near-copies (step 3). Each row pair gets an evidence score in bits for how unlikely
    # its agreement is by chance (maximum about 31). Across all 8.5 million pairs the scores form
    # two clusters with an almost empty gap between 23 and 27 bits (73 pairs); a threshold inside
    # that gap separates copies from ordinary similarity, so its exact value hardly matters.
    copy_threshold_bits: float = 24.0
    comparison_chunk_size: int = 256  # rows compared per batch, keeps memory use low

    # Train/test split (step 4). The test set is kept aside to estimate performance on new
    # customers; the checks guard against a split that would make that estimate misleading.
    test_size: float = 0.2              # 20% of the customers go to the test set
    max_prevalence_gap: float = 0.01    # share of buyers may differ at most 1 point, train vs test
    max_test_share_gap: float = 0.02    # realised test share may differ at most 2 points from 20%

    # =========================================================================
    # Model training (used by model_building/train.py)
    # =========================================================================
    # Every rule below was fixed BEFORE any model was trained. Choosing thresholds after seeing
    # the results would let the favourite model win by construction and make the selection
    # meaningless. The reasoning behind each rule is in the header comments of train.py.

    # --- Registration: the selected model goes to a private Hugging Face model repository ---
    hf_model_name: str = "tourism-wellness-model"
    model_file: str = "model.joblib"                  # fitted pipeline: preprocessing + model
    metadata_file: str = "model_metadata.json"        # everything the app needs to use the model
    operating_points_file: str = "operating_points.csv"

    # --- Experiment tracking (MLflow) ---
    mlflow_experiment: str = "tourism-wellness-precontact"
    feature_set_name: str = "pre_contact"  # tag on every run: which features the model may use
    max_tuning_runs: int = 5               # a hyperparameter search logs only its 5 best candidates

    # --- Validation ---
    # Models are compared with cross-validation on the train set only; the test set is used
    # once, at the very end. Folds are group-aware, like the train/test split, so a copy of a
    # customer never sits in the training folds while the original is being validated.
    cv_folds: int = 5
    # The two best models are compared once more on 3 different fold arrangements (15 folds in
    # total), because with ~640 buyers in train a single 5-fold run is too noisy to separate
    # models that differ by a few hundredths.
    finale_seeds: tuple[int, ...] = (42, 43, 44)

    # Models are ranked on PR-AUC (average precision): how well they put buyers above
    # non-buyers across the whole ranking. With only ~19% buyers it is more informative than
    # ROC-AUC, and unlike F1 it does not depend on a chosen probability threshold.
    scoring: str = "average_precision"
    search_n_iter: int = 30  # hyperparameter combinations tried per tuned model

    # --- Hyperparameter search spaces ("model__" addresses the model step inside the Pipeline) ---
    rf_search_space: dict = field(default_factory=lambda: {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [None, 6, 10, 16, 24],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
        # Both options handle the 19/81 imbalance; which works better with bootstrapping
        # cannot be reasoned out in advance, so the search decides.
        "model__class_weight": ["balanced", "balanced_subsample"],
    })
    xgb_search_space: dict = field(default_factory=lambda: {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [3, 4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma": [0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [1, 5, 10],
        # scale_pos_weight is NOT searched: it is fixed at (non-buyers / buyers). The imbalance
        # is a known fact, and the precision/recall trade-off is handled by the threshold.
    })

    # --- Selection rules ---
    # Gate: CV PR-AUC must beat "guess the base rate" (the dummy model) by at least 0.10.
    # There is deliberately no gate on how much PR-AUC varies between folds: with only ~130 buyers
    # per validation fold every good model varies by 0.06-0.09, and chance differences between
    # models are handled by the final comparison on 15 paired folds instead.
    min_lift_over_dummy: float = 0.10
    tie_tolerance_se: float = 1.0      # finalists within 1 standard error count as a tie
    # Last tie-break: prefer the simpler model family (faster, easier to explain, lighter app).
    simplicity_order: tuple[str, ...] = ("random_forest", "xgboost")

    # --- Business view: marketing contacts a fixed budget of customers ---
    # The size of the customer base is unknown, so the budget is expressed as a share of it.
    # Precision ("share of contacted customers who buy") and lift ("times better than calling
    # at random") are reported for the top 5%, 10% and 20% of the model's ranking.
    contact_fractions: tuple[float, ...] = (0.05, 0.10, 0.20)
    tiebreak_contact_fraction: float = 0.10  # first tie-break between the two finalists
    n_bootstrap: int = 1_000                 # resamples for the uncertainty ranges on test

    # --- Final checks on the test set ---
    max_cv_test_gap: float = 0.05          # test PR-AUC far from the CV estimate -> warning
    leakage_roc_auc_ceiling: float = 0.90  # pre-contact features cannot plausibly score higher;
                                           # above it the script stops before registering

    # =========================================================================
    # Deployment and hosting (used by hosting/hosting.py)
    # =========================================================================
    # The Streamlit app runs in a public Hugging Face Space built from a Dockerfile. The app itself
    # (deployment/app.py) does not import this Config, because the container only contains the
    # files listed below; it reads everything it needs from model_metadata.json instead.
    hf_space_name: str = "tourism-wellness-app"
    space_app_port: int = 8501  # must equal app_port in deployment/README.md and EXPOSE in the Dockerfile

    # The only files ever uploaded to the Space: what the container needs plus the Space's own
    # configuration. A new file in deployment/ is therefore never published by accident. This
    # mirrors the whitelist in deployment/.dockerignore.
    deployment_files: tuple[str, ...] = (
        "Dockerfile",
        ".dockerignore",
        "README.md",               # Space configuration (sdk, app_port) in its front matter
        "requirements.txt",
        "app.py",
        ".streamlit/config.toml",
    )

    # Building the image and starting the app takes a few minutes; hosting.py checks the status
    # every 15 seconds and gives up after 20 minutes.
    space_build_timeout_s: int = 1_200
    space_poll_interval_s: int = 15

    # =========================================================================
    # Derived values (computed from the settings above, never set directly)
    # =========================================================================
    @property
    def hf_dataset_repo(self) -> str:
        """Full id of the dataset repository, e.g. 'richvrb/tourism-wellness-data'."""
        return f"{self.hf_user}/{self.hf_dataset_name}"

    @property
    def hf_model_repo(self) -> str:
        """Full id of the model repository, e.g. 'richvrb/tourism-wellness-model'."""
        return f"{self.hf_user}/{self.hf_model_name}"

    @property
    def hf_space_repo(self) -> str:
        """Full id of the Space, e.g. 'richvrb/tourism-wellness-app'."""
        return f"{self.hf_user}/{self.hf_space_name}"

    @property
    def space_url(self) -> str:
        """Public address of the running app, e.g. 'https://richvrb-tourism-wellness-app.hf.space'."""
        return f"https://{self.hf_user}-{self.hf_space_name}.hf.space"

    @property
    def deployment_dir(self) -> Path:
        """Folder with the files that make up the Space."""
        return self.project_root / "tourism_project" / "deployment"

    @property
    def model_dir(self) -> Path:
        """Folder where train.py saves the model bundle before uploading it."""
        return self.work_dir / "model"

    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow database location; MLflow 3 requires a database instead of a plain folder."""
        return f"sqlite:///{self.work_dir / 'mlflow.db'}"

    @property
    def data_path(self) -> Path:
        """Location of the raw CSV in the project, used when registering the dataset."""
        return self.project_root / "tourism_project" / "data" / self.raw_data_file

    @property
    def work_dir(self) -> Path:
        """Folder for generated files such as train.csv and test.csv; never committed."""
        return self.project_root / "outputs"

    @property
    def feature_columns(self) -> list[str]:
        """All model features in a fixed order."""
        return [*self.numeric_features, *self.binary_features, *self.categorical_features]

    @property
    def split_columns(self) -> list[str]:
        """Columns of train.csv and test.csv: features, target and group id, in this order."""
        return [*self.feature_columns, self.target, self.group_column]
