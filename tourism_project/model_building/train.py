"""Model training stage of the Visit with Us MLOps pipeline.

Context
    Visit with Us wants to know, before the sales team contacts a customer, whether that customer
    is likely to buy the Wellness Tourism Package (column ProdTaken, 1 = bought). Marketing can
    then contact the most promising customers first. The pipeline has four stages:

        data_register.py  ->  prep.py            ->  train.py        ->  deployment
        raw data on Hub       train/test on Hub      model on Hub        Streamlit app

    This script is the third stage. It compares several models, picks one with rules that were
    fixed before any model was trained, measures how well it predicts new customers, and
    registers it in a private Hugging Face model repository for the app to use. Every model and
    every tuning attempt is recorded in MLflow, so the choice can be traced afterwards. The same
    code runs from the notebook (%run -m) and as a job in GitHub Actions (python -m).

What it does
    1. Load train      - read train.csv from the Hub (test.csv stays untouched until step 6)
    2. Build models    - five candidate pipelines: a "guess the base rate" floor, and random
                         forest and XGBoost models with default and with tuned settings
    3. Evaluate        - tune where needed and cross-validate every candidate; log all
                         hyperparameters and metrics to MLflow
    4. Select          - a minimum-quality gate, ranking and a head-to-head final between
                         the two best models
    5. Operating points- turn model scores into contact decisions: best-F1 threshold and the
                         thresholds for contacting the top 5%, 10% or 20% of customers
    6. Test            - one final, honest measurement on the untouched test set
    7. Register        - upload the model, its metadata and the operating points to the Hub

Why these choices
    - Ranking quality, measured with PR-AUC (average precision), decides which model wins. With
      only ~19% buyers, what matters is whether buyers end up at the top of the list; PR-AUC
      measures exactly that and, unlike F1, does not depend on a chosen probability threshold.
    - Cross-validation folds are group-aware: rows that are near-copies of each other (see
      prep.py) always stay in the same fold, so a model is never validated on a copy of a
      customer it was trained on.
    - The test set is used exactly once, after the model has been chosen. Choosing with the test
      set would turn it into another validation set, and no honest score would be left.

How the file is organised
    Settings come from the central Config (config.py, section "Model training") and are passed to
    every function. Each step has a header with its input, output and reasoning, followed by
    small functions that each do one task. main() at the bottom connects the steps.

Running it
    python -m tourism_project.model_building.train   (from the project root)
    Needs a Hugging Face token with write access, from HF_TOKEN or `hf auth login`.
    Takes several minutes: two hyperparameter searches of 30 combinations x 5 folds each.
    Prints aggregate results only, never customer rows, because notebook output and workflow
    logs are public.
"""
import json
import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import version

# MLflow prints a notice meant for AI coding assistants on import; it is not useful here.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import disable_progress_bars
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from tourism_project.ci import export_github_output, markdown_table, write_step_summary
from tourism_project.config import Config
from tourism_project.hub import ensure_private_repo, load_split, resolve_data_revision


# =============================================================================
# Data containers
# =============================================================================
#
# Two small dataclasses keep related values together as they move through the steps:
# a Candidate describes a model before training, a CandidateResult holds what was measured.
#

@dataclass
class Candidate:
    """One model on the ladder: its name, role, untrained pipeline and optional search space."""

    name: str            # MLflow run name, e.g. "xgboost_tuned"
    family: str          # "dummy" | "random_forest" | "xgboost"
    stage: str           # "baseline" (default settings) or "tuned" (hyperparameter search)
    pipeline: Pipeline   # preprocessing + model, not yet trained
    search_space: dict | None = None


@dataclass
class CandidateResult:
    """Everything measured for one candidate, plus its selection outcome."""

    candidate: Candidate
    pipeline: Pipeline             # with the best hyperparameters for tuned candidates
    model_params: dict             # hyperparameters of the model step, as logged
    metrics: dict                  # cv_* and train_* metrics, as logged
    run_id: str                    # MLflow run that holds the details
    status: str = "candidate"      # reference | rejected_gate | ranked_out | finalist | selected
    reason: str = ""               # why a candidate was rejected or ranked out
    finale: dict = field(default_factory=dict)  # finale_* metrics for the two finalists


# =============================================================================
# Step 1 - Load train
# =============================================================================
#
# Input : the dataset repository on the Hub, at DATA_REVISION (set by the data preparation job
#         in GitHub Actions) or at the newest commit when run from the notebook
# Output: features X, target y and the group id of every training row, plus the revision used
#
# Only train.csv is loaded here. test.csv is read for the first time in step 6, after the model
# has been chosen, so nothing about the test set can influence any decision before that.
#

def split_columns(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate a split file into features, target and group id."""
    return frame[cfg.feature_columns], frame[cfg.target], frame[cfg.group_column]


def load_training_data(
    api: HfApi, cfg: Config
) -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """Load train.csv from the Hub; return features, target, groups and the data revision."""
    data_revision = resolve_data_revision(api, cfg)
    train = load_split(cfg, cfg.train_file, data_revision)
    features, target, groups = split_columns(train, cfg)
    return features, target, groups, data_revision


# =============================================================================
# Step 2 - Build models
# =============================================================================
#
# Input : the training target (to measure the class imbalance)
# Output: five candidates, each a Pipeline of the same preprocessing followed by a model
#
# Preprocessing is part of every pipeline, so it is fitted inside each cross-validation fold on
# that fold's training rows only. Fitting it once on all training rows would let the validation
# rows influence, for example, the median used to fill in missing incomes.
#   - numeric features    : fill missing values (the 7 cleaned entry errors) with the median
#   - binary features     : fill missing values with the most frequent value (none occur today,
#                           but the app may receive incomplete input)
#   - categorical features: fill missing values with the most frequent value, then one-hot
#                           encode; categories never seen in training are ignored instead of
#                           causing an error
#
# The candidate ladder, from simple to complex:
#   dummy_baseline         always predicts the share of buyers; the floor every model must beat
#   randomforest_baseline  random forest with default settings
#   xgboost_baseline       gradient boosting with default settings
#   randomforest_tuned     random forest with searched hyperparameters
#   xgboost_tuned          gradient boosting with searched hyperparameters
# Every model except the dummy is a candidate; a baseline may win if tuning does not help.
#
# Class imbalance (about 81% non-buyers) is handled by weighting buyers more heavily:
# class_weight="balanced" for the tree models, scale_pos_weight = non-buyers / buyers for
# XGBoost. That weight is fixed, not searched: the imbalance is a known fact, and the trade-off
# between precision and recall is set later with the decision threshold (step 5).
#

def build_preprocessor(cfg: Config) -> ColumnTransformer:
    """Fill in missing values per feature type and one-hot encode the categorical features."""
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(cfg.numeric_features)),
            ("binary", SimpleImputer(strategy="most_frequent"), list(cfg.binary_features)),
            ("categorical", categorical_steps, list(cfg.categorical_features)),
        ]
    )


def build_pipeline(model: object, cfg: Config) -> Pipeline:
    """Put the shared preprocessing in front of a model; the step names are used in search spaces."""
    return Pipeline([("preprocess", build_preprocessor(cfg)), ("model", model)])


def model_ladder(target: pd.Series, cfg: Config) -> list[Candidate]:
    """Return the five candidates, from the base-rate floor to the tuned models."""
    # Weight for buyers in XGBoost: how many non-buyers there are per buyer (about 4.2).
    buyer_weight = float((target == 0).sum() / (target == 1).sum())
    seed = cfg.random_state

    def xgboost() -> XGBClassifier:
        return XGBClassifier(
            scale_pos_weight=buyer_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

    def random_forest() -> RandomForestClassifier:
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)

    return [
        Candidate("dummy_baseline", "dummy", "baseline",
                  build_pipeline(DummyClassifier(strategy="prior"), cfg)),
        Candidate("randomforest_baseline", "random_forest", "baseline",
                  build_pipeline(random_forest(), cfg)),
        Candidate("xgboost_baseline", "xgboost", "baseline",
                  build_pipeline(xgboost(), cfg)),
        Candidate("randomforest_tuned", "random_forest", "tuned",
                  build_pipeline(random_forest(), cfg), cfg.rf_search_space),
        Candidate("xgboost_tuned", "xgboost", "tuned",
                  build_pipeline(xgboost(), cfg), cfg.xgb_search_space),
    ]


# =============================================================================
# Step 3 - Evaluate candidates
# =============================================================================
#
# Input : the five candidates and the training data
# Output: one CandidateResult per candidate, each logged as its own MLflow run
#
# All candidates are measured on exactly the same group-aware folds (one shared
# StratifiedGroupKFold): 5 folds, each fold keeping every group of copies together and holding
# about the same share of buyers. Only then is a difference between two models a difference
# between the models, and not between the folds they happened to get.
#
# For every fold the same metrics are computed by one function, ranking_metrics:
#   pr_auc               how well buyers are ranked above non-buyers (the deciding metric)
#   roc_auc              the same idea on a scale where 0.5 means random; for reference
#   top_{q}pct_precision share of buyers among the top q% of the ranking (q = 5, 10, 20)
#   top_{q}pct_lift      that share divided by the base rate: "q% contacted, x times better
#                        than calling at random"
# The fold values are summarised as mean and standard deviation (cv_pr_auc_mean, ...).
#
# Tuned candidates first run a RandomizedSearchCV: 30 random hyperparameter combinations, each
# scored by PR-AUC on the same 5 group-aware folds. MLflow autologging is switched on only during
# that search and records the 5 best combinations as child runs of the candidate's run, so all
# tried hyperparameters stay traceable. The best combination is then measured with the same
# cross-validation function as every other candidate, so all candidates are compared alike.
#
# train_pr_auc is the score on the training rows the model was fitted on. The gap to the CV score
# (overfit_gap_pr_auc) shows how much a model memorises instead of generalises. It is logged and
# used as a tie-break, but it is not a gate: random forests always score almost perfectly on
# their own training rows, which is how they work, not a defect.
#
# Candidate runs never receive test metrics. If test scores of all models stood side by side, the
# choice would effectively be made on the test set.
#

def share_key(share: float) -> str:
    """Name of a contact share in metric keys, e.g. 0.10 -> 'top_10pct'."""
    return f"top_{round(share * 100)}pct"


def top_share_mask(scores: np.ndarray, share: float) -> np.ndarray:
    """True for the highest-scoring `share` of rows: the customers marketing would contact."""
    n_contacted = max(1, round(share * len(scores)))
    ranking = np.argsort(-scores, kind="stable")  # highest score first; ties keep row order
    mask = np.zeros(len(scores), dtype=bool)
    mask[ranking[:n_contacted]] = True
    return mask


def ranking_metrics(target: pd.Series, scores: np.ndarray, cfg: Config) -> dict:
    """PR-AUC, ROC-AUC and precision, recall, F1 and lift per contact share: the one metric implementation."""
    actual = np.asarray(target)
    base_rate = actual.mean()  # share of buyers in these rows
    metrics = {
        "pr_auc": float(average_precision_score(actual, scores)),
        "roc_auc": float(roc_auc_score(actual, scores)),
    }
    for share in cfg.contact_fractions:
        # Contacting this share of the customers: who is called, who of them buys, and how many
        # of all buyers that reaches. Precision and recall are both fixed by that one decision,
        # so F1 (their harmonic mean) summarises it in one number.
        contacted = top_share_mask(scores, share)
        buyers_contacted = actual[contacted].sum()
        precision = float(buyers_contacted / contacted.sum())
        recall = float(buyers_contacted / actual.sum())
        metrics[f"{share_key(share)}_precision"] = precision
        metrics[f"{share_key(share)}_recall"] = recall
        metrics[f"{share_key(share)}_f1"] = (
            float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        )
        metrics[f"{share_key(share)}_lift"] = float(precision / base_rate)
    return metrics


def make_folds(cfg: Config, seed: int) -> StratifiedGroupKFold:
    """Group-aware, stratified cross-validation folds with a fixed seed."""
    return StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)


def cross_validate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    folds: StratifiedGroupKFold,
    cfg: Config,
) -> list[dict]:
    """Train a fresh copy of the pipeline on each fold and return one metrics dict per fold."""
    fold_metrics = []
    for train_rows, valid_rows in folds.split(features, target, groups):
        # clone() gives an untrained copy, so no fold can reuse what another fold learned.
        model = clone(pipeline).fit(features.iloc[train_rows], target.iloc[train_rows])
        scores = model.predict_proba(features.iloc[valid_rows])[:, 1]
        fold_metrics.append(ranking_metrics(target.iloc[valid_rows], scores, cfg))
    return fold_metrics


def summarise_folds(fold_metrics: list[dict], prefix: str) -> dict:
    """Mean and standard deviation of every fold metric, e.g. cv_pr_auc_mean and cv_pr_auc_std."""
    table = pd.DataFrame(fold_metrics)
    summary = {}
    for metric in table.columns:
        summary[f"{prefix}_{metric}_mean"] = float(table[metric].mean())
        summary[f"{prefix}_{metric}_std"] = float(table[metric].std(ddof=1))
    return summary


def model_params(pipeline: Pipeline) -> dict:
    """Hyperparameters of the model step, prefixed with 'model__' like in the search spaces."""
    # The prefix also keeps these names apart from the parameters MLflow autolog records for the
    # search itself (such as its own n_jobs), which MLflow would otherwise refuse as a conflict.
    params = pipeline.named_steps["model"].get_params()
    return {f"model__{name}": value for name, value in params.items()}


def setup_tracking(cfg: Config) -> None:
    """Point MLflow at its database in the work folder and select the experiment."""
    cfg.work_dir.mkdir(parents=True, exist_ok=True)  # SQLite does not create missing folders
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)


def log_run(params: dict, metrics: dict) -> None:
    """Log parameters and metrics to the active MLflow run: the one place that writes them."""
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


def tune_pipeline(
    pipeline: Pipeline,
    search_space: dict,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    folds: StratifiedGroupKFold,
    cfg: Config,
) -> Pipeline:
    """Search hyperparameters on the group-aware folds; return an untrained pipeline with the best."""
    search = RandomizedSearchCV(
        pipeline,
        search_space,
        n_iter=cfg.search_n_iter,
        scoring=cfg.scoring,
        cv=folds,
        refit=False,  # the best settings are re-evaluated below, no need to fit them here
        random_state=cfg.random_state,
    )
    # Autolog records the best tried combinations as child runs of the active run. It is switched
    # off right after the search, otherwise every later fit would create unwanted extra runs.
    mlflow.sklearn.autolog(
        max_tuning_runs=cfg.max_tuning_runs, log_models=False, log_datasets=False, silent=True
    )
    try:
        search.fit(features, target, groups=groups)  # groups reach the group-aware folds
    finally:
        mlflow.sklearn.autolog(disable=True)
    return clone(pipeline).set_params(**search.best_params_)


def evaluate_candidate(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    folds: StratifiedGroupKFold,
    cfg: Config,
    data_revision: str,
) -> CandidateResult:
    """Tune if needed, cross-validate and log one candidate as its own MLflow run."""
    print(f"  {candidate.name:<24} ...", end="", flush=True)
    with mlflow.start_run(run_name=candidate.name) as run:
        mlflow.set_tags(
            {
                "feature_set": cfg.feature_set_name,
                "stage": candidate.stage,
                "model_family": candidate.family,
                "data_revision": data_revision,
            }
        )
        setup_params = {
            "n_features": len(cfg.feature_columns),
            "cv_folds": cfg.cv_folds,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "scoring": cfg.scoring,
        }

        pipeline = candidate.pipeline
        if candidate.search_space:
            pipeline = tune_pipeline(pipeline, candidate.search_space, features, target, groups, folds, cfg)
            setup_params["search_n_iter"] = cfg.search_n_iter

        # Cross-validated performance: the basis for every selection decision.
        metrics = summarise_folds(
            cross_validate_pipeline(pipeline, features, target, groups, folds, cfg), "cv"
        )

        # In-sample performance, only to measure how much the model memorises.
        in_sample_scores = clone(pipeline).fit(features, target).predict_proba(features)[:, 1]
        in_sample = ranking_metrics(target, in_sample_scores, cfg)
        metrics["train_pr_auc"] = in_sample["pr_auc"]
        metrics["train_roc_auc"] = in_sample["roc_auc"]
        metrics["overfit_gap_pr_auc"] = metrics["train_pr_auc"] - metrics["cv_pr_auc_mean"]

        params = model_params(pipeline)
        log_run({**setup_params, **params}, metrics)

    print(f" cv PR-AUC {metrics['cv_pr_auc_mean']:.3f} (± {metrics['cv_pr_auc_std']:.3f})", flush=True)
    return CandidateResult(candidate, pipeline, params, metrics, run.info.run_id)


# =============================================================================
# Step 4 - Select
# =============================================================================
#
# Input : the five CandidateResults
# Output: the selected CandidateResult; every candidate's outcome is tagged in MLflow
#
# A. Gate. A candidate is rejected when it does not beat the dummy's CV PR-AUC by at least 0.10:
#    a model barely better than guessing the base rate is useless for choosing whom to call.
#    There is no gate on the fold-to-fold spread of PR-AUC. A trial run showed that with ~130
#    buyers per validation fold every good model varies by 0.06-0.09, so such a gate would reject
#    all of them; whether two models really differ is decided in the final (C) instead.
# B. Ranking. Candidates that pass are sorted by cv_pr_auc_mean; the best two are finalists.
# C. Final. Both finalists are cross-validated again on 15 folds (the 5-fold arrangement repeated
#    with 3 different seeds), always on the same folds, and their PR-AUC difference is computed
#    per fold.
#      - Clear winner: the average difference is larger than one standard error of the
#        differences -> the higher one wins.
#      - Tie: otherwise decide, in this order, on higher precision in the top 10%, lower CV
#        standard deviation, smaller overfit gap, and finally the simpler model family.
#    The standard error over overlapping folds is optimistic (folds share training rows), so it
#    is only used to decide whether the tie-break rules apply, never to claim that one model is
#    "significantly" better.
#
# Each candidate's outcome is written to MLflow as the tag selection_status (with
# rejected_reason where relevant), so the decision can be read back in the MLflow UI.
#

def apply_gates(results: list[CandidateResult], cfg: Config) -> list[CandidateResult]:
    """Mark the dummy as reference, reject candidates that fail the gate, return the rest."""
    dummy = next(result for result in results if result.candidate.family == "dummy")
    dummy.status = "reference"

    passed = []
    for result in results:
        if result is dummy:
            continue
        lift = result.metrics["cv_pr_auc_mean"] - dummy.metrics["cv_pr_auc_mean"]
        result.metrics["lift_over_dummy"] = lift
        if lift < cfg.min_lift_over_dummy:
            result.status, result.reason = "rejected_gate", f"lift_over_dummy={lift:.3f}"
        else:
            passed.append(result)
    return passed


def run_finale(
    first: CandidateResult,
    second: CandidateResult,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    cfg: Config,
) -> None:
    """Cross-validate both finalists on the same 15 folds and store the finale_* metrics."""
    per_fold = {first.candidate.name: [], second.candidate.name: []}
    for seed in cfg.finale_seeds:
        folds = make_folds(cfg, seed)  # same seed -> identical folds for both finalists
        for finalist in (first, second):
            per_fold[finalist.candidate.name] += cross_validate_pipeline(
                finalist.pipeline, features, target, groups, folds, cfg
            )

    tiebreak = f"{share_key(cfg.tiebreak_contact_fraction)}_precision"
    first_table = pd.DataFrame(per_fold[first.candidate.name])
    second_table = pd.DataFrame(per_fold[second.candidate.name])
    differences = first_table["pr_auc"] - second_table["pr_auc"]  # paired: same fold, two models

    for finalist, table in ((first, first_table), (second, second_table)):
        finalist.finale = {
            "finale_pr_auc_mean": float(table["pr_auc"].mean()),
            f"finale_{tiebreak}_mean": float(table[tiebreak].mean()),
        }
    first.finale["finale_diff_mean"] = float(differences.mean())
    first.finale["finale_diff_se"] = float(differences.std(ddof=1) / np.sqrt(len(differences)))


def break_tie(first: CandidateResult, second: CandidateResult, cfg: Config) -> tuple[CandidateResult, str]:
    """Pick the winner of the final and describe which rule decided."""
    diff, se = first.finale["finale_diff_mean"], first.finale["finale_diff_se"]
    if abs(diff) > cfg.tie_tolerance_se * se:
        winner = first if diff > 0 else second
        return winner, f"clear winner: PR-AUC difference {abs(diff):.4f} > {cfg.tie_tolerance_se:g} SE ({se:.4f})"

    tiebreak = f"finale_{share_key(cfg.tiebreak_contact_fraction)}_precision_mean"
    rules = [  # (description, value per finalist, higher is better?)
        (tiebreak, lambda r: r.finale[tiebreak], True),
        ("cv_pr_auc_std", lambda r: r.metrics["cv_pr_auc_std"], False),
        ("overfit_gap_pr_auc", lambda r: r.metrics["overfit_gap_pr_auc"], False),
        ("simplicity", lambda r: cfg.simplicity_order.index(r.candidate.family), False),
    ]
    for name, value, higher_is_better in rules:
        a, b = value(first), value(second)
        if a != b:
            winner = first if (a > b) == higher_is_better else second
            return winner, f"tie within {cfg.tie_tolerance_se:g} SE; decided by {name} ({a:.4f} vs {b:.4f})"
    return first, "tie on every rule; kept the higher-ranked finalist"


def tag_selection(results: list[CandidateResult]) -> None:
    """Write each candidate's selection outcome and the gate/finale metrics back to MLflow."""
    client = MlflowClient()
    for result in results:
        client.set_tag(result.run_id, "selection_status", result.status)
        if result.reason:
            client.set_tag(result.run_id, "rejected_reason", result.reason)
        extra_metrics = {**result.finale}
        if "lift_over_dummy" in result.metrics:
            extra_metrics["lift_over_dummy"] = result.metrics["lift_over_dummy"]
        for name, value in extra_metrics.items():
            client.log_metric(result.run_id, name, value)


def select_model(
    results: list[CandidateResult],
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    cfg: Config,
) -> tuple[CandidateResult, str]:
    """Run gate, ranking and final; return the selected candidate and the deciding rule."""
    # A. Gate
    passed = apply_gates(results, cfg)
    if not passed:
        tag_selection(results)
        raise RuntimeError("No candidate passed the gates: no model learns more than the base rate")

    # B. Ranking
    ranked = sorted(passed, key=lambda result: result.metrics["cv_pr_auc_mean"], reverse=True)
    for position, result in enumerate(ranked[2:], start=3):
        result.status, result.reason = "ranked_out", f"rank {position} on cv_pr_auc_mean"

    # C. Final (only needed when there are two finalists)
    if len(ranked) == 1:
        selected, decision = ranked[0], "only candidate that passed the gates"
    else:
        first, second = ranked[0], ranked[1]
        print(f"  final: {first.candidate.name} vs {second.candidate.name} on "
              f"{len(cfg.finale_seeds) * cfg.cv_folds} folds ...", flush=True)
        run_finale(first, second, features, target, groups, cfg)
        first.status = second.status = "finalist"
        selected, decision = break_tie(first, second, cfg)

    selected.status = "selected"
    tag_selection(results)
    return selected, decision


# =============================================================================
# Step 5 - Operating points
# =============================================================================
#
# Input : the selected pipeline and the training data
# Output: a decision threshold per operating point, and a table with what each threshold means
#
# The model produces a score per customer; marketing needs a yes/no decision. A threshold turns
# scores into decisions, and which threshold is right depends on the situation:
#   max_f1      the threshold with the best balance between precision and recall (F1); a neutral
#               choice when the contact budget is unknown
#   top_5pct,   the threshold that selects the 5%, 10% or 20% highest-scoring customers; for a
#   top_10pct,  fixed contact budget (e.g. 10,000 calls), where the budget sets the threshold
#   top_20pct
# No single operating point is "the" answer: marketing picks one once the budget is known, and
# the model does not need retraining for that.
#
# Thresholds are set on out-of-fold scores: every training row is scored by a model that did not
# see it during training (cross_val_predict on the same group-aware folds). Scores of rows a model
# was trained on are too optimistic and would give thresholds that do not hold for new customers.
#
# Columns of the table:
#   threshold        score from which a customer is contacted
#   share_contacted  share of customers at or above the threshold
#   precision        share of contacted customers who buy
#   recall           share of all buyers that is contacted
#   lift             precision divided by the base rate: how many times better than calling at random
#   pct_of_ceiling   precision relative to the best possible precision for that share; when more
#                    customers are contacted than there are buyers, precision cannot reach 1
#

def out_of_fold_scores(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    folds: StratifiedGroupKFold,
) -> np.ndarray:
    """Score every training row with a model that was not trained on that row."""
    probabilities = cross_val_predict(
        pipeline, features, target, groups=groups, cv=folds, method="predict_proba"
    )
    return probabilities[:, 1]


def threshold_for_max_f1(target: pd.Series, scores: np.ndarray) -> float:
    """Threshold with the highest F1 score (balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(target, scores)
    # The last precision/recall pair has no threshold; drop it so the arrays line up.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def threshold_for_top_share(scores: np.ndarray, share: float) -> float:
    """Score of the last customer inside the top `share` of the ranking."""
    n_contacted = max(1, round(share * len(scores)))
    return float(np.sort(scores)[::-1][n_contacted - 1])


def operating_thresholds(target: pd.Series, scores: np.ndarray, cfg: Config) -> dict:
    """All operating points and their thresholds, determined on out-of-fold scores."""
    thresholds = {"max_f1": threshold_for_max_f1(target, scores)}
    for share in cfg.contact_fractions:
        thresholds[share_key(share)] = threshold_for_top_share(scores, share)
    return thresholds


def decision_metrics(target: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """What contacting every customer at or above the threshold would achieve."""
    actual = np.asarray(target)
    contacted = scores >= threshold
    base_rate = actual.mean()
    share_contacted = contacted.mean()
    precision = actual[contacted].mean() if contacted.any() else np.nan
    recall = actual[contacted].sum() / actual.sum()
    ceiling = min(1.0, base_rate / share_contacted) if share_contacted > 0 else np.nan
    return {
        "threshold": float(threshold),
        "share_contacted": float(share_contacted),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0,
        "lift": float(precision / base_rate),
        "pct_of_ceiling": float(precision / ceiling),
    }


def operating_point_table(target: pd.Series, scores: np.ndarray, thresholds: dict) -> pd.DataFrame:
    """One row per operating point with the metrics of decision_metrics."""
    rows = {name: decision_metrics(target, scores, threshold) for name, threshold in thresholds.items()}
    return pd.DataFrame(rows).T.rename_axis("operating_point")


# =============================================================================
# Step 6 - Test
# =============================================================================
#
# Input : the selected pipeline, the full training data, the thresholds from step 5, test.csv
# Output: test metrics, the operating-point table on test, uncertainty ranges and final checks
#
# The selected pipeline is trained once more on all training rows. test.csv is loaded here for the
# first time, and every threshold from step 5 is applied unchanged, exactly as it would be to new
# customers in production.
#
# Uncertainty ranges: the top 10% of the test set is only about 83 customers, so a single
# precision number would suggest more certainty than there is. The test set is resampled 1,000
# times with replacement; the 2.5th and 97.5th percentiles give a 95% range.
#
# Two fixed checks, each with a fixed response:
#   - CV was reliable: test PR-AUC within 0.05 of the CV estimate. Otherwise a warning is printed;
#     a different model is NOT chosen, because that would turn the test set into a validation set.
#   - No leakage: test ROC-AUC at most 0.90. Pre-contact customer profiles cannot plausibly predict
#     a purchase better than that; a higher value points to leaked information. The script then
#     stops before registering the model.
# The test result never changes which model was chosen; it only confirms the process or exposes
# an error in it.
#

def evaluate_on_test(
    model: Pipeline, test: pd.DataFrame, thresholds: dict, cfg: Config
) -> tuple[dict, pd.DataFrame, np.ndarray, pd.Series]:
    """Score the test set once; return test metrics, the operating-point table, scores and target."""
    features, target, _groups = split_columns(test, cfg)
    scores = model.predict_proba(features)[:, 1]

    table = operating_point_table(target, scores, thresholds)
    ranking = ranking_metrics(target, scores, cfg)
    predicted_buyer = scores >= thresholds["max_f1"]

    metrics = {
        "test_pr_auc": ranking["pr_auc"],
        "test_roc_auc": ranking["roc_auc"],
        "test_accuracy": float((predicted_buyer == np.asarray(target)).mean()),  # reference only
    }
    for point, row in table.iterrows():
        for column in ("precision", "recall", "lift", "pct_of_ceiling", "share_contacted"):
            metrics[f"test_{point}_{column}"] = float(row[column])
    return metrics, table, scores, target


def bootstrap_intervals(target: pd.Series, scores: np.ndarray, thresholds: dict, cfg: Config) -> dict:
    """95% ranges for precision and lift of each top-share operating point on the test set."""
    rng = np.random.default_rng(cfg.random_state)
    actual = np.asarray(target)
    intervals = {}
    for share in cfg.contact_fractions:
        point = share_key(share)
        precisions, lifts = [], []
        for _ in range(cfg.n_bootstrap):
            rows = rng.integers(0, len(actual), len(actual))  # resample with replacement
            resampled = decision_metrics(actual[rows], scores[rows], thresholds[point])
            precisions.append(resampled["precision"])
            lifts.append(resampled["lift"])
        for name, values in (("precision", precisions), ("lift", lifts)):
            low, high = np.nanpercentile(values, [2.5, 97.5])
            intervals[f"test_{point}_{name}_ci_low"] = float(low)
            intervals[f"test_{point}_{name}_ci_high"] = float(high)
    return intervals


def check_final(test_metrics: dict, cv_pr_auc_mean: float, cfg: Config) -> list[str]:
    """Run the two final checks: return warnings, or raise when leakage is suspected."""
    warnings = []
    gap = abs(test_metrics["test_pr_auc"] - cv_pr_auc_mean)
    test_metrics["cv_test_gap_pr_auc"] = gap
    if gap > cfg.max_cv_test_gap:
        warnings.append(
            f"test PR-AUC differs {gap:.3f} from the CV estimate (limit {cfg.max_cv_test_gap}); "
            "the CV estimate was less reliable than expected"
        )
    if test_metrics["test_roc_auc"] > cfg.leakage_roc_auc_ceiling:
        raise RuntimeError(
            f"test ROC-AUC {test_metrics['test_roc_auc']:.3f} exceeds {cfg.leakage_roc_auc_ceiling}: "
            "suspected data leakage. The model is NOT registered; check the features."
        )
    return warnings


def log_final_run(
    selected: CandidateResult,
    oof_table: pd.DataFrame,
    test_table: pd.DataFrame,
    test_metrics: dict,
    cfg: Config,
    data_revision: str,
) -> str:
    """Log the selected model's full evaluation as the run final_<name>; return its run id."""
    with mlflow.start_run(run_name=f"final_{selected.candidate.name}") as run:
        mlflow.set_tags(
            {
                "feature_set": cfg.feature_set_name,
                "stage": "final",
                "model_family": selected.candidate.family,
                "selection_status": "selected",
                "data_revision": data_revision,
            }
        )
        oof_metrics = {
            f"oof_{point}_{column}": float(value)
            for point, row in oof_table.iterrows()
            for column, value in row.items()
        }
        metrics = {**selected.metrics, **selected.finale, **oof_metrics, **test_metrics}
        log_run({"n_features": len(cfg.feature_columns), **selected.model_params}, metrics)

        # The operating-point table as a file: aggregated numbers per operating point, no rows
        # per customer.
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        table_path = cfg.model_dir / cfg.operating_points_file
        combined_operating_points(oof_table, test_table).to_csv(table_path)
        mlflow.log_artifact(str(table_path))
    return run.info.run_id


def combined_operating_points(oof_table: pd.DataFrame, test_table: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold and test operating points in one table, marked by a 'split' column."""
    return pd.concat(
        [oof_table.assign(split="train_out_of_fold"), test_table.assign(split="test")]
    ).reset_index().set_index(["split", "operating_point"])


# =============================================================================
# Step 7 - Register
# =============================================================================
#
# Input : the trained model, thresholds, test results and the MLflow run id
# Output: model.joblib, model_metadata.json and operating_points.csv in the private model
#         repository on the Hub; the commit id is the MODEL_REVISION
#
# Registering means publishing the chosen model to the place the app loads it from: a private
# model repository on the Hugging Face Hub. Like the dataset repository it keeps every upload as a
# commit, so the app can pin an exact model version (MODEL_REVISION) and a new training run never
# changes the app by surprise.
#
# model_metadata.json holds everything the app needs besides the model itself, so the app does
# not keep its own copy of feature lists or thresholds that could drift apart from the model:
#   - package and Python versions: a saved model only loads reliably with the same versions
#   - feature lists, the value range of every numeric feature and the category values the model
#     knows, so the app's input form only offers values the model has seen during training
#   - the operating-point thresholds and the base rate
#   - where the model came from: selected run, MLflow run id, data revision
#   - parity examples: a few made-up customer profiles with the score this model gives them, so
#     the deployed app can verify it reproduces exactly the same scores. They are synthetic
#     (medians and most common values), not real customers.
#

def input_bounds(features: pd.DataFrame, cfg: Config) -> dict:
    """Smallest and largest training value of each numeric feature, and whether it is a whole number."""
    # A tree model cannot extrapolate: for an age or income outside the range it was trained on it
    # simply repeats the prediction for the nearest value it knows. The app therefore limits every
    # numeric input field to the range seen in training. Missing values are ignored here.
    bounds = {}
    for column in cfg.numeric_features:
        values = features[column].dropna()
        bounds[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "integer": bool((values % 1 == 0).all()),  # e.g. Age and CityTier take whole numbers only
        }
    return bounds


def category_levels(model: Pipeline, cfg: Config) -> dict:
    """Categories per categorical feature as learned by the fitted one-hot encoder."""
    encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
    return {
        feature: [str(level) for level in levels]
        for feature, levels in zip(cfg.categorical_features, encoder.categories_)
    }


def parity_examples(model: Pipeline, features: pd.DataFrame, cfg: Config) -> list[dict]:
    """A few synthetic customer profiles with the score the trained model gives them."""
    # A "typical" profile: median for numeric features, most common value for the others.
    typical = {column: float(features[column].median()) for column in cfg.numeric_features}
    for column in (*cfg.binary_features, *cfg.categorical_features):
        value = features[column].mode().iloc[0]
        typical[column] = int(value) if column in cfg.binary_features else str(value)

    # Two variations, so the check also covers a different binary and categorical value.
    with_passport = {**typical, "Passport": 1 - typical["Passport"]}
    levels = category_levels(model, cfg)["Designation"]
    other_designation = {**typical, "Designation": next(l for l in levels if l != typical["Designation"])}

    examples = []
    for profile in (typical, with_passport, other_designation):
        score = model.predict_proba(pd.DataFrame([profile])[cfg.feature_columns])[:, 1][0]
        examples.append({"input": profile, "expected_score": float(score)})
    return examples


def build_metadata(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    thresholds: dict,
    selected: CandidateResult,
    final_run_id: str,
    data_revision: str,
    cfg: Config,
) -> dict:
    """Everything the app needs to use the model correctly, as a JSON-ready dict."""
    return {
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "packages": {
            package: version(package)
            for package in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib")
        },
        "feature_columns": {
            "numeric": list(cfg.numeric_features),
            "binary": list(cfg.binary_features),
            "categorical": list(cfg.categorical_features),
        },
        "input_bounds": input_bounds(features, cfg),
        "category_levels": category_levels(model, cfg),
        "operating_points": thresholds,
        "prevalence": float(target.mean()),
        "selected_run": f"final_{selected.candidate.name}",
        "mlflow_run_id": final_run_id,
        "data_revision": data_revision,
        "parity_examples": parity_examples(model, features, cfg),
    }


def save_model_bundle(model: Pipeline, metadata: dict, cfg: Config) -> list:
    """Save model and metadata next to operating_points.csv in the model folder; return the paths."""
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / cfg.model_file
    metadata_path = cfg.model_dir / cfg.metadata_file
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [model_path, metadata_path, cfg.model_dir / cfg.operating_points_file]


def upload_model_bundle(api: HfApi, cfg: Config, paths: list) -> str:
    """Upload the bundle to the private model repository in one commit; return the commit id."""
    ensure_private_repo(api, cfg.hf_model_repo, repo_type="model")
    # The model file is ~20 MB; the Hub library would draw upload progress bars that fill the
    # notebook output and the workflow log with dozens of lines, so they are switched off.
    disable_progress_bars()
    # One commit for all files, so every model version on the Hub has matching metadata and
    # operating points.
    commit = api.create_commit(
        repo_id=cfg.hf_model_repo,
        repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path) for path in paths],
        commit_message="Register selected model",
    )
    return commit.oid


# =============================================================================
# Reporting
# =============================================================================
#
# Everything printed here also goes to the GitHub Actions run summary (when running there), so the
# results can be read without opening MLflow. Only aggregated numbers are shown.
#

def runs_overview(results: list[CandidateResult], cfg: Config) -> pd.DataFrame:
    """One row per candidate: selection outcome and CV metrics next to its tuned hyperparameters."""
    searched = sorted({name for space in (cfg.rf_search_space, cfg.xgb_search_space) for name in space})
    rows = {}
    for result in results:
        row = {
            "status": result.status,
            "cv_pr_auc_mean": result.metrics["cv_pr_auc_mean"],
            "cv_pr_auc_std": result.metrics["cv_pr_auc_std"],
            "cv_roc_auc_mean": result.metrics["cv_roc_auc_mean"],
            # Precision, recall and F1 of one concrete decision: contacting the top 10% of the
            # ranking. They make the ranking metrics above tangible without fixing a threshold yet.
            **{
                f"cv_top10_{name}": result.metrics[
                    f"cv_{share_key(cfg.tiebreak_contact_fraction)}_{name}_mean"
                ]
                for name in ("precision", "recall", "f1")
            },
            "overfit_gap_pr_auc": result.metrics["overfit_gap_pr_auc"],
        }
        # Only hyperparameters that were searched are shown; a column stays empty for models
        # that do not have that hyperparameter.
        for name in searched:
            if name in result.model_params:
                row[name.removeprefix("model__")] = result.model_params[name]
        rows[result.candidate.name] = row
    table = pd.DataFrame(rows).T.rename_axis("run")
    return table.sort_values("cv_pr_auc_mean", ascending=False)


def report(title: str, body: str | pd.DataFrame) -> None:
    """Print a titled block and add the same block to the GitHub Actions summary."""
    text = body.to_string() if isinstance(body, pd.DataFrame) else body
    print(f"\n=== {title} ===\n{text}")
    markdown = markdown_table(body) if isinstance(body, pd.DataFrame) else f"```\n{body}\n```"
    write_step_summary(f"### {title}\n\n{markdown}")


def confusion_text(target: pd.Series, scores: np.ndarray, threshold: float) -> str:
    """Confusion matrix at a threshold, as readable text."""
    (tn, fp), (fn, tp) = confusion_matrix(target, scores >= threshold, labels=[0, 1])
    return (f"contacted & bought: {tp:>4}   contacted, did not buy: {fp:>4}\n"
            f"not contacted, would have bought: {fn:>4}   not contacted, did not buy: {tn:>4}")


# =============================================================================
# Run all steps
# =============================================================================
#
# main() is the only place where the steps are connected. The order is the protocol: test.csv is
# loaded only after the model and its thresholds have been fixed.
#

def main() -> None:
    """Run steps 1 to 7 in order."""
    cfg = Config()
    api = HfApi()  # picks up the token from HF_TOKEN or `hf auth login`
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    setup_tracking(cfg)

    # 1. Load train
    features, target, groups, data_revision = load_training_data(api, cfg)
    print(f"Train data    : {len(features)} rows, {target.mean():.1%} buyers, "
          f"revision {data_revision[:8]}")
    folds = make_folds(cfg, cfg.random_state)  # shared by every candidate

    # 2. Build models
    ladder = model_ladder(target, cfg)

    # 3. Evaluate candidates
    print(f"\nEvaluating {len(ladder)} candidates on {cfg.cv_folds} group-aware folds "
          f"(tuned models try {cfg.search_n_iter} combinations first):")
    results = [
        evaluate_candidate(candidate, features, target, groups, folds, cfg, data_revision)
        for candidate in ladder
    ]

    # 4. Select
    selected, decision = select_model(results, features, target, groups, cfg)
    report("Candidates (hyperparameters next to CV metrics)", runs_overview(results, cfg))
    rejected = [f"{r.candidate.name}: {r.status} ({r.reason})" for r in results if r.reason]
    report("Selection", f"selected: {selected.candidate.name}\nrule    : {decision}"
           + ("\n" + "\n".join(rejected) if rejected else ""))

    # 5. Operating points on out-of-fold scores
    oof_scores = out_of_fold_scores(selected.pipeline, features, target, groups, folds)
    thresholds = operating_thresholds(target, oof_scores, cfg)
    oof_table = operating_point_table(target, oof_scores, thresholds)
    report("Operating points (train, out-of-fold)", oof_table)

    # 6. Test: train on all training rows, then load and score the test set once
    model = clone(selected.pipeline).fit(features, target)
    test = load_split(cfg, cfg.test_file, data_revision)
    test_metrics, test_table, test_scores, test_target = evaluate_on_test(model, test, thresholds, cfg)
    test_metrics.update(bootstrap_intervals(test_target, test_scores, thresholds, cfg))
    warnings = check_final(test_metrics, selected.metrics["cv_pr_auc_mean"], cfg)
    final_run_id = log_final_run(selected, oof_table, test_table, test_metrics, cfg, data_revision)

    ranges = pd.DataFrame(
        {
            share_key(share): {
                "precision": f"{test_metrics[f'test_{share_key(share)}_precision']:.3f} "
                             f"[{test_metrics[f'test_{share_key(share)}_precision_ci_low']:.3f}, "
                             f"{test_metrics[f'test_{share_key(share)}_precision_ci_high']:.3f}]",
                "lift": f"{test_metrics[f'test_{share_key(share)}_lift']:.2f} "
                        f"[{test_metrics[f'test_{share_key(share)}_lift_ci_low']:.2f}, "
                        f"{test_metrics[f'test_{share_key(share)}_lift_ci_high']:.2f}]",
            }
            for share in cfg.contact_fractions
        }
    ).T.rename_axis("operating_point")
    report("Test set (used once)",
           f"PR-AUC {test_metrics['test_pr_auc']:.3f} (CV estimate {selected.metrics['cv_pr_auc_mean']:.3f}) | "
           f"ROC-AUC {test_metrics['test_roc_auc']:.3f} | accuracy at max_f1 {test_metrics['test_accuracy']:.3f}\n"
           + ("warnings: " + "; ".join(warnings) if warnings else "checks: CV estimate reliable, no sign of leakage"))
    report("Operating points (test, thresholds from out-of-fold)", test_table)
    report("Test precision and lift with 95% ranges", ranges)
    report("Decisions at max_f1 (test)", confusion_text(test_target, test_scores, thresholds["max_f1"]))
    report("Decisions when contacting the top 10% (test)",
           confusion_text(test_target, test_scores, thresholds[share_key(cfg.tiebreak_contact_fraction)]))

    # 7. Register
    metadata = build_metadata(model, features, target, thresholds, selected, final_run_id, data_revision, cfg)
    paths = save_model_bundle(model, metadata, cfg)
    model_revision = upload_model_bundle(api, cfg, paths)
    report("Registered model",
           f"repository : https://huggingface.co/{cfg.hf_model_repo} (private)\n"
           f"files      : {', '.join(path.name for path in paths)}\n"
           f"MODEL_REVISION: {model_revision}\n"
           f"MLflow run : final_{selected.candidate.name} ({final_run_id})")

    # In GitHub Actions: tell the deployment job which model version to use.
    export_github_output("model_revision", model_revision)


if __name__ == "__main__":
    main()
