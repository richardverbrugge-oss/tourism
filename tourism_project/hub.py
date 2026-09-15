"""Shared helpers for the pipeline's Hugging Face Hub repositories.

Context
    The Hugging Face Hub is the shared storage between the pipeline stages: data registration
    and data preparation write the dataset repository, model training reads the prepared splits
    from it and writes the selected model to a model repository. The same helpers are used in
    the notebook and in GitHub Actions.

    A Hub repository keeps every upload as a commit with an id (a "revision"). Reading a file
    at a specific revision guarantees that training uses exactly the split that data
    preparation produced, even if the repository receives a newer commit later.

What this module offers
    - ensure_private_repo    : create a dataset or model repository if needed and keep it private
    - resolve_revision       : decide which revision (commit) of a repository to use
    - resolve_data_revision  : the dataset revision for training
    - resolve_model_revision : the model revision for the app
    - load_split             : download one split file at a revision and check its columns
"""
import os

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from tourism_project.config import Config


def ensure_private_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    """Create a Hub repository (dataset or model) if it does not exist, and make sure it is private."""
    # exist_ok=True: no error when the repository already exists, e.g. on every pipeline run
    # after the first one.
    api.create_repo(repo_id=repo_id, repo_type=repo_type, visibility="private", exist_ok=True)

    # create_repo ignores the visibility setting for a repository that already exists. Setting it
    # again explicitly guarantees the data or model stays private even if someone changed the
    # setting by hand on the website.
    api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, visibility="private")


def resolve_revision(
    api: HfApi, repo_id: str, repo_type: str, env_variable: str, revision: str | None = None
) -> str:
    """Return the revision to use: the one passed in, the value of env_variable, or the newest commit."""
    # 1. A revision passed in explicitly always wins.
    if revision:
        return revision

    # 2. In GitHub Actions the previous job passes on the commit it produced (DATA_REVISION from data
    #    preparation, MODEL_REVISION from training), so the next job uses exactly that version.
    revision_from_pipeline = os.environ.get(env_variable)
    if revision_from_pipeline:
        return revision_from_pipeline

    # 3. Otherwise (for example in the notebook) use the newest commit of the repository.
    return api.repo_info(repo_id, repo_type=repo_type).sha


def resolve_data_revision(api: HfApi, cfg: Config, revision: str | None = None) -> str:
    """Dataset revision for training: explicit, $DATA_REVISION, or the newest dataset commit."""
    return resolve_revision(api, cfg.hf_dataset_repo, "dataset", "DATA_REVISION", revision)


def resolve_model_revision(api: HfApi, cfg: Config, revision: str | None = None) -> str:
    """Model revision for the app: explicit, $MODEL_REVISION, or the newest model commit."""
    return resolve_revision(api, cfg.hf_model_repo, "model", "MODEL_REVISION", revision)


def load_split(cfg: Config, filename: str, revision: str) -> pd.DataFrame:
    """Download one split file (train.csv or test.csv) at a fixed revision and read it."""
    local_path = hf_hub_download(
        cfg.hf_dataset_repo, filename, repo_type="dataset", revision=revision
    )
    split = pd.read_csv(local_path)

    # The model code relies on these exact columns in this order; stop with a clear message
    # if the file on the Hub was made by a different version of prep.py.
    if list(split.columns) != cfg.split_columns:
        raise ValueError(
            f"{filename} at revision {revision[:8]} has columns {list(split.columns)}, "
            f"expected {cfg.split_columns}"
        )
    return split
