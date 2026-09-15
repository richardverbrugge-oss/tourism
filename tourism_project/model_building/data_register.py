"""Data registration stage of the Visit with Us MLOps pipeline.

Context
    The pipeline predicts, before the sales team contacts a customer, whether that customer is
    likely to buy the Wellness Tourism Package. It has four stages:

        data_register.py  ->  prep.py            ->  train.py        ->  deployment
        raw data on Hub       train/test on Hub      model on Hub        Streamlit app

    This script is the first stage. "Registering" the data means placing the raw file in one
    central, versioned location that every later stage reads from. That location is a dataset
    repository on the Hugging Face Hub. A Hub repository works like a Git repository: every
    upload is a commit with an id, so each version of the data can be referred to later.

    The repository is private because tourism.csv contains customer data (age, income,
    occupation, ...). Only people and pipelines with a Hugging Face token can read it.

What it does
    1. Repository - create the dataset repository if it does not exist yet, and make sure it
                    is private
    2. Upload     - upload tourism.csv from tourism_project/data/

How the file is organised
    Each step has a header with its input, output and reasoning, followed by one function.
    main() at the bottom connects the steps. Names and paths come from the central Config.

Running it
    python -m tourism_project.model_building.data_register   (from the project root)
    Runs the same way from the notebook (%run -m) and in GitHub Actions (python -m).
    Needs a Hugging Face token with write access, from HF_TOKEN or `hf auth login`.
    Running it again with an unchanged file creates no new commit on the Hub.
"""
from huggingface_hub import HfApi

from tourism_project.config import Config
from tourism_project.hub import ensure_private_repo


# =============================================================================
# Step 1 - Repository
# =============================================================================
#
# Input : the repository name from Config (richvrb/tourism-wellness-data)
# Output: a private dataset repository on the Hugging Face Hub, created if it is missing
#
# The first run creates the repository; every later run, for example each GitHub Actions run,
# finds it already exists and continues. Visibility is set explicitly on every run (see
# hub.ensure_private_repo), so the customer data stays private even if someone changed the
# setting by hand on the website.
#

def ensure_private_dataset_repo(api: HfApi, cfg: Config) -> None:
    """Create the dataset repository if needed, and make sure it is private."""
    # The shared helper is also used by model training for the model repository.
    ensure_private_repo(api, cfg.hf_dataset_repo, repo_type="dataset")


# =============================================================================
# Step 2 - Upload
# =============================================================================
#
# Input : tourism_project/data/tourism.csv (uploaded into Colab, or committed in the repository)
# Output: the same file in the dataset repository, plus the id of the commit
#
# After this step the Hub holds the registered raw data. Data preparation reads it from there,
# in the notebook as well as in GitHub Actions, so later stages never depend on a local file.
#

def upload_raw_data(api: HfApi, cfg: Config) -> str:
    """Upload tourism.csv and return the id of the resulting commit."""
    # A clear message is more helpful than the Hub's error when the file was not uploaded
    # to the data folder (in Colab this is a manual step).
    if not cfg.data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at {cfg.data_path}")

    # Upload just this one file, not the whole data folder: only the raw CSV belongs in
    # the registration. If the file is unchanged, the Hub skips the commit.
    commit = api.upload_file(
        path_or_fileobj=cfg.data_path,
        path_in_repo=cfg.raw_data_file,
        repo_id=cfg.hf_dataset_repo,
        repo_type="dataset",
        commit_message=f"Register raw dataset {cfg.raw_data_file}",
    )
    return commit.oid


# =============================================================================
# Run all steps
# =============================================================================
#
# main() connects the steps and prints where the data ended up. It prints repository
# details only, never the contents of the file.
#

def main() -> None:
    """Run steps 1 and 2, then print where the data lives."""
    cfg = Config()
    api = HfApi()  # picks up the token from HF_TOKEN or `hf auth login`

    # 1. Repository
    ensure_private_dataset_repo(api, cfg)

    # 2. Upload (remember the latest commit first, to report whether anything changed)
    commit_before = api.repo_info(cfg.hf_dataset_repo, repo_type="dataset").sha
    commit_after = upload_raw_data(api, cfg)

    # Report
    info = api.repo_info(cfg.hf_dataset_repo, repo_type="dataset")
    files = sorted(sibling.rfilename for sibling in info.siblings)
    unchanged_note = "  (unchanged, no new commit)" if commit_after == commit_before else ""

    print(f"Dataset repo : https://huggingface.co/datasets/{cfg.hf_dataset_repo}")
    print(f"Private      : {info.private}")
    print(f"Files        : {', '.join(files)}")
    print(f"Commit       : {commit_after}{unchanged_note}")


if __name__ == "__main__":
    main()
