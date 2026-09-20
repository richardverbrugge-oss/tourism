"""Data preparation stage of the Visit with Us MLOps pipeline.

Context
    Visit with Us wants to know, before the sales team contacts a customer, whether that
    customer is likely to buy the new Wellness Tourism Package (column ProdTaken, 1 = bought).
    Marketing can then spend its calls on the most promising customers. The pipeline that
    delivers this prediction has four stages, each building on the output of the previous one:

        data_register.py  ->  prep.py            ->  train.py        ->  deployment
        raw data on Hub       train/test on Hub      model on Hub        Streamlit app

    This script is the second stage. It turns the registered raw data into the train and test
    sets that model training learns from and is evaluated on. The same code runs in two places:
    from the notebook (%run -m) and as a job in GitHub Actions (python -m). Both read from and
    write to the Hugging Face Hub, so both produce identical files.

What it does
    1. Load  - read tourism.csv (4,128 customers, 20 columns) from the private dataset
               repository on the Hugging Face Hub
    2. Clean - merge a spelling variant and mark impossible values as missing
    3. Group - detect rows that are near-copies of each other and give them one group id
    4. Split - make a stratified 80/20 train/test split that never separates copies
    5. Save  - write train.csv and test.csv and upload both to the same repository

Why step 3 matters
    About three quarters of the rows have a near-copy elsewhere in the data: the same customer
    profile with one or two values changed, most likely created when this teaching dataset was
    enlarged. With an ordinary random split, 478 of the 826 test rows had a near-copy in the
    training set. A model evaluated on that test set would partly be recognising customers it
    has already seen, so its score would overstate how well it predicts new customers. Keeping
    each group of copies on one side of the split brings that number to 0.

How the file is organised
    Settings come from the central Config (config.py, section "Data preparation") and are
    passed to every function, so this file contains no hardcoded values. Each step has a header
    with its input, output and reasoning, followed by small functions that each do one task.
    main() at the bottom is the only place where the steps are connected.

Running it
    python -m tourism_project.model_building.prep   (from the project root)
    Needs a Hugging Face token with write access, from HF_TOKEN or `hf auth login`.
    Prints counts only, never customer rows, because notebook output and workflow logs are
    public. Running it twice gives identical files, and the Hub then creates no new commit.
"""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from tourism_project.ci import export_github_output
from tourism_project.config import Config


# =============================================================================
# Step 1 - Load
# =============================================================================
#
# Input : the private dataset repository on the Hugging Face Hub (filled by data_register.py)
# Output: a DataFrame with the 4,128 raw customer rows and all 20 columns
#
# The raw file is read from the Hub, not from tourism_project/data/. The Hub copy is the
# registered version of the data: GitHub Actions has no Colab upload folder, and reading the
# same registered file everywhere guarantees that the notebook and the pipeline start from
# identical data.
#
# The Hugging Face token (needed because the repository is private) is picked up automatically
# from the HF_TOKEN environment variable or from a local `hf auth login`.
#

def load_raw(cfg: Config) -> pd.DataFrame:
    """Download tourism.csv from the private dataset repository and read it into a DataFrame."""
    local_path = hf_hub_download(cfg.hf_dataset_repo, cfg.raw_data_file, repo_type="dataset")

    # index_col=0: the CSV starts with an unnamed row-number column that carries no information.
    return pd.read_csv(local_path, index_col=0)


# =============================================================================
# Step 2 - Clean
# =============================================================================
#
# Input : the raw DataFrame
# Output: a cleaned copy with the same 4,128 rows and 20 columns
#
# An inspection of the raw data found three problems. Each has a fixed rule in Config:
#   1. Gender has three values: "Male", "Female" and "Fe Male" (155 rows). "Fe Male" is a
#      spelling variant of "Female" and is merged into it.
#   2. MonthlyIncome: all incomes lie between 16,009 and 38,304, except three values
#      (1,000, 4,678 and 98,678). These are entry errors and are set to NaN.
#   3. NumberOfTrips: all values are 1-8, except four values of 19-22. Also entry errors,
#      also set to NaN.
#
# Why NaN instead of deleting the row: the customer's other 19 values are still useful.
# The model pipeline (training stage) fills missing values in with an imputer.
#
# Why fixed limits instead of limits computed from the data, such as percentiles: limits
# computed from all rows would be influenced by the test rows. Nothing about the test set may
# influence how the training data is prepared, otherwise the test score is no longer an honest
# estimate for new customers.
#

def clean_values(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the cleaning rules from Config and return a cleaned copy; the input is unchanged."""
    cleaned = df.copy()

    # Merge spelling variants into one category, e.g. "Fe Male" -> "Female".
    cleaned["Gender"] = cleaned["Gender"].replace(dict(cfg.gender_replacements))

    # Incomes outside the realistic range are entry errors -> NaN.
    low_income, high_income = cfg.monthly_income_range
    income_out_of_range = ~cleaned["MonthlyIncome"].between(low_income, high_income)
    cleaned.loc[income_out_of_range, "MonthlyIncome"] = np.nan

    # An unrealistically high number of trips per year is an entry error -> NaN.
    too_many_trips = cleaned["NumberOfTrips"] > cfg.max_number_of_trips
    cleaned.loc[too_many_trips, "NumberOfTrips"] = np.nan

    return cleaned


# =============================================================================
# Step 3 - Group copies
# =============================================================================
#
# Input : the cleaned DataFrame
# Output: the same DataFrame with an extra column group_id
#
# What was found: 194 pairs of rows are identical on all 14 features, and many more rows match
# on all but one or two features. These are near-copies of the same customer profile, not
# different customers who happen to be similar. Two observations support that:
#   - Scoring all 8.5 million row pairs (see below) gives two clearly separated clusters:
#     millions of pairs with ordinary similarity, about 1,500 pairs close to the maximum score,
#     and almost nothing in between (73 pairs between 23 and 27 bits).
#   - The method below never looks at CustomerID, yet 99% of the pairs it links turn out to be
#     exactly the same distance apart in customer number: the trace of a program that copied
#     records and changed a value or two.
#
# How copies are detected: compare every pair of rows and give the pair an "evidence score"
# that says how unlikely it is that two different customers agree this much by chance.
# Not every agreement is equally convincing:
#   - Two random customers have the same Passport value (yes/no) about half of the time.
#     That agreement proves almost nothing.
#   - Two random customers have exactly the same MonthlyIncome about once in 2,000 pairs.
#     That agreement proves a lot.
# So each feature gets a weight in bits: weight = -log2(chance that two random rows agree).
#   chance 0.5     -> 1 bit       (Passport)
#   chance 0.0005  -> ~11 bits    (MonthlyIncome)
# The score of a pair is the sum of the weights of the features on which both rows agree
# (maximum about 31 bits). Pairs scoring at least Config.copy_threshold_bits (24, in the
# empty gap between the two clusters) are copies. This is the standard idea behind record
# linkage: matching records that describe the same entity.
#
# Rows linked by copies, directly or through a chain (A~B and B~C), form one group. Result:
# 2,574 groups of 1, 2 or 4 rows; about three quarters of all rows belong to a group of 2+.
#
# The target (ProdTaken) is not compared: a copy leaks information even if its outcome differs.
#

def encode_features(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Turn every feature value into an integer code, so rows can be compared quickly."""
    encoded_columns = []
    for column in cfg.feature_columns:
        # Convert to text first so that 3 and 3.0 get the same code, and give missing values
        # their own code so that two NaN values count as agreeing.
        as_text = df[column].astype("string").fillna("<NA>")
        codes, _unique_values = pd.factorize(as_text)
        encoded_columns.append(codes)

    # Result: one row per customer, one column per feature, small integers only.
    return np.column_stack(encoded_columns).astype(np.int32)


def agreement_weights(codes: np.ndarray) -> np.ndarray:
    """Weight per feature in bits: -log2 of the chance that two random rows agree on it."""
    weights = []
    for column_codes in codes.T:
        # Share of rows per value, e.g. Passport: 0.7 "no" and 0.3 "yes".
        value_shares = np.bincount(column_codes) / len(column_codes)

        # Chance that two randomly drawn rows have the same value = sum of the squared shares,
        # e.g. 0.7^2 + 0.3^2 = 0.58. Rare values push this chance down and the weight up.
        chance_of_agreement = np.sum(value_shares**2)

        weights.append(-np.log2(chance_of_agreement))
    return np.array(weights)


def iter_pair_scores(
    codes: np.ndarray, weights: np.ndarray, chunk_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (i, j, score) arrays for every row pair with i < j, one row i at a time."""
    n_rows = len(codes)

    # 4,128 rows give about 8.5 million pairs. A full rows x rows x features comparison would
    # need gigabytes of memory, so `chunk_size` rows are compared with all rows at a time.
    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        chunk_codes = codes[chunk_start:chunk_end]

        # scores[a, b] = evidence that row a of this chunk and row b of the dataset are copies.
        scores = np.zeros((chunk_end - chunk_start, n_rows))
        for column_index, column_weight in enumerate(weights):
            # True where both rows have the same value for this feature; add its weight there.
            agrees = chunk_codes[:, [column_index]] == codes[:, column_index]
            scores += agrees * column_weight

        # Report every pair once and never compare a row with itself: only pairs with i < j.
        for position_in_chunk in range(chunk_end - chunk_start):
            i = chunk_start + position_in_chunk
            later_rows = np.arange(i + 1, n_rows)
            if len(later_rows) == 0:
                continue
            yield np.full(len(later_rows), i), later_rows, scores[position_in_chunk, i + 1:]


def find_copy_pairs(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Return all row pairs whose evidence score reaches the copy threshold, as [i, j] rows."""
    codes = encode_features(df, cfg)
    weights = agreement_weights(codes)

    copy_pairs = []
    for rows_i, rows_j, scores in iter_pair_scores(codes, weights, cfg.comparison_chunk_size):
        is_copy = scores >= cfg.copy_threshold_bits
        if is_copy.any():
            copy_pairs.append(np.column_stack([rows_i[is_copy], rows_j[is_copy]]))

    if not copy_pairs:
        return np.empty((0, 2), dtype=int)
    return np.vstack(copy_pairs)


def add_group_id(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add a group id column: rows that are copies, directly or through a chain, share it."""
    grouped = df.copy()
    copy_pairs = find_copy_pairs(grouped, cfg)
    n_rows = len(grouped)

    # Picture every row as a dot and every copy pair as a line between two dots. A group is a
    # set of dots connected by lines (a "connected component" in graph terms); a row without
    # any copy is a group on its own. scipy expects the lines as a sparse rows x rows matrix.
    line_matrix = coo_matrix(
        (np.ones(len(copy_pairs)), (copy_pairs[:, 0], copy_pairs[:, 1])),
        shape=(n_rows, n_rows),
    )
    _n_groups, group_per_row = connected_components(line_matrix, directed=False)

    grouped[cfg.group_column] = group_per_row
    return grouped


# =============================================================================
# Step 4 - Split
# =============================================================================
#
# Input : the grouped DataFrame (every row has a group_id)
# Output: a train set (about 80%, used to fit the model) and a test set (about 20%, used
#         only once at the end to estimate how well the model predicts new customers)
#
# The split has to meet two requirements at the same time:
#   - Groups stay together: all copies of a record go to train, or all go to test. This is
#     what keeps copies of training customers out of the test set (step 3).
#   - Stratified: train and test get about the same share of buyers. Only about 19% of the
#     customers bought a package, so a purely random test set of 826 rows could by chance
#     hold noticeably more or fewer buyers and give a misleading score.
# StratifiedGroupKFold from scikit-learn meets both requirements.
#
# check_split verifies the result and stops the script if a requirement is not met, so a
# broken split can never reach model training. select_columns then drops CustomerID and the
# post-contact columns, which the model must not see (see Config.post_contact_columns).
#

def split_by_group(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test so that each group lands entirely on one side."""
    # StratifiedGroupKFold divides the groups into k parts, each with about the same share of
    # buyers. With test_size 0.2 that is 5 parts; the first part becomes the test set and the
    # other four together form the train set.
    n_parts = round(1 / cfg.test_size)
    splitter = StratifiedGroupKFold(
        n_splits=n_parts, shuffle=True, random_state=cfg.random_state
    )
    parts = splitter.split(df, df[cfg.target], groups=df[cfg.group_column])
    train_positions, test_positions = next(parts)

    train = df.iloc[train_positions].copy()
    test = df.iloc[test_positions].copy()
    return train, test


def check_split(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
) -> None:
    """Raise an error if the split lost rows, leaked a customer or group, or is unbalanced."""
    prevalence_gap = abs(train[cfg.target].mean() - test[cfg.target].mean())
    test_share_gap = abs(len(test) / len(full) - cfg.test_size)

    # Each check has a readable name, so a failure message says exactly which requirement broke.
    checks = {
        "no rows lost": len(train) + len(test) == len(full),
        "no customer in both sets": not set(train[cfg.id_column]) & set(test[cfg.id_column]),
        "no group in both sets": not set(train[cfg.group_column]) & set(test[cfg.group_column]),
        "share of buyers similar in train and test": prevalence_gap <= cfg.max_prevalence_gap,
        "test set has the intended size": test_share_gap <= cfg.max_test_share_gap,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        # A real error instead of `assert`, because Python skips asserts when run with -O.
        raise ValueError(f"Split checks failed: {failed}")


def select_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep only the features, the target and the group id, in a fixed column order."""
    # Dropped: CustomerID (an identifier, not information about the customer) and the four
    # post-contact columns. group_id stays because training needs it to keep copies together
    # in cross-validation as well; it is never used as a model feature.
    return df[cfg.split_columns].reset_index(drop=True)


# =============================================================================
# Step 5 - Save and upload
# =============================================================================
#
# Input : the train and test DataFrames
# Output: train.csv and test.csv in outputs/data/ and in the dataset repository on the Hub
#
# The files go to the Hub because the next stage, model training, reads them from there, both
# in the notebook and in GitHub Actions. That way every stage works with exactly the same data.
#
# Every upload to a Hugging Face repository is a commit with a unique id, just like in Git.
# That id is the DATA_REVISION: a permanent label for this exact version of train and test.
# In GitHub Actions it is handed to the training job, so training uses the split this run
# produced even if the repository receives another commit in the meantime.
#

def save_splits(cfg: Config, train: pd.DataFrame, test: pd.DataFrame) -> list[Path]:
    """Save train.csv and test.csv in the work folder (outputs/data/) and return their paths."""
    output_dir = cfg.work_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / cfg.train_file
    test_path = output_dir / cfg.test_file
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return [train_path, test_path]


def upload_splits(api: HfApi, cfg: Config, paths: list[Path]) -> str:
    """Upload train.csv and test.csv in one commit and return that commit's id."""
    # One commit for both files, so every version of the repository holds a train set and a
    # test set that belong together. If both files are unchanged, the Hub skips the commit and
    # returns the id of the latest existing commit.
    files_to_upload = [
        CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path) for path in paths
    ]
    commit = api.create_commit(
        repo_id=cfg.hf_dataset_repo,
        repo_type="dataset",
        operations=files_to_upload,
        commit_message="Prepare cleaned train/test split",
    )
    return commit.oid


def plain_split_leakage(grouped: pd.DataFrame, cfg: Config) -> tuple[int, int]:
    """How many test rows an ordinary random split would leave with a near-copy in train."""
    # The comparison that justifies the group-aware split: split the same rows the ordinary way
    # (stratified on the target, ignoring the groups) and count the test rows whose group also
    # occurs in train. Those rows would be scored on a customer the model had already seen.
    train, test = train_test_split(
        grouped,
        test_size=cfg.test_size,
        stratify=grouped[cfg.target],
        random_state=cfg.random_state,
    )
    groups_in_train = set(train[cfg.group_column])
    leaking = int(test[cfg.group_column].isin(groups_in_train).sum())
    return leaking, len(test)


def print_data_quality(
    raw: pd.DataFrame, cleaned: pd.DataFrame, grouped: pd.DataFrame, cfg: Config
) -> None:
    """Print the counts behind the cleaning rules and the split, so no number is unexplained."""
    # Only counts and shares are printed: this output ends up in a public repository and in the
    # GitHub Actions log, where customer rows do not belong.
    print(f"Raw data       : {len(raw)} customers | buyers {raw[cfg.target].mean():.2%}")

    renamed = int((raw["Gender"] != cleaned["Gender"]).sum())
    incomes_removed = int(cleaned["MonthlyIncome"].isna().sum() - raw["MonthlyIncome"].isna().sum())
    trips_removed = int(cleaned["NumberOfTrips"].isna().sum() - raw["NumberOfTrips"].isna().sum())
    valid_income = cleaned["MonthlyIncome"].dropna()
    print(f"Cleaning       : {renamed} Gender spelling variants merged "
          f"| {incomes_removed} incomes and {trips_removed} trip counts outside the realistic range "
          f"-> missing")
    print(f"                 remaining incomes run from {valid_income.min():,.0f} to {valid_income.max():,.0f}")

    group_sizes = grouped[cfg.group_column].value_counts()
    rows_with_copy = int(group_sizes[group_sizes > 1].sum())
    print(f"Near-copies    : {rows_with_copy} of {len(grouped)} rows ({rows_with_copy / len(grouped):.0%}) "
          f"have at least one near-copy | {len(group_sizes)} groups")

    leaking, test_rows = plain_split_leakage(grouped, cfg)
    print(f"Why grouping   : an ordinary random split would leave {leaking} of {test_rows} test rows "
          f"with a near-copy in train; the group-aware split below leaves 0")


def print_report(
    cfg: Config, train: pd.DataFrame, test: pd.DataFrame, data_revision: str, unchanged: bool
) -> None:
    """Print a short summary with counts only, so no customer data appears in public output."""
    for name, split in (("Train", train), ("Test", test)):
        print(
            f"{name:<13} : {len(split)} rows "
            f"| buyers {split[cfg.target].mean():.2%} "
            f"| {split[cfg.group_column].nunique()} groups"
        )
    groups_in_both = set(train[cfg.group_column]) & set(test[cfg.group_column])
    print(f"Groups in both: {len(groups_in_both)}")
    unchanged_note = "  (unchanged, no new commit)" if unchanged else ""
    print(f"DATA_REVISION : {data_revision}{unchanged_note}")


# =============================================================================
# Run all steps
# =============================================================================
#
# main() is the only place where the steps are connected. Every function above receives what
# it needs as arguments and returns its result, so each step can also be run or inspected on
# its own, for example in a notebook cell.
#

def main() -> None:
    """Run steps 1 to 5 in order."""
    cfg = Config()
    api = HfApi()  # picks up the token from HF_TOKEN or `hf auth login`

    # 1. Load the registered raw data
    raw = load_raw(cfg)

    # 2. Clean
    cleaned = clean_values(raw, cfg)

    # 3. Group near-copies, then report the counts behind the cleaning rules and the grouping
    grouped = add_group_id(cleaned, cfg)
    print_data_quality(raw, cleaned, grouped, cfg)

    # 4. Split, check the split while CustomerID is still present, then drop extra columns
    train, test = split_by_group(grouped, cfg)
    check_split(grouped, train, test, cfg)
    train = select_columns(train, cfg)
    test = select_columns(test, cfg)

    # 5. Save locally, upload to the Hub, report
    paths = save_splits(cfg, train, test)
    commit_before = api.repo_info(cfg.hf_dataset_repo, repo_type="dataset").sha
    data_revision = upload_splits(api, cfg, paths)

    print_report(cfg, train, test, data_revision, unchanged=data_revision == commit_before)

    # In GitHub Actions: tell the training job which version of the data to use.
    export_github_output("data_revision", data_revision)


if __name__ == "__main__":
    main()
