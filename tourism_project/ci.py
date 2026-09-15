"""Helpers for running pipeline stages inside GitHub Actions.

Context
    In GitHub Actions every stage of the pipeline (data preparation, model training, ...) runs
    as a separate job on its own fresh machine. Jobs share nothing by default, so a stage that
    produces something the next stage needs, such as the id of the dataset version it uploaded,
    has to hand it over explicitly. GitHub also shows a summary page for every workflow run,
    which is the easiest place to read a stage's results.

    Both helpers do nothing when the code runs elsewhere (Colab, a local terminal), so the same
    scripts work unchanged in the notebook and in the pipeline.

What this module offers
    - export_github_output : pass a value (e.g. data_revision) on to later jobs
    - write_step_summary   : add markdown (e.g. a results table) to the workflow run's summary page
    - markdown_table       : format a DataFrame as a markdown table for that summary page
"""
import os

import pandas as pd


def export_github_output(name: str, value: str) -> None:
    """Pass a value to later jobs when running inside GitHub Actions; do nothing elsewhere."""
    # GitHub provides a file via GITHUB_OUTPUT; every "name=value" line written to it becomes an
    # output of the current step that later jobs can read. Outside GitHub Actions it is not set.
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def write_step_summary(markdown: str) -> None:
    """Append markdown to the GitHub Actions run summary; do nothing elsewhere."""
    # GITHUB_STEP_SUMMARY points to a markdown file that GitHub renders on the run's summary page.
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip("\n") + "\n\n")


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Format a DataFrame (index included) as a markdown table; no extra packages needed."""
    # pandas' own to_markdown() needs the optional 'tabulate' package, which GitHub Actions
    # would have to install just for this; a plain loop keeps the pipeline's dependencies small.
    table = frame.reset_index()

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else float_format.format(value)
        return str(value)

    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    divider = "|" + "---|" * len(table.columns)
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *rows])
