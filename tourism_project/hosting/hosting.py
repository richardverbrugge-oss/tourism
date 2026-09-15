"""Hosting stage of the Visit with Us MLOps pipeline: publish the Streamlit app on Hugging Face.

Context
    The pipeline ends with an app that marketing can use: enter a customer profile, see whether that
    customer is worth contacting first. The app runs in a public Hugging Face Space, which builds a
    Docker image from the files in tourism_project/deployment/ and starts it.

        data_register.py  ->  prep.py  ->  train.py         ->  hosting.py
                                           model on the Hub      app in a public Space

    This script pushes the deployment files to that Space and tells the app which model version to
    load. It runs from the notebook (%run -m) and later as the last job in GitHub Actions (python -m).

What it does
    1. Space          - make sure the Docker Space exists and is public
    2. Model version  - pick the model commit the app must load (MODEL_REVISION)
    3. Space settings - store that commit as a Space variable; check the read-only token secret exists
    4. Upload         - upload only the whitelisted deployment files, in one commit
    5. Wait           - follow the build until the app runs, then check the public address answers

Tokens
    Uploading needs a token with write access (HF_TOKEN or `hf auth login`). The app itself uses a
    different token: the Space secret HF_TOKEN, which may only READ the model repository. This script
    checks that the secret exists but never sets it: the token available here (e.g. in GitHub Actions)
    has write access, and a public app must not hold a write token.

How the file is organised
    Settings come from the central Config (section "Deployment and hosting"). Each step has a header
    with its input, output and reasoning; main() connects the steps.

Running it
    python -m tourism_project.hosting.hosting   (from the project root)
    Building the image and starting the app takes a few minutes; the script waits and reports.
"""
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

from tourism_project.config import Config
from tourism_project.hub import resolve_model_revision

# Space states that mean the build or the app failed; waiting longer will not help.
FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING", "STOPPED", "PAUSED"}
# Space states that show a new build or restart is in progress.
BUSY_STAGES = {"BUILDING", "APP_STARTING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"}
# If no build or restart becomes visible within this time, the current state is taken as final.
RESTART_GRACE_S = 180


# =============================================================================
# Step 1 - Space
# =============================================================================
#
# Input : the Space name from Config
# Output: a public Docker Space (created when missing)
#
# Creating a Docker Space requires a Hugging Face PRO account. Visibility is set on every run, so the
# app stays reachable even if the setting was changed by hand on the website.
#

def ensure_public_space(api: HfApi, cfg: Config) -> None:
    """Create the Docker Space if it does not exist, and make sure it is public."""
    api.create_repo(cfg.hf_space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.update_repo_settings(cfg.hf_space_repo, repo_type="space", visibility="public")


# =============================================================================
# Step 2 - Model version
# =============================================================================
#
# Input : the model repository, and $MODEL_REVISION when the training job passed one on
# Output: the commit id of the model the app has to load
#
# In GitHub Actions the training job passes on the commit it just registered, so the app serves exactly
# that model. From the notebook the newest commit of the model repository is used.
#

# resolve_model_revision lives in hub.py, next to the matching helper for the dataset revision.


# =============================================================================
# Step 3 - Space settings
# =============================================================================
#
# Input : the model commit
# Output: the Space variables MODEL_REVISION and MODEL_REPO set; the secret HF_TOKEN confirmed to exist
#
# Variables are visible settings (the app reads them as environment variables); secrets are hidden
# ones. Changing a variable makes the Space restart, so the app picks up the new model version.
#

def configure_space(api: HfApi, cfg: Config, model_revision: str) -> bool:
    """Set the model variables and check the token secret exists; return True if a variable changed."""
    wanted = {"MODEL_REPO": cfg.hf_model_repo, "MODEL_REVISION": model_revision}
    current = {key: variable.value for key, variable in api.get_space_variables(cfg.hf_space_repo).items()}

    changed = False
    for key, value in wanted.items():
        if current.get(key) != value:
            api.add_space_variable(cfg.hf_space_repo, key, value)
            changed = True

    # Only the NAME of the secret can be read back; its value stays hidden, which is the point.
    secret_names = set(api.get_space_secrets(cfg.hf_space_repo))
    if "HF_TOKEN" not in secret_names:
        raise RuntimeError(
            "The Space has no secret HF_TOKEN. Add a fine-grained token with READ access to "
            f"{cfg.hf_model_repo} under Settings -> Variables and secrets of {cfg.hf_space_repo}."
        )
    return changed


# =============================================================================
# Step 4 - Upload
# =============================================================================
#
# Input : the files listed in Config.deployment_files, from tourism_project/deployment/
# Output: those files in the Space repository, in one commit (which starts a new build)
#
# Only whitelisted files are uploaded, never the whole project: the project also holds customer data
# and an experiment database, and the Space is public. If no file changed, the Hub skips the commit.
#

def upload_deployment_files(api: HfApi, cfg: Config) -> tuple[str, bool]:
    """Upload the whitelisted deployment files; return the Space commit and whether it is new."""
    missing = [name for name in cfg.deployment_files if not (cfg.deployment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Deployment files missing in {cfg.deployment_dir}: {missing}")

    commit_before = api.space_info(cfg.hf_space_repo).sha
    disable_progress_bars()  # keep notebook output and workflow logs readable
    commit = api.upload_folder(
        repo_id=cfg.hf_space_repo,
        repo_type="space",
        folder_path=cfg.deployment_dir,
        allow_patterns=list(cfg.deployment_files),  # whitelist: nothing else can be published
        commit_message="Deploy Streamlit app",
    )
    return commit.oid, commit.oid != commit_before


# =============================================================================
# Step 5 - Wait for the app
# =============================================================================
#
# Input : whether anything changed in steps 3 and 4
# Output: the app running, and its public address answering the health check
#
# After a change the Space first rebuilds or restarts. Until that starts, the status still shows the
# OLD state: the old app running, or "no app file" before the very first deployment. The script therefore
# only trusts "running" or an error status after it has seen the Space building or starting. If no build
# becomes visible within a few minutes (for example because nothing needed rebuilding), the state at that
# moment is taken as final. On a build or runtime error the last lines of the logs are printed.
#

def print_log_tail(api: HfApi, cfg: Config, lines: int = 30) -> None:
    """Print the last lines of the build log and the container log, to show why the Space failed."""
    for build in (True, False):
        try:
            log = list(api.fetch_space_logs(cfg.hf_space_repo, build=build))
        except Exception as error:  # a missing log must not hide the original failure
            print(f"  (could not read {'build' if build else 'container'} log: {error})")
            continue
        print(f"--- last {lines} lines of the {'build' if build else 'container'} log ---")
        print("\n".join(line.rstrip() for line in log[-lines:]))


def wait_until_running(api: HfApi, cfg: Config, expect_restart: bool) -> None:
    """Follow the Space status until the (new) app runs; raise on failure or timeout."""
    start = time.monotonic()
    deadline = start + cfg.space_build_timeout_s
    seen_activity = not expect_restart  # nothing changed -> the current state is already the final one
    last_stage = None
    while time.monotonic() < deadline:
        stage = api.get_space_runtime(cfg.hf_space_repo).stage
        if stage != last_stage:
            print(f"  Space status: {stage}", flush=True)
            last_stage = stage

        if stage in BUSY_STAGES:
            seen_activity = True
        # Trust the status once a build/restart was seen, or when none appeared within the grace time.
        trusted = seen_activity or time.monotonic() - start > RESTART_GRACE_S
        if trusted and stage == "RUNNING":
            return
        if trusted and stage in FAILED_STAGES:
            print_log_tail(api, cfg)
            raise RuntimeError(f"The Space stopped with status {stage}; see the log above.")
        time.sleep(cfg.space_poll_interval_s)

    print_log_tail(api, cfg)
    raise TimeoutError(f"The Space was not running after {cfg.space_build_timeout_s} seconds.")


def check_public_health(cfg: Config, attempts: int = 10) -> None:
    """Confirm the public address answers, without any login, the way a visitor would reach it."""
    url = f"{cfg.space_url}/_stcore/health"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                if response.status == 200:
                    print(f"  Health check: {url} -> 200 OK")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass  # the proxy may need a few more seconds after the status turns RUNNING
        time.sleep(cfg.space_poll_interval_s)
    raise RuntimeError(f"The app is running but {url} did not answer after {attempts} attempts.")


# =============================================================================
# Run all steps
# =============================================================================

def main() -> None:
    """Run steps 1 to 5 in order and print where the app can be reached."""
    cfg = Config()
    api = HfApi()  # write access, from HF_TOKEN or `hf auth login`

    # 1. Space
    ensure_public_space(api, cfg)

    # 2. Model version
    model_revision = resolve_model_revision(api, cfg)
    print(f"Space          : https://huggingface.co/spaces/{cfg.hf_space_repo}")
    print(f"Model version  : {cfg.hf_model_repo} @ {model_revision[:8]}")

    # 3. Space settings
    variables_changed = configure_space(api, cfg, model_revision)

    # 4. Upload
    space_commit, files_changed = upload_deployment_files(api, cfg)
    print(f"Space commit   : {space_commit[:8]}{'' if files_changed else '  (files unchanged, no new commit)'}")
    print(f"Uploaded files : {', '.join(cfg.deployment_files)}")

    # 5. Wait for the app
    print("Waiting for the Space to build and start ...", flush=True)
    wait_until_running(api, cfg, expect_restart=variables_changed or files_changed)
    check_public_health(cfg)
    print(f"\nApp is live    : {cfg.space_url}")


if __name__ == "__main__":
    main()
