"""Check, inside the built app container, that the app reproduces the trained model exactly.

Context
    The deploy job in GitHub Actions builds the app's Docker image and starts it before anything is
    uploaded to the public Hugging Face Space. This script then runs INSIDE that container:

        docker exec -i tourism-app python - < tourism_project/hosting/container_check.py

    Piping it in means the script itself never becomes part of the image. It uses the app's own functions,
    so it tests exactly what the Space will run: the model is downloaded at MODEL_REVISION with the token
    given to the container, the installed package versions are compared with those recorded at training
    time, and the synthetic check profiles are scored again.

    A container that starts and answers can still score differently from the trained model when package
    versions drift. This check catches that before the app goes public; the deploy job stops on a failure.

What it does
    1. Load the model bundle with app.load_bundle, as the app does at start-up
    2. Compare installed package versions with the model metadata (app.version_mismatches)
    3. Score the parity examples again and compare with the stored scores (app.parity_mismatches)
    4. Print the result and exit with code 1 if anything differs
"""
import sys
import warnings

# Streamlit warns that it is not running inside a Streamlit server; that is expected in this check.
warnings.filterwarnings("ignore")

import app  # the container's working directory holds app.py


def main() -> None:
    """Run the checks and exit with a non-zero code when the app would not reproduce the model."""
    settings = app.read_settings()
    if not settings.model_revision:
        print("MODEL_REVISION is not set in the container")
        sys.exit(1)

    # 1. Load the model exactly as the app does
    model, metadata, _operating_points = app.load_bundle(
        settings.model_repo, settings.model_revision, settings.token
    )
    print(f"Model          : {settings.model_repo} @ {settings.model_revision[:8]} ({metadata['selected_run']})")

    # 2-3. Versions and parity
    version_problems = app.version_mismatches(metadata)
    parity_problems = app.parity_mismatches(model, metadata)
    print(f"Versions       : {'OK' if not version_problems else version_problems}")
    print(f"Parity examples: {'OK' if not parity_problems else parity_problems} "
          f"({len(metadata['parity_examples'])} profiles)")

    # 4. Result
    if version_problems or parity_problems:
        print("Container check FAILED: the app would not reproduce the trained model's scores")
        sys.exit(1)
    print("Container check passed")


if __name__ == "__main__":
    main()
