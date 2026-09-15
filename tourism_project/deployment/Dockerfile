# Container image for the Wellness Tourism Streamlit app, built by the Hugging Face Space.
#
# The trained model is NOT part of this image. At start-up app.py downloads it from the private
# Hugging Face model repository at the exact commit set in the Space variable MODEL_REVISION.
# A new model therefore needs a new revision on the Hub and an updated variable, not a new image.

# Python version and package versions must match the training environment (model_metadata.json):
# a saved scikit-learn model only loads reliably with the versions it was saved with.
# "slim" is enough: the scikit-learn and XGBoost wheels bring their own compiled libraries.
FROM python:3.12-slim

# No .pyc files, unbuffered logs (visible immediately in the Space logs), smaller image, quieter pip.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies first: Docker caches this layer and only reinstalls when requirements.txt changes,
# not on every change to app.py. They are installed as root, so the app user cannot alter them.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Hugging Face Spaces run containers as user id 1000; running as a normal user instead of root
# limits what a compromised app could do.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Only the two files the app needs. Together with the whitelist in .dockerignore this guarantees that
# nothing else from the project (customer data, experiment database, notebooks) ends up in the image.
COPY --chown=user app.py .
COPY --chown=user .streamlit/config.toml .streamlit/config.toml

# Streamlit listens on 8501; the Space README sets app_port: 8501 so Hugging Face routes traffic here.
EXPOSE 8501

# Reports whether the app answers. Hugging Face uses its own checks; this one is for local and CI runs.
# The slim image has no curl, so the probe uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)"

# Start the app on all network interfaces (required inside a container). Security settings such as
# XSRF protection and hidden error details are in .streamlit/config.toml.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
