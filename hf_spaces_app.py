"""
hf_spaces_app.py  —  Entry point for Hugging Face Spaces deployment.

HF Spaces runs this file directly. It starts the FastAPI backend
in a background thread and the Streamlit UI in the foreground.

To deploy:
  1. Create a new Space at https://huggingface.co/spaces
  2. Set SDK to "Docker" or "Gradio"
  3. Upload this repo
  4. Space will auto-build and deploy

OR use the CLI:
  huggingface-cli repo create smartrag --type space --space_sdk docker
  git remote add space https://huggingface.co/spaces/YOUR_USERNAME/smartrag
  git push space main
"""

import subprocess
import threading
import time
import os

API_PORT = 8000
UI_PORT  = 7860   # HF Spaces default port


def start_api():
    """Start FastAPI in background."""
    subprocess.run([
        "uvicorn", "api.app:app",
        "--host", "0.0.0.0",
        "--port", str(API_PORT),
    ])


def start_ui():
    """Wait for API, then start Streamlit UI."""
    time.sleep(15)   # Give API time to load the model
    env = os.environ.copy()
    env["API_BASE"] = f"http://localhost:{API_PORT}"
    subprocess.run([
        "streamlit", "run", "ui/app.py",
        "--server.port", str(UI_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ], env=env)


if __name__ == "__main__":
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    start_ui()
