# Nexa Web-UI Test Guide

## Overview

This project integrates Nexa SDK with Web-UI to enable local multimodal LLM-driven browser automation. Currently tested with the `NexaAI/Qwen3-VL-4B-Instruct-GGUF` model (recommended).

## Environment Setup

### 1. Install Nexa SDK

Download and install the package from the official GitHub repository:

Visit https://github.com/NexaAI/nexa-sdk/releases/tag/v0.2.49 to download the appropriate installer for your platform and install it.

### 2. Install Web-UI Dependencies

This project builds upon the foundation of [@browser-use/web-ui](https://github.com/browser-use/web-ui), which is designed to make websites accessible for AI agents.

Choose one of the following methods to set up your Python environment:

#### Option A: Using uv (Recommended)

```bash
# Navigate to web-ui directory
cd web-ui

# Create virtual environment with uv
uv venv --python 3.11

# Activate virtual environment  
source .venv/bin/activate  # macOS/Linux
# or .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install Python dependencies
uv pip install -r requirements.txt

# (Optional) Install memory features for enhanced agent learning capabilities
# This adds ~110MB of ML dependencies (torch, transformers, etc.)
uv pip install "browser-use[memory]"

# Install Playwright browsers (recommend Chromium only)
playwright install chromium --with-deps
```

#### Option B: Using conda

```bash
# Navigate to web-ui directory
cd web-ui

# Create conda environment
conda create -n nexa-webui python=3.11 -y
conda activate nexa-webui

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install memory features for enhanced agent learning capabilities
# This adds ~110MB of ML dependencies (torch, transformers, etc.)
pip install "browser-use[memory]"

# Install Playwright browsers (recommend Chromium only)
playwright install chromium --with-deps
```

### 3. Configure Environment Variables

The project includes a preconfigured `web-ui/.env` file with the following main settings:

```bash
# LLM Provider Settings
DEFAULT_LLM=nexa
NEXA_ENDPOINT=http://127.0.0.1:8080/v1

# Other API Keys (if using other LLMs)
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
```

### 4. Download Model

Set up Hugging Face token and download the model (this is a private model requiring HF Token):

```bash
# Set Hugging Face token (both environment variables required)
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token"
export NEXA_HFTOKEN="your_huggingface_token"

# Download multimodal VLM model (recommended - GGUF format with better context support)
nexa pull NexaAI/Qwen3-VL-4B-Instruct-GGUF

```

**Note**:
- This is a private model requiring a valid Hugging Face Token
- Model is approximately 4GB, ensure sufficient storage and bandwidth
- Ensure your HF Token has access permissions for this model

## Test Preparation

### 0. Clean Up Ports

Before starting tests, ensure ports are clean:

```bash
# Kill all related processes
lsof -ti:8080,7788 | xargs kill -9 2>/dev/null
pkill -f "nexa serve"
pkill -f "webui.py"
```

### 1. Start Nexa Server

```bash
# Navigate to project root directory
cd Nexa-Web-UI
nexa serve --host 127.0.0.1:8080 --keepalive 600
```

Wait until you see the `Localhosting on http://127.0.0.1:8080/docs/ui` message.

### 2. Start Web-UI

In a new terminal window:

```bash
# Navigate to project root directory
cd Nexa-Web-UI
source web-ui/.venv/bin/activate  # or conda activate nexa-webui
python web-ui/webui.py --ip 127.0.0.1 --port 7788
```

Wait until you see the `Running on local URL: http://127.0.0.1:7788` message.

## Test Steps

Visit http://127.0.0.1:7788 for testing:

### Example Test Task
Input task: `Go to google.com, search for 'nexa ai', and click the first element`

**Expected Behavior:**
1. Navigate to google.com
2. Enter 'nexa ai' in the search box and search
3. Click the first element in the search results

The agent will automatically execute these steps and report completion status.

## Acknowledgments

We would like to officially thank the [browser-use/web-ui](https://github.com/browser-use/web-ui) project and its contributors for providing the foundation that makes this integration possible.

---

*Last updated: October 16, 2024*