# Nexa Web-UI Test Guide

## Overview

This project integrates Nexa SDK with Web-UI to enable local multimodal LLM-driven browser automation. Currently tested with the `NexaAI/Qwen3-VL-4B-MLX-8bit` model.

## Environment Setup

### 1. Install Nexa SDK

Download and install the package from the official GitHub repository:

Visit https://github.com/NexaAI/nexa-sdk/releases/tag/v0.2.35 to download the appropriate installer for your platform and install it.

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

# Download multimodal VLM model
nexa pull NexaAI/Qwen3-VL-4B-MLX-8bit
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
cd /Users/jason/Desktop/Nexa-Web-UI
nexa serve --host 127.0.0.1:8080 --keepalive 600
```

Wait until you see the `Localhosting on http://127.0.0.1:8080/docs/ui` message.

### 2. Start Web-UI

In a new terminal window:

```bash
cd /Users/jason/Desktop/Nexa-Web-UI
source .venv/bin/activate  # or conda activate nexa-webui
python web-ui/webui.py --ip 127.0.0.1 --port 7788
```

Wait until you see the `Running on local URL: http://127.0.0.1:7788` message.

## Test Steps

Visit http://127.0.0.1:7788 for testing:

### Complete Test Task (Three Steps)
Input task: `Go to google.com, search for 'nexa ai', and click the first result`

**Step 1: Navigate to Google (Should succeed)**
- Action: Navigate to google.com
- Expected result: ✅ Successfully opens Google homepage

**Step 2: Perform search (Should succeed)**  
- Action: Enter 'nexa ai' in search box and search
- Expected result: ✅ Successfully displays search results page

**Step 3: Click first result (Currently fails)**
- Action: Click the first search result
- Expected result: ❌ Fails, unable to execute click action
- Phenomenon: Agent will keep retrying, but `Action 1/1: {}` shows empty, continues attempting based on previous memory

## Current Issue Analysis

### Core Error

**Action passing failure**: Although the LLM correctly outputs JSON-formatted action (e.g., `{"click_element": {"index": 8}}`), the final action passed to browser-use is empty `{}`.

### Log Analysis Example

```
INFO [src.utils.nexa_adapter] 📝 Model raw output: {"current_state": {...}}, "action": [{"click_element": {"index": 8}}]}
INFO [src.utils.nexa_adapter] 🔧 Fixing structural error between current_state and action
INFO [src.utils.nexa_adapter] ✅ JSON fix successful
INFO [agent] 🛠️ Action 1/1: {}  ← Issue: action is empty
```

### Observed Performance Issues

Compared to other LLM providers like Ollama, using Nexa has the following issues:

1. **More steps required**: Same tasks need more steps to complete
2. **Higher retry frequency**: Retries are triggered when LLM output format doesn't fully meet requirements
3. **Context length sensitivity**: Issues become more apparent as conversation history grows

## Error Cause Analysis

### 1. LLM Capability Limitations
- **Model scale**: Quantized models have limitations in complex JSON structure generation
- **Instruction following ability**: Unstable adherence to strict JSON format requirements
- **Context processing**: Performance degradation with long conversation history

### 2. JSON Format Issues
- **Structural errors**: Model often outputs `{"current_state": {...}}, "action": [...]}` instead of correct single JSON object
- **Imperfect fixes**: Current regex-based fix solution still has edge cases

### 3. Multimodal Processing Challenges
- **Image understanding**: May not accurately understand blank pages or complex pages
- **Vision-language alignment**: Issues with alignment between image content and JSON output

## Possible Solutions

### Solution 1: Model Upgrade
**Pros**: Fundamental solution to capability issues
```
- Use larger parameter models (e.g., 7B/13B)
- Choose specially optimized instruction-following models
- Consider models supporting longer context
```

**Cons**: Increased resource consumption, slower inference speed

### Solution 2: Two-Stage Processing
**Pros**: Keep current model, add format correction step
```
Stage 1: VLM model understands image and generates preliminary JSON
Stage 2: Lightweight LLM (without vision) corrects JSON format in pure text
```

**Cons**: Increased latency and complexity

### Solution 3: Enhanced Regex Fixes
**Pros**: Low latency, currently used approach
```
- Improve regex expressions to cover more edge cases
- Add multi-layered format checking and fixing
- Implement smarter JSON structure reconstruction
```

**Cons**: Treats symptoms not cause, difficult to cover all cases

### Solution 4: Prompt Engineering Optimization
**Pros**: No architectural changes needed
```
- Provide more JSON format examples
- Use few-shot prompting
- Add self-checking mechanism for format validation
```

**Cons**: Limited effectiveness, depends on model's instruction following capability

## Current Implementation Status

### Implemented Features
- ✅ Nexa SDK and LangChain integration
- ✅ Multimodal image processing (base64 to file path conversion)
- ✅ Markdown code block parsing
- ✅ Basic JSON format fixes
- ✅ Debug logging and image saving

### Known Limitations
- ❌ Unstable action passing
- ❌ Low success rate for complex tasks
- ❌ Obvious performance gap compared to other LLM providers

### Next Steps
1. Deep debug browser-use JSON parsing logic
2. Implement Solution 2's two-stage processing architecture
3. Evaluate feasibility of upgrading to larger-scale models

## Debugging Tips

### View Detailed Logs
```bash
# View Nexa server logs
tail -f nexa_server.log

# View Web-UI logs (console output)
# Focus on [src.utils.nexa_adapter] output
```

### Check Saved Images
```bash
ls -la /Users/jason/Desktop/Nexa-Web-UI/debug_images/
# View actual image content received by LLM
```

### Verify JSON Fixes
Look for fix logs in `nexa_adapter.py`:
- `🔧 Fixing structural error between current_state and action`
- `✅ JSON fix successful`
- `❌ JSON fix failed`

## Acknowledgments

We would like to officially thank the [browser-use/web-ui](https://github.com/browser-use/web-ui) project and its contributors for providing the foundation that makes this integration possible.

---

*Last updated: September 28, 2025*