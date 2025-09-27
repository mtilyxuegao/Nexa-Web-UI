# Nexa-Web-UI

Integration of Nexa SDK and Browser-Use Web UI for AI-powered browser automation.

## Structure

```
Nexa-Web-UI/
├── nexa-sdk/    # Nexa AI SDK submodule
└── web-ui/      # Browser-Use Web UI submodule
```

## Quick Start

### Clone with submodules

```bash
git clone --recursive https://github.com/your-username/Nexa-Web-UI.git
cd Nexa-Web-UI
```

### Initialize submodules (if needed)

```bash
git submodule update --init --recursive
```

### Use Nexa SDK

```bash
cd nexa-sdk
# Follow nexa-sdk installation guide
```

### Use Web UI

```bash
cd web-ui
pip install -r requirements.txt
playwright install --with-deps
cp .env.example .env
# Edit .env with your API keys
python webui.py --ip 127.0.0.1 --port 7788
```

## Submodules

- **Nexa SDK**: https://github.com/NexaAI/nexa-sdk.git
- **Web UI**: https://github.com/browser-use/web-ui.git

## License

- Nexa SDK: Apache-2.0
- Web UI: MIT