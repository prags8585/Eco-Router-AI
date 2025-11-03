# AI Router (Frontend + Backend)

## Quick start

```bash
cd ai_router_full
python3 -m venv .venv
source .venv/bin/activate            # on Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx_your_token    # on Windows PowerShell: $env:HF_TOKEN="hf_xxx"
python -m uvicorn main:app --reload --port 8080
```

In another terminal:

```bash
cd ai_router_full/Frontend
python -m http.server 5500
```

Open http://localhost:5500/ and click **Run**.
