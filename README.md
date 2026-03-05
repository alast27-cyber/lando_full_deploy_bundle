# Tiny-ando API

Flask API with endpoints:
- `GET /healthz`
- `POST /chat`
- `POST /agent/chat` (agent-mode metadata v2)

## Local run
```bash
pip install -r requirements-dev.txt
python app/main.py
```

## Test
```bash
pytest -q
```

## Vercel deployment
This repo is configured for Vercel Python runtime using `api/index.py` and `vercel.json`.

Notes:
- Python is pinned to 3.12 for Vercel compatibility (`.python-version`, `runtime.txt`).
- The `/chat` endpoint lazy-loads model dependencies. If transformers/model packages are unavailable, the API now returns a lightweight fallback response so deployments still serve traffic.


## Vercel troubleshooting
If Vercel logs show `npm install` / `vite build` or TypeScript errors (for example files like `*.tsx`), then Vercel is building the wrong project.

For this repo, expected Vercel behavior is Python build using `vercel.json` and `api/index.py`.

Checklist:
- Vercel project must point to the correct Git repository and branch.
- Root Directory should be repo root (`/`).
- Framework Preset should be **Other** (or no framework).
- Build Command should be empty (let `@vercel/python` handle build).
- Output Directory should be empty.
- Ensure `vercel.json` routes all paths to `api/index.py` via `@vercel/python`.


## GitHub push quickstart
```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin <branch-name>
```

## Vercel deploy quickstart
1. Import this repository into Vercel.
2. Set **Framework Preset** to `Other`.
3. Leave **Build Command** and **Output Directory** empty.
4. Confirm `vercel.json` is detected and deploy.


## 404: NOT_FOUND quick fix
If deployment succeeds but returns `404: NOT_FOUND`, verify:
1. The Vercel project is connected to the same repo/branch you just pushed.
2. `vercel.json` is at the repository root.
3. Root Directory is `/` in Vercel project settings.
4. Re-deploy after clearing any stale project overrides.


## Agent evolution (v2)
- New `POST /agent/chat` route returns both `reply` and `agent` metadata.
- Agent metadata currently includes:
  - `mode`: lightweight intent routing (`chat`, `planner`, `summarizer`)
  - `version`: current orchestration schema version (`2`)
- Model initialization now memoizes failures to avoid repeated expensive import/load attempts under constrained environments.
