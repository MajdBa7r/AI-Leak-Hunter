# Deployment and repository setup

This guide takes the code from `AI-Leak-Hunter_clean_repo.zip` to a live Streamlit Community Cloud app. Every command was run on a fresh clone of this exact tree.

---

## 1. Use one archive: `AI-Leak-Hunter_clean_repo.zip`

Do **not** merge the older archives. Their relationship:

| Archive | What it is | Use it? |
|---|---|---|
| `ai_leak_hunter_product.zip` (63 files) | Product files only, meant to be unzipped over the old repo | **No.** `ai_leak_hunter_copilot.zip` contains all 63 of them (newer versions where they changed) |
| `ai_leak_hunter_copilot.zip` (68 files) | Product + copilot, still an "add on top of the old repo" overlay | **No.** It lacks the datasets, and carries no cleanup |
| `leak_hunter_*.zip`, `*.diff` | Earlier stage deliveries | **No.** Superseded |
| **`AI-Leak-Hunter_clean_repo.zip`** (93 files) | The complete, final repository: code, datasets, published reports, docs, tests | **Yes, this one only** |

What the clean archive leaves out on purpose:

- **Generated folders:** `outputs/`, `outputs_rules/` and `data_replica_seed2026/` are rebuilt by the app or by `scripts/run_all.sh`.
- **The old secret-scanner prototype:** `main.py`, `scanner.py`, `detector.py`, `rules.py` and the misnamed `__init.__py`. Nothing in the product imports them. A proper `src/leak_hunter/__init__.py` replaces the misnamed file.
- **Dead folders:** the empty `src/leak_hunter/Code/` and the `__init__.py` inside the data folder `Dataset/Data/Unified/`.
- **Caches and local files:** `__pycache__/`, `.pytest_cache/`, local SQLite logs and any real `secrets.toml`.

### Final tree (exactly what `git ls-files` lists)

```
AI-Leak-Hunter/
├── .streamlit/
│   ├── config.toml                 theme + headless server settings
│   └── secrets.toml.example        template only, real secrets.toml is git-ignored
├── app/
│   ├── charts.py
│   └── streamlit_app.py            ← Streamlit entry point
├── docs/
│   ├── DEPLOYMENT.md               this guide
│   ├── detection_engine.md
│   └── PRODUCT.md
├── reports/                        detection evaluation (published results)
├── reports_ml/                     unsupervised benchmark (+ ROC/PR and SHAP plots)
├── reports_queue/                  queue evaluation, DECISION_RULE.md, original_dataset_v1/, replica_seed2026/
├── reports_rules/                  rules benchmark + noise benchmark
├── scripts/
│   ├── booth_check.py              one-command check + launch (Windows/macOS/Linux)
│   ├── make_replica.py
│   ├── rules_noise_benchmark.py
│   ├── run_all.sh                  regenerates every output and report (~8 min)
│   └── unsupervised_benchmark.py
├── src/leak_hunter/
│   ├── Dataset/
│   │   ├── Data/                   5 procurement CSVs + ground_truth.csv + validation_report.md
│   │   │   └── Unified/            unified CSV, schema, build script
│   │   ├── generate_synthetic_data.py
│   │   └── README.md
│   ├── __init__.py
│   ├── anomaly_detection.py  case_store.py  config.py  copilot.py  data_loader.py
│   ├── detection_rules.py    evaluation.py  exposure.py  findings.py  history.py
│   ├── investigation.py      pipeline.py    prioritization.py  queue_evaluation.py
│   └── reconciliation.py     scenarios.py
├── tests/                          8 test files (unit, integration, robustness, app, copilot)
├── .gitignore
├── pytest.ini
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

The `reports*/` folders are committed on purpose: the app's **Evaluation** page reads them, and they are the preserved, published results.

---

## 2. Production configuration files

These are already in the archive. They are reproduced here for review.

### `requirements.txt` (what Streamlit Cloud installs)

```text
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
streamlit>=1.40
plotly>=5.20
requests>=2.31
```

The copilot calls OpenAI, Groq and Gemini over HTTPS with `requests`, so no vendor SDK is needed. Test and benchmark tools live in `requirements-dev.txt` (`-r requirements.txt`, `pytest`, `matplotlib`, `shap`) and are **not** installed on the cloud.

### `.gitignore`

```gitignore
# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# --- Virtual environments ---
.venv/
venv/
env/

# --- Secrets: never commit real keys ---
.streamlit/secrets.toml
.env
.env.*

# --- Generated at runtime (the app rebuilds outputs/ on first load) ---
outputs/
outputs_*/
data_replica_seed*/

# --- Local databases (case decision log) ---
*.sqlite
*.sqlite3
*.db
*.db-journal
*-wal
*-shm

# --- Archives and OS / editor clutter ---
*.zip
.DS_Store
Thumbs.db
.idea/
.vscode/
```

### `.streamlit/secrets.toml.example`

```toml
# Copy to .streamlit/secrets.toml (local) or paste into Streamlit Cloud > App settings > Secrets.
# Set ONE provider key. With no key, the copilot runs in offline mode (templates over the case evidence).

OPENAI_API_KEY = "sk-..."
# GROQ_API_KEY = "gsk_..."
# GEMINI_API_KEY = "AIza..."

# Optional
# LEAK_HUNTER_LLM_PROVIDER = "openai"      # openai | groq | gemini - preferred when several keys are set
# OPENAI_MODEL = "gpt-4o-mini"
# GROQ_MODEL = "llama-3.3-70b-versatile"
# GEMINI_MODEL = "gemini-2.5-flash"
# LEAK_HUNTER_COPILOT_OFFLINE = "true"     # force offline mode (booth safety)
```

`GOOGLE_API_KEY` is also accepted for Gemini. Every value can come from an environment variable instead of secrets. Environment variables take precedence.

---

## 3. Create the GitHub repository and push

**On GitHub:** New repository → name `AI-Leak-Hunter` → **leave it empty.** Do not add a README, `.gitignore` or license, or the first push is rejected.

### macOS / Linux

```bash
unzip AI-Leak-Hunter_clean_repo.zip          # creates ./AI-Leak-Hunter
cd AI-Leak-Hunter

git init -b main
git add .
git status --short | grep -E "secrets.toml$|\.sqlite|outputs/" && echo "STOP: something private is staged" || echo "OK: nothing private staged"
git commit -m "AI Leak Hunter: detection engine, investigation app and copilot"

git remote add origin https://github.com/<YOUR-USER>/AI-Leak-Hunter.git
git push -u origin main
```

### Windows (PowerShell)

```powershell
Expand-Archive AI-Leak-Hunter_clean_repo.zip -DestinationPath .
cd AI-Leak-Hunter

git init -b main
git add .
git status --short          # must NOT list secrets.toml, *.sqlite or outputs/
git commit -m "AI Leak Hunter: detection engine, investigation app and copilot"

git remote add origin https://github.com/<YOUR-USER>/AI-Leak-Hunter.git
git push -u origin main
```

GitHub asks for a **personal access token** as the password over HTTPS, not your account password. With the GitHub CLI you can skip creating the repo by hand:

```bash
gh repo create AI-Leak-Hunter --public --source . --remote origin --push
```

If `git commit` complains about identity, run this once first:
`git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.

The repository must be **public**, or your Streamlit account must be authorized for private repos.

---

## 4. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → **Deploy a public app from GitHub**.
2. Fill in:

   | Field | Value |
   |---|---|
   | Repository | `<YOUR-USER>/AI-Leak-Hunter` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |
   | App URL | optional, e.g. `ai-leak-hunter` |
   | Advanced settings → Python version | **3.12** or **3.13** (verified on 3.13) |
   | Advanced settings → Secrets | the snippet below |

3. Click **Deploy**. The build installs the six packages from `requirements.txt`.

### Secrets snippet (App settings → Secrets)

Set **one** key. Leave all keys out for a no-key, offline deployment.

```toml
OPENAI_API_KEY = "sk-..."
# or: GROQ_API_KEY = "gsk_..."
# or: GEMINI_API_KEY = "AIza..."

# optional:
# LEAK_HUNTER_LLM_PROVIDER = "openai"   # openai | groq | gemini
# OPENAI_MODEL = "gpt-4o-mini"
# LEAK_HUNTER_COPILOT_OFFLINE = "true"  # force offline copilot for everyone
```

Changing secrets later (⋮ → **Settings** → **Secrets** → Save) takes effect within about a minute without a redeploy.

### First-run self-healing: how the app builds its own data

`outputs/` is git-ignored, so a fresh deploy has no detection results. On the first page load, `app/streamlit_app.py` does this:

1. **Check.** Is `outputs/run_summary.json` missing?
2. **Build.** If it is missing, it shows the spinner *"Running detection (rules + Isolation Forest)..."* and calls `run_detection()` + `save_outputs()` from `leak_hunter.pipeline`. That reads the committed CSVs in `src/leak_hunter/Dataset/Data/` (never `ground_truth.csv`) and writes `findings.csv`, `incidents.csv`, `if_scores.csv`, `supplier_profiles.csv` and the rest to `outputs/`.
3. **Reload.** It reruns itself and loads the results. In the verified fresh-clone run, detection took about 21 s and the first page appeared after about 35 s.
4. **Recover.** If detection fails, the page shows the error and a **Try again** button instead of crashing.

Later visitors share the cached results. The **Re-run detection** button in the sidebar rebuilds them on demand.

**Decision-log caveat.** Analyst decisions go to `outputs/case_decisions.sqlite`. Streamlit Cloud's disk is ephemeral, so a reboot, redeploy or sleep wipes it, and the data is rebuilt automatically. That is fine for a demo. For persistence, set `LEAK_HUNTER_CASE_DB` to durable storage.

**Before the talk:** apps on Community Cloud go to sleep after a period with no visitors. Open the URL 5 minutes before presenting. If it shows "This app has gone to sleep", click **Yes, get this app back up** and let the first-run build finish.

---

## 5. Local and booth quick-start

### One command: build, test, launch

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
# PowerShell blocks Activate.ps1? run once:  Set-ExecutionPolicy -Scope Process Bypass

python scripts/booth_check.py                 # build outputs → 35 smoke tests (~45 s) → open the app
python scripts/booth_check.py --offline       # same, copilot forced offline
python scripts/booth_check.py --full          # full suite (several minutes) instead of the smoke tests
python scripts/booth_check.py --no-launch     # checks only
```

The script runs the detection pipeline, then `tests/test_copilot.py`, `tests/test_product.py` and `tests/test_exposure.py`. Those cover app pages, the decision log, the copilot (offline, mocked providers, fallback, citation check) and the exposure arithmetic. It then starts the app at `http://localhost:8501`. If any check fails, it stops with *"do not demo until this is fixed"*.

Real API keys and the offline switch are kept out of the test environment, so they cannot change the result.

To launch the app without the checks: `streamlit run app/streamlit_app.py`.

### Force the offline copilot when Wi-Fi is unstable

| Where | How | Effect |
|---|---|---|
| In the running app (instant) | Sidebar → turn on **"Copilot: offline mode only"** | This browser session; caption changes to *Offline copilot (no network needed)* |
| Booth laptop, at launch | `python scripts/booth_check.py --offline` | Whole app |
| Any launch, macOS/Linux | `LEAK_HUNTER_COPILOT_OFFLINE=true streamlit run app/streamlit_app.py` | Whole app |
| Any launch, PowerShell | `$env:LEAK_HUNTER_COPILOT_OFFLINE="true"; streamlit run app/streamlit_app.py` | Whole app |
| Streamlit Cloud | Secrets: `LEAK_HUNTER_COPILOT_OFFLINE = "true"` | Every visitor |

Even without the switch, any provider failure (timeout, no network, quota, unknown model) falls back to the offline copilot for that answer. The caption under the answer says so.

The switch is still the better booth choice: it skips the network timeout, so answers are instant.

The app itself needs no internet once installed. Detection, the case file and the offline copilot all run locally.

### Pre-talk checklist

- [ ] `python scripts/booth_check.py --no-launch` ends with **All checks passed**.
- [ ] The live URL is awake and shows the Overview page.
- [ ] With a key set: ask one quick prompt in a Case file. The caption should name the model, not *Offline copilot*. Model names can be retired by providers; override with `OPENAI_MODEL`, `GROQ_MODEL` or `GEMINI_MODEL`.
- [ ] Have the booth laptop's local app running as a backup tab.

### Regenerate every report (optional, ~8 min, macOS/Linux/Git Bash)

```bash
bash scripts/run_all.sh
```
