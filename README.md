# AI Leak Hunter

Procurement intelligence that finds financial leakage across purchase orders, goods receipts, invoices and suppliers, ranks the cases worth investigating, and helps an analyst work each one.

- **Detection:** four auditable rules (duplicate invoices, quantity over-billing, PO price mismatch, historical price spikes) plus an Isolation Forest for unusual transactions.
- **Prioritization:** one case per incident with de-duplicated exposure, a transparent risk score and an expected-value queue.
- **Investigation app (Streamlit):** case queue, evidence, calculation steps, timeline, supplier profiles and an append-only decision log.
- **Investigation Copilot:** answers grounded in the case evidence, with OpenAI, Groq or Gemini, and an offline mode that needs no API key.

The data is synthetic (seed 42). See [docs/PRODUCT.md](docs/PRODUCT.md), [docs/detection_engine.md](docs/detection_engine.md) and, to deploy, [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Run it

```bash
pip install -r requirements-dev.txt
python scripts/booth_check.py            # builds outputs, runs smoke tests, opens the app
python scripts/booth_check.py --offline  # same, copilot never calls an LLM
```

App only: `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`. The first load builds `outputs/` automatically.

Copilot keys: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set one key. The real file is git-ignored.

## Layout

```
app/                     Streamlit app (entry point: app/streamlit_app.py)
src/leak_hunter/         detection engine, prioritization, investigation, copilot
src/leak_hunter/Dataset/ synthetic data, generator and data dictionary
tests/                   pytest suite (unit, integration, robustness, app, copilot)
scripts/                 benchmarks, replica generator, run_all.sh, booth_check.py
reports*/                published evaluation results
docs/                    product and engine documentation
```
