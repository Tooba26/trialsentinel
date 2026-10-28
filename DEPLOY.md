# Deploying TrialSentinel

Two pieces go live: the **model API** (FastAPI in Docker) and the **dashboard** (Next.js). Deploy the API
first, because the dashboard needs its URL. Total time is about 30 minutes, and all three services below have
free or low-cost tiers.

```
Browser ──► Vercel: Next.js dashboard + serverless route handlers (/api/predict, /api/sponsors)
                          │  server-side fetch, API_URL env var
                          ▼
            Render: Docker container running FastAPI + the model bundle
```

## 1. Push the repository to GitHub

```bash
cd trialsentinel
git init && git add . && git commit -m "TrialSentinel: clinical trial termination risk"
gh repo create trialsentinel --public --source . --push     # or create it on github.com and push
```

The raw CSV and processed parquet are git-ignored. The trained artifacts in `artifacts/` are committed so the
API can be built without retraining.

## 2. Deploy the API on Render

1. Sign in at <https://render.com> with GitHub.
2. **New → Blueprint**, pick the `trialsentinel` repository. Render reads `render.yaml` and creates
   `trialsentinel-api` from `api/Dockerfile`.
3. Wait for the first build (5–8 minutes; it installs LightGBM, XGBoost, CatBoost and SHAP).
4. Open `https://trialsentinel-api.onrender.com/health`. You should see `{"status":"ok", ...}`, and the
   interactive API docs are at `/docs`.

**Plan note:** with the model loaded the service uses about 410 MB of RAM and answers a prediction in under
100 ms. That fits Render's free plan (512 MB), but with little headroom, and free services sleep after 15
minutes idle, so the first request after a pause takes about a minute. For a demo you share widely, the
**Starter** plan avoids both. To try the free plan, change `plan: starter` to `plan: free` in `render.yaml`.

**Alternatives** (same Docker image, no code changes):

* **Railway:** New Project → Deploy from GitHub → set *Dockerfile path* to `api/Dockerfile` and the root
  directory to the repository root. Railway injects `PORT`, which the image already respects.
* **AWS App Runner:** push the image to ECR (`docker build -f api/Dockerfile -t trialsentinel-api .`), create
  an App Runner service from it with port 8000 and health check path `/health`, 1 vCPU / 2 GB.

## 3. Deploy the dashboard on Vercel

1. At <https://vercel.com/new>, import the same repository.
2. Set **Root Directory** to `web`. The framework is detected as Next.js.
3. Add the environment variables:
   * `API_URL` = `https://trialsentinel-api.onrender.com` (no trailing slash)
   * `NEXT_PUBLIC_REPO_URL` = your GitHub repository URL (used by the "Code" link)
4. Deploy. The dashboard calls the API from Vercel's serverless functions, so the browser never needs CORS
   access to Render.
5. Back on Render, set `ALLOWED_ORIGINS` to your Vercel URL (only matters if you call the API directly from a
   browser).

**Netlify** works the same way: base directory `web`, build command `npm run build`, and the same
environment variables. The Next.js runtime is installed automatically.

If the API is unreachable at build time, the page falls back to the analytics snapshot in
`web/data/overview.json`, so the site still renders. Only live scoring needs the API.

## 4. Publish the Kaggle notebook

1. On Kaggle, **Datasets → New dataset**, upload `ctg-studies.csv` (cite ClinicalTrials.gov as the source).
2. **Code → New notebook → File → Import notebook**, upload `notebooks/trialsentinel_eda_and_model.ipynb`.
3. Add your dataset and a GitHub dataset or utility script with the `src/trialsentinel` package, then change
   the paths in the first code cell (`C.RAW_CSV`) to `/kaggle/input/...`.
4. Run all, make it public, and paste the link into the README's *Links* section.

## 5. Final checks

- [ ] `/health` returns `ok` on the deployed API
- [ ] A prediction from the dashboard returns in under 2 seconds once the API is warm
- [ ] Dark and light mode both readable; keyboard tab order works through the form
- [ ] README links (demo, notebook, API docs) updated

## Updating the model

Download a fresh export, then run `make data train analytics notebook test`, commit `artifacts/` and
`web/data/overview.json`, and push. Render and Vercel redeploy automatically.
