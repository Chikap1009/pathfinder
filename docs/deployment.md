# PathFinder — Deployment

Three free-tier targets give you a public, persistent demo URL:

| Layer | Provider | Free tier | Plays well with |
| ----- | -------- | --------- | --------------- |
| Frontend (Next.js 16) | **Vercel Hobby** | unlimited static + edge fn time | direct push from GitHub |
| Backend (FastAPI 0.115+) | **HuggingFace Spaces (Docker)** | 2 vCPU, 16 GB RAM, sleeps after ~50 min idle | docker subdomain + auto-deploy on git push |
| Knowledge graph (Neo4j 5) | **Neo4j AuraDB Free** | 50k nodes / 175k rels, auto-pauses after 3 days idle | Bolt URI + password |

Combined cost: **$0/mo**. Sleeps are mitigated by the existing keep-alive
GitHub Actions workflow (`.github/workflows/keepalive.yml`) which pings
`/health` and AuraDB every 5 minutes.

---

## Architecture in production

```
                user browser
                     │
                     ▼
     pathfinder.vercel.app          ← Vercel CDN, serves Next.js
                     │
                     │  POST /v1/search/stream
                     ▼
     <user>-pathfinder.hf.space     ← HF Space Docker, FastAPI + bundled data
            │            │
            │            └────────► AuraDB Free (Bolt over TLS)
            └────────► Gemini Flash-Lite (intent + paraphrase)
```

CORS allow-list is read from the `API_CORS_ORIGINS` env var on the HF Space.

---

## Step 1 · Knowledge graph → Neo4j AuraDB Free

You'll do this once and the credentials get stored in the HF Space secrets.

1. Sign up at <https://console.neo4j.io/> (free, GitHub auth works).
2. **Create instance → AuraDB Free → Region:** any close to you. Neo4j 5.x.
3. **Save the auto-generated password** — they only show it once! Copy it
   into a password manager, NOT this repo.
4. Connection URI looks like `neo4j+s://<random-id>.databases.neo4j.io`
   (`+s` enables TLS; required for AuraDB).
5. Wait ~30 s for "Running" status, then back at your dev machine:

   ```bash
   # Set env vars locally (do NOT commit)
   export NEO4J_URI=neo4j+s://<your-id>.databases.neo4j.io
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=<the-saved-password>

   # Re-run the ingest pointed at AuraDB
   uv --directory apps/api run python scripts/07_kg_build.py --reset
   ```

   The `07_kg_build.py` script reads `NEO4J_URI/USER/PASSWORD` from env (via
   `pydantic-settings`) and writes 9,138 nodes + 45,731 relationships in
   ~30 s on a decent connection.

6. Verify with the Neo4j browser at the AuraDB console → "Open with Neo4j
   Browser" → run `MATCH (n) RETURN count(n)`.

---

## Step 2 · Backend → Hugging Face Spaces

1. Sign up at <https://huggingface.co/join>.
2. Create a write token: <https://huggingface.co/settings/tokens> → "Create
   new token" → role **Write** → name e.g. `pathfinder-deploy`. Save the
   token (`hf_…`).
3. Create the Space: <https://huggingface.co/new-space>
   - Owner: your username
   - Space name: `pathfinder-api`
   - License: MIT
   - SDK: **Docker**
   - Hardware: **CPU basic - free**
   - Visibility: Public
4. Configure secrets in the Space's **Settings → Variables and secrets**:

   | Type | Name | Value |
   | ---- | ---- | ----- |
   | Secret | `NEO4J_URI` | `neo4j+s://<id>.databases.neo4j.io` |
   | Secret | `NEO4J_USER` | `neo4j` |
   | Secret | `NEO4J_PASSWORD` | (your AuraDB password) |
   | Secret | `GEMINI_API_KEY` | (your Gemini key from Day 2) |
   | Variable | `API_CORS_ORIGINS` | `https://pathfinder-<your-vercel-id>.vercel.app,http://localhost:3000` |
   | Variable | `APP_ENV` | `production` |
   | Variable | `LOG_LEVEL` | `INFO` |
   | Variable | `QDRANT_URL` | `http://0.0.0.0:6333` (placeholder; Qdrant isn't used in v1 retrieval — in-memory cosine instead) |

5. Push the backend code to the Space:

   ```bash
   # From repo root
   cd apps/api
   ./scripts/stage_for_hf.sh        # stages 15 MB of pre-computed indexes / embeddings

   # First-time push (one-off)
   git init -b main
   git add .
   git commit -m "Initial PathFinder API push"
   git remote add space https://<your-username>:<hf-token>@huggingface.co/spaces/<your-username>/pathfinder-api
   git push --force space main
   ```

   The Space starts building (~5-7 min). Watch progress at
   `https://huggingface.co/spaces/<your-username>/pathfinder-api/logs`.

6. Once "Running", smoke-test:
   ```bash
   curl https://<your-username>-pathfinder-api.hf.space/health
   curl -X POST https://<your-username>-pathfinder-api.hf.space/v1/search \
       -H "content-type: application/json" \
       -d '{"query":"Test Manager Selenium Azure","pipeline":"rrf3","top_k":3}'
   ```

7. **First request takes ~30 s** while BGE-M3 + cross-encoder load. Subsequent
   requests are ~1-3 s on CPU (the GPU rerank is ~285 ms; on free-tier CPU
   expect ~4-7 s for a `rrf3_rerank` pipeline). Default to `rrf3` (no
   rerank) for snappy demos; full pipeline for the quality story.

---

## Step 3 · Frontend → Vercel

1. Sign up at <https://vercel.com/signup> with the **same GitHub account**
   that owns the `pathfinder` repo.
2. After signup, click **Add New → Project**.
3. **Import the `pathfinder` repo.** Vercel auto-detects Next.js.
4. **Root Directory:** click "Edit" and set to `apps/web` (this monorepo
   has the Next.js app under apps/web).
5. **Environment Variables** — add one:

   | Name | Value |
   | ---- | ----- |
   | `NEXT_PUBLIC_API_BASE_URL` | `https://<your-username>-pathfinder-api.hf.space` |

6. Click **Deploy**. First build takes ~2 min.
7. Vercel assigns a URL like `pathfinder-xyz.vercel.app`. Future pushes to
   `main` auto-deploy.

---

## Step 4 · Wire the keep-alive cron

`.github/workflows/keepalive.yml` pings `/health` every 5 min and prevents
HF Space sleeps + AuraDB pauses. It needs:

1. Repo variable: `API_URL` = `https://<your-username>-pathfinder-api.hf.space`
2. Repo secrets: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (same values as the HF Space).

Set both at <https://github.com/Chikap1009/pathfinder/settings/secrets/actions>.

---

## Step 5 · Verify the live demo

| URL | Expected behavior |
| --- | ----------------- |
| `pathfinder-xyz.vercel.app/` | Home page with hero + 3 KPI cards |
| `pathfinder-xyz.vercel.app/search` | Search input. Submit a query → SSE stages stream in (intent → encode → bm25 → dense → kg → rrf [→ rerank] → results) |
| `pathfinder-xyz.vercel.app/eval` | KPI tiles + 3 ablation tables + latency budget |
| `pathfinder-xyz.vercel.app/result/person_181457` | Full profile detail rendered server-side |
| `<hf-space>.hf.space/docs` | FastAPI auto-generated Swagger UI |
| `<hf-space>.hf.space/v1/eval/summary` | Raw JSON snapshot of the eval state |

---

## Troubleshooting

**Vercel build fails with "couldn't find pages directory":** double-check
the Root Directory is `apps/web`, not the repo root.

**HF Space build OOMs during torch install:** the CPU torch wheel is ~750 MB.
Free Spaces have a 16 GB build cache cap; this should be fine. If you see
"no space left on device", the Space hit a transient issue — click
"Restart Space" to retry.

**HF Space stuck on "Building" >15 min:** it's downloading the BGE-M3
weights for the first time (~2.3 GB). Watch the logs; progress will show
under "Loading weights".

**CORS error in the browser console:** the Space's `API_CORS_ORIGINS`
env var doesn't include your Vercel URL. Re-set it in HF Space settings
and click "Restart Space".

**Search returns 500 with "Neo4j unavailable":** AuraDB auto-paused. Visit
the AuraDB console; click "Resume" on the instance. The keep-alive cron
should prevent this from recurring.

**First search takes forever:** BGE-M3 + cross-encoder loading on cold
start. ~30 s on CPU. The lifespan warmup task tries to pre-load these but
the Space's first ever boot still has them downloading.

---

## Updating the live demo

```bash
# Backend changes
cd apps/api
./scripts/stage_for_hf.sh       # if data files changed
git add . && git commit -m "..."
git push space main             # auto-deploys

# Frontend changes
git push origin main            # Vercel auto-deploys from main
```

The keep-alive cron handles ongoing uptime; nothing else to do.
