# ATO Tax Assistant — Deployment Brief

> **Scope update (2026-07-03):** the actual deployment target is now the successor project
> `regtech-compliance-rag` (new repo, HKMA/SFC corpus, layered learning build — see RegTech
> build guide). This repo stays as the private reference implementation. The budget, the
> "production deployment on AWS" CV bar, and the AWS-account checklist below all carry over
> unchanged. Architecture decision + RegTech evolution hooks: see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Current State

**What it is:** A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
Australian tax returns based on official ATO documents (currently: ATO Individual Tax Return
Instructions 2024, NAT 71050, 17 MB PDF, 289 chunks).

**Tech stack:**

| Layer | Tech |
|---|---|
| Framework | LangChain |
| Vector store | ChromaDB (local, default) or Pinecone (cloud, via `VECTOR_STORE_TYPE=pinecone`) |
| LLM / Embeddings | Google Gemini 2.5 Flash + `gemini-embedding-001` (primary); OpenAI GPT-4o + `text-embedding-3-small` (alt) |
| UI | Streamlit |
| Async ingestion | asyncio + tenacity (exponential backoff, up to 7 retries) |
| Platform | Linux / Python 3.13, tested on aarch64 (Raspberry Pi) |

**Current status:** Fully working locally. All core pipeline passes (PDF load → chunk → embed →
ChromaDB persist → similarity search → LLM answer). Streamlit UI with confidence dashboard and
source document viewer is complete. Pinecone backend wired but not battle-tested in prod.

---

## Monthly Budget Ceiling

**Hard limit: US$15/month** (side project — treat it like a phone bill you don't want to see).

Breakdown target:

| Item | Target |
|---|---|
| AWS compute (EC2 t3.micro or Fargate Spot) | ~$5–8 |
| AWS data transfer + storage (S3, EBS) | ~$1–2 |
| Gemini API (free tier: 1,500 req/day on Flash) | $0 under low traffic |
| Pinecone serverless free tier (100k vectors) | $0 for this dataset (289 chunks) |
| Misc (Route 53, ACM cert, CloudWatch logs) | ~$1–2 |
| **Total** | **~$7–12, ceiling $15** |

**Levers if cost spikes:**
- Switch from EC2 always-on → Fargate Spot (70% cheaper, acceptable for demo app)
- Keep ChromaDB (no Pinecone cost at all — 289 chunks fits on disk trivially)
- Gemini free tier covers ~50 questions/day; only pay if traffic exceeds that

---

## Deployment Goal

> **"Production deployment on AWS"** — suitable for CV / resume under Projects.

Minimum bar to claim this honestly:
- App reachable on a public HTTPS URL (not `localhost`, not ngrok)
- Running continuously without manual restarts (health check + auto-restart)
- Environment secrets managed properly (not hardcoded, not in repo)
- At least one infra-as-code artefact (e.g., `docker-compose.yml`, CDK stack, or
  `apprunner.yaml`) checked into the repo

Target architecture (simplest path that meets the bar):

```
Route 53 (optional)  →  CloudFront / ALB  →  EC2 t3.micro
                                               └── Docker: Streamlit app (port 8501)
                                               └── chroma_db/ on EBS volume
                                               └── .env from AWS Secrets Manager or SSM
```

Alternative (serverless, zero-ops, slightly higher cold start):

```
AWS App Runner  →  ECR image  →  Streamlit (port 8501)
                                  chroma_db persisted to EFS mount (or swap to Pinecone)
```

**Recommended first step:** EC2 t3.micro + Docker + Elastic IP. Cheapest, most CV-legible,
no cold-start issues for a demo. Total setup time: ~2–3 hours.

---

## AWS Account Status

**Action required — fill this in before starting:**

| Question | Your answer |
|---|---|
| AWS account age | ??? |
| Free tier still active? (expires 12 months after signup) | ??? |
| Free tier EC2 remaining this month (750 hrs t3.micro/t2.micro) | ??? |
| Free tier S3 / EBS remaining | ??? |
| Any existing resources consuming free tier quota? | ??? |

**If free tier is active:** EC2 t3.micro + 30 GB EBS = $0 compute cost for up to 12 months.
Total cost ≈ $2–3/month (data transfer + Elastic IP + optional Route 53).

**If free tier is expired:** t3.micro on-demand = ~$0.0104/hr ≈ $7.50/month. Still within budget.
Consider Reserved Instance (1-year, no upfront) at ~$4.60/month if keeping it running > 3 months.

**Check your free tier usage:** AWS Console → Billing → Free Tier

---

## Pre-deployment Checklist

- [ ] Confirm AWS account free tier status (see above)
- [ ] Write `Dockerfile` for the Streamlit app
- [ ] Add `.env.example` with all required keys documented (no real values)
- [ ] Move secrets to AWS SSM Parameter Store or Secrets Manager
- [ ] Decide vector backend: ChromaDB on EBS (simple) vs Pinecone serverless (no disk mgmt)
- [ ] Add `requirements.txt` version pins (currently unpinned — will break on redeploy otherwise)
- [ ] Set up CloudWatch log group (free tier: 5 GB/month)
- [ ] Register a domain or use EC2 public DNS for the CV URL

---

## Files to Create for Deployment

```
ato-tax-assistant/
├── Dockerfile                  # to create
├── docker-compose.yml          # to create (local parity with prod)
├── .env.example                # to create
├── infra/
│   ├── ec2-setup.sh            # to create (user-data bootstrap script)
│   └── (optional) cdk/         # to create if going IaC route
└── DEPLOY_BRIEF.md             # this file
```
