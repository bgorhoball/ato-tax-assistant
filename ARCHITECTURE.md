# ATO Tax Assistant — Deployment Architecture Decision

> Companion to [DEPLOY_BRIEF.md](DEPLOY_BRIEF.md). Budget ceiling: **US$15/month**.
> Prices below were verified against public sources on **2026-07-03** (links in References).
> Anything not verifiable today is marked **[待確認]** — do not treat those as quotes.
>
> **Scope update (2026-07-03):** this architecture now also targets the successor project
> (`regtech-compliance-rag`, new repo, HKMA/SFC corpus — see RegTech build guide). The ato repo
> becomes the private reference implementation. The core decision below is unchanged; the
> [Evolution hooks](#evolution-hooks-for-the-regtech-rag-successor) section maps each planned
> Layer 4/5 enhancement onto this infra. Copy this file + DEPLOY_BRIEF.md into the new repo's
> `infra/` at Phase 0.

---

## Decision

**ECS Fargate (Spot) + Cloudflare free tier for TLS/DNS. Vector DB: ChromaDB baked into the
Docker image. Pinecone kept as a supported-but-off alternative.**

Estimated monthly cost: **~US$7–8** (Spot) / worst case ~US$14 if forced onto on-demand.

---

## The constraint that decides everything: Streamlit needs WebSockets

Streamlit is not a request/response app. It runs a persistent Tornado server; every browser
session holds an **open WebSocket** carrying script reruns and `st.session_state`. Any platform
that cannot terminate and hold WebSocket connections cannot run Streamlit, full stop. Two of the
three candidates fail on exactly this.

---

## Options compared

### ❌ AWS App Runner — rejected (hard technical blocker)

- App Runner's managed load balancer **does not support WebSockets**. The feature request
  ([apprunner-roadmap #13](https://github.com/aws/apprunner-roadmap/issues/13), open since
  May 2021) is now **closed as "not planned"** — verified 2026-07-03. This is not "wait for it",
  it is never coming.
- Streamlit community threads confirm apps deploy but hang at "Please wait..." because the
  WebSocket upgrade is dropped.
- Pricing therefore not evaluated in depth — no price makes a platform that can't run the app
  acceptable.
- *Would otherwise have been attractive:* simplest container deploy on AWS, scale-to-few,
  built-in HTTPS.

### ❌ Lambda + API Gateway — rejected (architectural mismatch)

- Lambda is request-scoped with a 15-minute hard cap; Streamlit needs a long-running server
  process holding per-connection state in memory. There is no sticky routing to a warm Lambda
  instance across a session.
- API Gateway *does* have a "WebSocket API", but it is a message-brokering model
  (`$connect`/`$disconnect`/route handlers), not a transparent socket to your server. Streamlit's
  Tornado protocol cannot run behind it without rewriting Streamlit's transport layer.
- Lambda Web Adapter enables HTTP streaming for containerised web apps, but not inbound
  WebSocket serving.
- Making this work means abandoning Streamlit for a split frontend/backend (e.g. static S3 site +
  Lambda REST API). That is a different project, not a deployment choice.
- *CV note:* "I rewrote the app to force it onto Lambda" is a worse story than "I chose the right
  compute for the workload".

### ✅ ECS Fargate — selected

- Runs any long-lived container; WebSockets are just TCP to your task. No platform fight.
- **CV value is the strongest of the three.** A real Fargate deployment legitimately exercises:
  ECR, task definitions, IAM task roles, Secrets Manager/SSM secret injection, CloudWatch Logs,
  Spot capacity providers, security groups — and optionally GitHub Actions → ECR → ECS CI/CD
  with OIDC. That's a full production-shaped AWS story for interviews, vs App Runner ("I clicked
  deploy") or Lambda (wrong tool, above).
- Fargate Spot risk (2-minute interruption notice) is acceptable: the ECS service replaces the
  task automatically, and a demo app tolerates a ~1-min blip. Chat history loss on interruption
  is acceptable for a demo (state is per-session anyway).

**Task sizing:** 0.25 vCPU / 1 GB (Python + LangChain + Streamlit comfortably; the Chroma index
is only 13 MB). Linux/x86, us-east-1 for verified pricing. Sydney (ap-southeast-2) rates differ
**[待確認]** — us-east-1 is fine for a demo; ATO PDF Q&A is not latency-sensitive.

**HTTPS without blowing the budget:** the textbook front door is an ALB, but an ALB alone is
**$0.0225/hr ≈ $16.20/month + LCU charges** — it busts the $15 ceiling before compute.
Budget answer: **Cloudflare free tier** as proxied DNS in front of the task's public IP
(Cloudflare proxies WebSockets on the free plan), with a tiny EventBridge→Lambda hook updating
the DNS record when ECS replaces the task. Slightly janky, costs $0, and is itself a talking
point. Document ALB as the scale-up path.

---

## Vector DB: ChromaDB baked into the image (Pinecone stays optional)

**Chosen: ChromaDB, shipped read-only inside the Docker image.**

- The entire index is a **13 MB sqlite file** (289 chunks). Ingestion happens offline at build
  time; the running app only reads. Copying `chroma_db/` into the image makes the container
  **stateless and immutable** — no EFS, no volumes, no external DB, nothing to pay for or babysit.
  Re-ingest ⇒ rebuild image ⇒ redeploy, which is exactly the immutable-infra story you want to
  tell in an interview.
- **Pinecone free (Starter) tier** would also cost $0 and easily fits (2 GB storage, ~2 M write /
  1 M read units per month, single region us-east-1). But the Starter plan **pauses indexes after
  ~3 weeks of inactivity** — fatal for a CV demo that a recruiter might open after a quiet month,
  greeted by an error. Exact current quota numbers: **[待確認]** against
  [pinecone.io/pricing](https://www.pinecone.io/pricing/) before relying on them.
- The codebase already supports both via `VECTOR_STORE_TYPE` (see `src/rag_engine_v2.py`), so the
  dual-backend design remains a resume line and Pinecone can be switched on with one env var if
  the corpus ever outgrows an image (roughly >1–2 GB of index).

---

## Monthly cost estimate (us-east-1, 24×7)

Verified unit prices (2026-07-03): Fargate on-demand **$0.04048/vCPU-hr** and
**$0.004445/GB-hr**; Fargate Spot vCPU **~$0.01291/vCPU-hr** (~68% off); public IPv4
**$0.005/hr**.

| Item | Basis | Monthly (730 hrs) |
|---|---|---|
| Fargate Spot vCPU (0.25) | 0.25 × $0.01291 × 730 | ~$2.36 |
| Fargate Spot memory (1 GB) | Spot GB-hr rate **[待確認]**; est. ~68% off on-demand | ~$1.0 (est.) |
| Public IPv4 address | $0.005 × 730 | $3.65 |
| ECR image storage (~1.5 GB) | rate **[待確認]** | <$0.50 (est.) |
| CloudWatch Logs | within 5 GB free tier | $0 |
| Data egress | demo traffic; free-tier egress allowance **[待確認]** | ~$0 |
| Cloudflare DNS/TLS | free plan | $0 |
| Gemini API (Flash + embeddings) | free tier at demo traffic | $0 |
| Pinecone | not used at runtime | $0 |
| **Total (Spot)** | | **~$7–8** |
| **Total if on-demand fallback** | vCPU $7.39 + mem $3.24 + IPv4 $3.65 + misc | **~$14–15** ⚠ at ceiling |

Notes:
- On-demand fallback sits right at the $15 ceiling — set the ECS capacity provider to
  Spot-only and accept downtime over cost, or set a billing alarm at $12.
- Rejected-option prices for the record: ALB $16.20/mo + LCU (why it's excluded); App Runner and
  Lambda pricing not itemised — both rejected on technical grounds, not cost.
- ARM/Graviton Fargate is ~20% cheaper than x86; whether Graviton is available on **Spot**:
  **[待確認]**. Worth checking — the dev box is aarch64, so images are ARM-native anyway.

---

## Resulting architecture

```
Browser ──HTTPS/WSS──▶ Cloudflare (free, proxied DNS, TLS, WebSocket passthrough)
                          │
                          ▼ (task public IP, security group allows Cloudflare ranges)
                    ECS Fargate Spot task (0.25 vCPU / 1 GB)
                    └── Docker image:
                        ├── Streamlit app (port 80 — Cloudflare only proxies
                        │   80/8080/8880/443/8443/…; 8501 is NOT on the list)
                        ├── chroma_db/ (13 MB, read-only, baked in at build)
                        └── secrets: GOOGLE_API_KEY via SSM Parameter Store → task def
                          │
                          ▼
                    Gemini API (LLM + embeddings, free tier)

EventBridge (ECS task state change) ──▶ Lambda ──▶ update Cloudflare DNS record
GitHub Actions (OIDC) ──▶ build/ingest ──▶ push ECR ──▶ ecs update-service   [optional CI/CD]
```

**Scale-up path (documented, not built):** ALB + ACM cert replaces Cloudflare hack;
Pinecone replaces baked-in Chroma when the corpus outgrows the image; on-demand replaces Spot
when uptime matters.

---

## Evolution hooks for the RegTech RAG successor

The RegTech build guide (Layers 4–5) adds four enhancements. None of them invalidates the
Fargate decision; each maps onto the existing infra as follows. **Do not build these now** —
this section exists so nothing chosen above blocks them later.

### 1. LLM: Gemini (dev) → Bedrock (prod narrative) — zero new infra

Provider swap is config + IAM, not architecture: add `bedrock:InvokeModel` to the **existing ECS
task role** and flip the provider env var (the engine is already provider-swappable). Bonus
security story: Bedrock auth is SigV4 via the task role — **no LLM API key exists anywhere**,
which upgrades the secrets-management interview answer.

Verified cost (2026-07-03, on-demand per 1M input/output tokens): **Nova Lite $0.06/$0.24**,
**Claude Haiku 4.5 $1/$5**. At demo traffic (~200 queries/mo, ~3k in + 500 out tokens each):
Nova Lite ≈ $0.06/mo, Haiku ≈ $1.10/mo — negligible against the $15 ceiling. Keep Gemini free
tier for dev/ingestion experiments; route prod queries to Bedrock.

Caveats: model availability per region varies — if the data-residency defence is pinned to a
specific region (e.g. ap-east-1 Hong Kong for HKMA/SFC data), Bedrock model availability there
is **[待確認]**. The generic "data never leaves the AWS boundary" line holds in any region.

### 2. Vector store: the guide's "Pinecone or OpenSearch Serverless" — OpenSearch is vetoed

- **OpenSearch Serverless: rejected on verified cost.** Minimum non-redundant dev config
  (0.5 OCU index + 0.5 OCU search at $0.24/OCU-hr) ≈ **$174/month**; production with redundancy
  ≈ **$350/month**. That is 10–20× the entire project budget for a corpus of 2–4 PDFs. Strike it
  from the guide.
- **Pinecone Starter becomes the prod-managed story if wanted.** The 3-week inactivity pause is
  killable for $0: an EventBridge cron → Lambda firing one trivial query per week keeps the index
  warm (and is itself a small serverless artefact for the repo).
- Baked-in Chroma remains the default: the HKMA/SFC corpus is the same order of magnitude as the
  ATO one (tens of MB indexed). Switch = `VECTOR_STORE_TYPE` env var, already wired.

### 3. Frontend split (Layer 5 "serverless story"): the only place Lambda comes back

The guide's "static frontend + API" variant is where Lambda container images become *correct*
(request/response API, no WebSocket): S3 + CloudFront static site → Lambda container (the RAG
backend; LangChain deps are why it must be a container image, not a zip) → Bedrock. Nothing in
the current stack blocks this: ECR is already there, same image family, and the Fargate Streamlit
app can run in parallel during migration. **The guide's "Streamlit on App Runner" option is dead**
— WebSocket support was closed as "not planned" (verified above); for Streamlit it is Fargate only.

### 4. PII masking (Presidio + spaCy NER): plan a memory bump, nothing else

NER models are RAM-hungry; when Layer 5 lands, resize the task **1 GB → 2 GB**
(≈ +$1/mo on Spot, est. — Spot GB-hr rate still **[待確認]**). Task definition change only.

### IaC note

The guide asks for Terraform/SAM. Recommendation: **Terraform** for the Fargate stack (task def,
service, security group, IAM roles, SSM params) — it covers the "IaC artefact in repo" CV bar
from DEPLOY_BRIEF.md and is the more recognisable tool in job ads. SAM only becomes relevant if
the Layer 5 Lambda split happens.

---

## References (verified 2026-07-03)

- App Runner WebSocket — closed as not planned: <https://github.com/aws/apprunner-roadmap/issues/13>
- Streamlit-on-App-Runner failures: <https://discuss.streamlit.io/t/deploying-on-aws-app-runner/85000>
- Fargate pricing: <https://aws.amazon.com/fargate/pricing/> (rates cross-checked via
  <https://cloudburn.io/blog/aws-fargate-pricing>, <https://fortem.dev/blog/aws-fargate-pricing-real-costs/>)
- Public IPv4 charge: <https://aws.amazon.com/blogs/aws/new-aws-public-ipv4-address-charge-public-ip-insights/>, <https://aws.amazon.com/vpc/pricing/>
- ALB pricing: <https://aws.amazon.com/elasticloadbalancing/pricing/> (cross-checked via
  <https://www.cloudzero.com/blog/aws-alb-pricing/>)
- Pinecone Starter tier: <https://www.pinecone.io/pricing/>, <https://docs.pinecone.io/reference/quotas-and-limits>
- Bedrock pricing: <https://aws.amazon.com/bedrock/pricing/> (cross-checked via
  <https://pecollective.com/tools/aws-bedrock-pricing/>)
- OpenSearch Serverless minimums: <https://aws.amazon.com/opensearch-service/pricing/> (cross-checked via
  <https://cloudburn.io/blog/amazon-opensearch-pricing>, <https://bigdataboutique.com/blog/opensearch-and-elasticsearch-pricing-guide>)
