# Outstanding Work — Execution Plan

Handoff document for continuing work on `ato-tax-assistant` and the RegTech
RAG project. Written to be executed step-by-step without prior context.
Last updated: 2026-07-12. Current branch: `feature/rag-eval-framework`.

---

## Rules — read before doing anything

1. **API quotas (Gemini free tier). Exceeding them breaks runs silently-ish
   (429 errors recorded per case, suite continues).**
   - `gemini-2.5-flash` generate: **20 requests/day**, **10 requests/min**.
     Every eval case = 1 generate request. **Failed requests also count.**
   - `gemini-embedding-001`: **100 requests/min**, 1 request per chunk.
     Ingestion must use `delay=60` between 50-chunk batches.
   - Budget ≤ 18 generate calls per day to leave headroom. Insert
     `sleep 90` between consecutive eval invocations.

2. **Do NOT re-ingest `chroma_db/` unless it is provably broken.** It was
   fixed on 2026-07-10 (289 chunks, `gemini-embedding-001`). To verify health
   WITHOUT re-ingesting, run the health check in Appendix A. Healthy =
   count 289 and cosine ≥ 0.85.

3. **Do NOT push or merge any branch without Brian's explicit OK.**
   The eval branch is intentionally local-only.

4. **`master` and `feature/aws-deployment` still contain the BROKEN
   chroma_db.** Never deploy from them. `feature/rag-eval-framework` must be
   merged first (Brian decides when).

5. **RegTech repo learning rule (applies to the future
   `regtech-compliance-rag` repo, NOT to ato-tax-assistant):** Brian writes
   the learning-critical code himself (ingestion, retrieval, grounding —
   Layers 1–3). AI may scaffold, tutor, and review only. Do not write those
   implementations for him, even if asked casually — remind him of the rule.

6. **Document quirks (will waste hours if unknown):**
   - `doc.metadata["page"]` is the 0-indexed PDF page, NOT the page number
     printed inside the document (printed p51 = metadata p52).
   - The PDF text uses U+2011 non-breaking hyphens: `tax‑free`, not
     `tax-free`. String searches against chunk text must use U+2011.
     LLM answers use normal ASCII hyphens.

7. **Untracked files `ROADMAP.md` and `specs/` belong to Brian's other
   session.** Do not delete, rewrite, or commit them without asking.

8. When a test case fails, **inspect the answer before blaming the app**
   (see Appendix B). Three eval cases so far (HALL-02, HALL-06, MEDIUM-08)
   turned out to be mis-designed cases, not app bugs — the fix was to
   correct the case, not the code.

---

## Task 1 — Finish the post-fix eval baseline (7 generate calls)

**Blocked until:** the daily generate quota resets (next calendar day after
2026-07-12, Google's reset time).

```bash
cd /home/brian/ai-workspace/ato-tax-assistant
./venv/bin/python eval/run_eval.py \
  --id "MEDIUM-04,HARD-03,HARD-04,HARD-05,HARD-06,HARD-07,HARD-08" \
  --output eval/results/$(date +%Y-%m-%d)-remaining.json
```

**Expected:** 7 cases run, 0 errors. Accuracy per case mostly ≥ 0.5 (hard
cases may legitimately score lower). Recall column will show N/A for most
(no verified ground-truth pages yet) — that is normal.

**If a case errors with 429:** wait until the next day and re-run only the
failed IDs (comma-separated `--id`).

**If a case scores below 0.5:** follow Appendix B before changing anything.

**Then:**
1. Update the "Post-fix results" table in `EVAL_DESIGN.md` with the final
   medium (14/14) and hard (8/8) numbers. Compute medium's final average
   across BOTH result files (2026-07-12-medium-postfix.json has 13 cases,
   the new file has MEDIUM-04).
2. Remove the "Still pending" paragraph from the findings section in
   `EVAL_DESIGN.md`.
3. Commit to `feature/rag-eval-framework`:
   ```
   git add EVAL_DESIGN.md eval/results/
   git commit -m "Complete post-fix eval baseline (all 38 cases)"
   ```
4. Tell Brian the branch is ready to push/merge. Do not push it yourself.

---

## Task 2 — Merge before deploy (Brian's decision, assist only)

When Brian says merge:

```bash
git checkout master
git merge feature/rag-eval-framework
```

There may be conflicts in `chroma_db/` binaries against
`feature/aws-deployment` — the eval branch's chroma_db is the CORRECT one;
resolve conflicts by taking the eval branch's version (`git checkout
--theirs` or `--ours` depending on merge direction — verify with the health
check in Appendix A afterwards, count must be 289).

---

## Task 3 — Implement specs 01→04 (in `specs/`, in order)

Four improvement specs exist: `01-chunking.md`, `02-hybrid-retrieval.md`,
`03-citation-faithfulness.md`, `04-doc-update-pipeline.md`. Read each spec
file fully before starting it. General protocol for each spec:

1. **Before:** run `./venv/bin/python eval/run_eval.py --retrieval-only
   --output eval/results/<date>-before-spec<NN>.json` (free — no generate
   quota) and, if quota allows, one accuracy category.
2. Implement per the spec.
3. **If the spec changes chunking or embeddings (spec 01 does):** re-ingest
   is required — use `ingest_pdf(pdf_path, batch_size=50, delay=60)` and
   re-verify with Appendix A. Chunk count will change from 289; update the
   expected count in this file and in eval ground-truth notes if pages shift.
4. **After:** re-run the same eval commands. Compare recall/accuracy.
   A spec that lowers recall is a regression — stop and report to Brian.
5. Commit results JSON alongside the implementation.

Spec-specific notes:
- **Spec 02 (hybrid retrieval):** `--retrieval-only` is the primary metric.
  Recall is currently 1.000 on verified cases, so the win to look for is on
  medium/hard cases once their `ground_truth_pages` are filled in (see
  Task 4) — do Task 4 for at least the medium cases BEFORE judging spec 02.
- **Spec 03 (citation faithfulness):** when implemented, activate eval case
  SKIP-01: in `eval/cases.py` set its `skip_reason=None`, category to
  `"hard"`, and add expected keywords matching the verbatim page-52 sentence.
  Also upgrade `compute_citation_faithfulness()` in `eval/metrics.py` from
  regex page-matching to verbatim quote lookup (docstring documents this
  upgrade path).

---

## Task 4 — Fill in eval ground-truth pages (no API quota needed)

Most medium/hard cases have `ground_truth_pages=[]`, so retrieval recall is
not measured for them. To fill them in, search the chroma store directly
(no embedding calls — `collection.get(where_document=...)` is a local
sqlite text search):

```bash
./venv/bin/python - <<'EOF'
import chromadb
c = chromadb.PersistentClient(path='./chroma_db').get_collection('langchain')
# Example: find the D1 car-expenses section. Remember U+2011 hyphens!
hits = c.get(where_document={'$contains': 'cents per kilometre'},
             include=['metadatas'], limit=10)
print(sorted({m.get('page') for m in hits['metadatas']}))
EOF
```

For each case in `eval/cases.py` marked `# approx`: find a distinctive
phrase from the case's `notes`, locate its page(s), set
`ground_truth_pages`, and remove the `# approx` comment. If the phrase does
not exist in the corpus at all, the case may need recalibration like
MEDIUM-08 (see Appendix B). Commit in batches with a note of which pages
were verified.

---

## Task 5 — RegTech repo Phase 0 (only when Brian says start)

Mechanical scaffold (allowed): create `~/ai-workspace/regtech-compliance-rag`,
`git init`, copy `ARCHITECTURE.md` + `DEPLOY_BRIEF.md` from ato-tax-assistant,
create directory skeleton, download 2–4 public HKMA SPM / SFC PDFs, port
`eval/` framework structure (cases will need rewriting for the new corpus).

NOT allowed (Rule 5): writing the ingestion/retrieval/grounding
implementations. Brian writes those; review and tutor only.

---

## Backlog (do only if asked)

- Optional `--llm-judge` flag for run_eval.py (use a cheap model; keyword
  matching stays the default for determinism).
- Latency benchmark once `aask()` is wired into the Streamlit UI
  (activates SKIP-04).
- PII guardrail layer (activates SKIP-03) — Layer 5, month 3.
- Bedrock cross-backend regression (activates SKIP-05) — after Layer 4 deploy.

---

## Appendix A — chroma_db health check (no API calls except 1 embed)

```bash
cd /home/brian/ai-workspace/ato-tax-assistant
./venv/bin/python - <<'EOF'
import chromadb, numpy as np
c = chromadb.PersistentClient(path='./chroma_db').get_collection('langchain')
print('count:', c.count())          # expect 289
one = c.get(where_document={'$contains': 'number of months'},
            include=['documents','embeddings'], limit=5)
print('copies of label-N chunk:', len(one['ids']))   # expect 1
from dotenv import load_dotenv; load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
emb = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',
                                   output_dimensionality=768)
fresh = np.array(emb.embed_query(one['documents'][0]))
stored = np.array(one['embeddings'][0])
cos = stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh))
print(f'cosine: {cos:.3f}')         # expect >= 0.85 (0.91 typical)
EOF
```

If count ≠ 289 or cosine < 0.5: the store is broken. Re-ingest:
`TaxRagEngine(...).ingest_pdf('./data/tax_guide.pdf', batch_size=50, delay=60)`
(idempotent — the engine resets the collection first). Takes ~6 minutes.

## Appendix B — triage protocol for a failing eval case

A low score means ONE of: (a) app bug, (b) mis-designed test case. Decide
which before changing anything:

1. Read the recorded answer and `retrieved_pages` in the results JSON.
2. If the answer says "not in the document": check whether the fact really
   is absent — search chunk text via `collection.get(where_document=
   {'$contains': ...})` (mind U+2011 hyphens). Absent → the app is RIGHT;
   recalibrate the case (precedent: MEDIUM-08). Present → retrieval or
   prompting bug; report to Brian with the retrieved pages.
3. If the answer states facts: check they exist in the retrieved chunks.
   In-corpus facts answered correctly with citations on a `should_decline`
   case → the case was wrong; reclassify (precedent: HALL-02, HALL-06).
   Facts NOT in any retrieved chunk → genuine hallucination; report it.
4. Every reclassification: update the case's `notes` with date + evidence,
   update category counts in `EVAL_DESIGN.md`, commit with explanation.

## Appendix C — key file map

| Path | What it is |
|---|---|
| `src/rag_engine.py` | Engine used by app + eval (sync) |
| `src/rag_engine_v2.py` | Async variant, not wired to UI yet |
| `eval/cases.py` | 43 test cases (10 easy / 14 medium / 8 hard / 6 hallucination / 5 skip) |
| `eval/metrics.py` | Pure metric functions |
| `eval/run_eval.py` | Runner: `--category`, `--id a,b,c`, `--retrieval-only`, `--dry-run`, `--output` |
| `eval/results/` | One JSON per run; pre-fix baseline = 2026-07-10-gemini-baseline.json |
| `EVAL_DESIGN.md` | Design rationale + findings + results tables |
| `specs/01..04-*.md` | Improvement specs (Brian's, untracked) |
| `infra/terraform/` | AWS stack, validated but never applied |
