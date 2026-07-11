# RAG Eval Design — ato-tax-assistant

## What this eval covers

The eval framework measures how well the **current basic RAG pipeline**
(ChromaDB + Gemini 2.5 Flash, `k=4`) answers questions about the
ATO Individual Tax Return Instructions 2024 (NAT 71050).

It is designed against the **target architecture** described in the project
roadmap, not just the current state. Tests for features not yet built are
present but marked `skip` so the suite is executable now and remains a
living contract for future layers.

---

## File layout

```
eval/
├── cases.py        43 EvalCase definitions
├── metrics.py      4 metric functions (pure, no LLM, deterministic)
└── run_eval.py     CLI runner, prints table, optionally writes JSON

EVAL_DESIGN.md      this file
```

---

## Running the eval

```bash
# Prerequisites: activate venv, ensure GOOGLE_API_KEY is set and chroma_db/ exists
source venv/bin/activate

# Full run (38 cases; 5 skipped automatically)
python eval/run_eval.py

# Single category
python eval/run_eval.py --category hallucination

# Single case by ID
python eval/run_eval.py --id HARD-01

# Retrieve 6 chunks instead of 4
python eval/run_eval.py --k 6

# Preview case list without calling the engine
python eval/run_eval.py --dry-run

# Retrieval recall only — no LLM calls, so it doesn't touch the
# 20-requests/day gemini-2.5-flash generate quota (uses embed quota, 100/min)
python eval/run_eval.py --retrieval-only

# Save results for tracking over time
python eval/run_eval.py --output eval/results/$(date +%Y%m%d).json

# Print full answers
python eval/run_eval.py --verbose
```

Exit code is `0` if every evaluable case reaches `answer_accuracy ≥ 0.5`
(adjustable with `--pass-threshold`).

---

## Test case taxonomy

### Category rationale

| Category | Count | Purpose |
|---|---|---|
| `easy` | 10 | Baseline: can the retriever find the right chunk at all? High lexical overlap between question and source text. Top-1 retrieval should hit the target. |
| `medium` | 14 | Core: multi-chunk synthesis or semantic rather than keyword match. Tests whether `k=4` surfaces enough context and the LLM connects it correctly. (Includes HALL-02/HALL-06, reclassified from hallucination on 2026-07-11 — their facts turned out to be in the document.) |
| `hard` | 8 | Stress: implicit reasoning, multi-hop, or arithmetic over retrieved facts. Retrieval may partially miss; tests LLM's ability to reason with imperfect context. |
| `hallucination` | 6 | Safety: facts absent from the corpus. The model should explicitly decline rather than recall from training memory. This is the highest-stakes failure mode for a tax assistant. |
| `skip` | 5 | Roadmap: tests for Layer 3 (enforced citation), Layer 5 (PII), prod infra, and multi-turn. Present as a contract — activate each test when the feature is built. |

### Why a difficulty gradient?

A single accuracy number over a flat test set is misleading: a RAG system can
score well on easy questions while completely failing on the harder synthesis
and hallucination cases that matter most in production.

The gradient lets you diagnose *where* the pipeline breaks:

- **Low easy accuracy** → retrieval is broken (embedding / chunking issue).
- **Good easy, low medium** → retrieval works but context window or `k` is too small.
- **Good medium, low hard** → LLM reasoning over retrieved context is weak.
- **Low hallucination accuracy** → system leaks training-data facts when it should decline; re-tune the system prompt grounding instruction.

### Ground truth page numbers

Pages are 0-indexed PDF pages from `PyPDFLoader` (metadata `"page"` field).
**These are not the printed page numbers inside the document** — the document's
own "page 51/69" references correspond to metadata pages 52/9 in the current store.

- **Page 52** (A2 Part-year threshold, "number of months" sentence) and
  **page 9** ("Are you an Australian resident?" section) are confirmed by
  direct chroma_db content search (2026-07-10).
- The PDF text uses **U+2011 non-breaking hyphens** (`tax‑free`, `Part‑year`).
  Keyword checks against *source chunk text* must use U+2011; checks against
  *LLM answers* are fine with ASCII hyphens.
- All other `ground_truth_pages` values are empty (`[]`) because the full PDF
  was not re-read at eval design time. Before trusting retrieval recall scores
  on medium/hard cases, verify the correct page numbers against the PDF and
  fill in `ground_truth_pages`.

---

## Metric definitions

### 1. Answer Accuracy

**What it measures:** Is the answer factually correct and complete?

**Method:** Keyword-match proxy. `expected_answer_contains` lists the minimum
set of concepts that must appear in a correct answer (case-insensitive
substring match). Score = `matched_keywords / total_keywords`.

```
score = 1.0   → all expected keywords found  (full credit)
score = 0.5   → half the keywords found      (partial)
score = 0.0   → fewer than half found        (fail)
```

**For `should_decline` cases:** Score = `1.0` if the answer contains any
decline-signal phrase (`"not stated"`, `"cannot find"`, `"the context does not"`,
etc.); `0.0` otherwise. This tests whether the system avoids confabulation.

**Why not an LLM judge?**
An LLM judge (e.g., GPT-4 grading the answer) is more nuanced but adds cost
($0.01–0.05 per case), non-determinism, and a hidden dependency on a second
model's behaviour. Keyword matching is free, deterministic, and reproducible
across runs — essential when tracking regressions over weeks of development.
The tradeoff is lower sensitivity to paraphrase: an answer that is correct but
uses different vocabulary scores 0 until `expected_answer_contains` is updated.
Add an LLM judge as an optional `--llm-judge` flag once the pipeline is stable.

### 2. Retrieval Recall

**What it measures:** Did the retriever surface the relevant source chunk(s)?

**Formula:** `hits / len(ground_truth_pages)` where a hit is any retrieved
document whose `metadata["page"]` matches a ground-truth page.

**Returns `None`** when `ground_truth_pages` is empty (most medium/hard cases
until page numbers are verified). `None` values are excluded from category means.

**Why this metric?**
Retrieval recall is independent of the LLM — it measures the vector search
quality. A failing recall score with passing accuracy means the LLM is answering
from hallucinated knowledge rather than retrieved context. A passing recall with
failing accuracy points to prompt or LLM issues. Separating them is essential
for targeted debugging.

### 3. Citation Faithfulness

**What it measures:** Are the page numbers cited in the answer actually in the
retrieved document set?

**Method (current):** Regex extraction of page references from the answer text
(`"Page 51"`, `"p. 51"`, etc.). Score = `cited_pages_in_retrieved / total_cited_pages`.
Returns `None` if the answer cites no pages.

**Why the current system prompt matters:** The system prompt says
`"Always cite relevant sections or page numbers when possible."` This encourages
citations so faithfulness is evaluable. Without the instruction most answers
would return `None` here.

**Limitation:** A model can cite page 51 faithfully (the page was retrieved) but
still fabricate the content of page 51. True citation faithfulness requires
verbatim quote matching against the source chunk text — this is a planned
**Layer 3** feature. The regex approach here is a shallow but useful proxy.

### 4. Hallucination Penalty (supplementary)

**What it measures:** Does the answer contain a forbidden string (fabricated
specific fact)?

**Method:** `expected_answer_excludes` lists strings that indicate fabrication
(e.g., `"18,200"` for a question whose answer is not in the document).
Score = `1.0` if none appear, `0.0` if any appear. `None` if list is empty.

This metric is complementary to answer accuracy on `hallucination` cases:
- Accuracy catches failure to decline.
- Hallucination penalty catches fabricated specifics even in partial declines
  (e.g., "I'm not sure, but the threshold might be $18,200").

---

## Skipped tests and what they represent

| ID | Feature | Roadmap layer |
|---|---|---|
| SKIP-01 | Verbatim citation extraction | Layer 3: enforced grounding |
| SKIP-02 | Multi-turn conversation context | Not yet scheduled |
| SKIP-03 | PII detection (TFN in input) | Layer 5: PII masking |
| SKIP-04 | Async path latency benchmark | After `aask()` wired to UI |
| SKIP-05 | Bedrock Nova Lite backend | Layer 4: prod infra |

When you implement a feature, remove its `skip_reason` and update
`expected_answer_contains` / `ground_truth_pages` as needed. The test
immediately enters the run suite on the next eval pass.

---

## Findings from the first baseline run (2026-07-10)

The first full run (`eval/results/2026-07-10-gemini-baseline.json`) surfaced
three real defects in the app — before any manual testing had noticed them:

1. **Embedding model mismatch (critical) — FIXED on this branch.** The
   chroma_db store was ingested with `text-embedding-004` (see
   TEST_SUMMARY.md, Feb 2026), but `rag_engine.py` was later switched to
   `gemini-embedding-001` for queries. Both output 768 dims, so nothing
   errors — but cosine similarity between a stored vector and a fresh
   embedding of the *same text* was **−0.007**. Retrieval was effectively
   random (the A2 question retrieved Medicare-levy pages); retrieval recall
   was 0.000 across all verified cases because of this.
   Fix applied 2026-07-10: chroma_db re-ingested with `gemini-embedding-001`.

2. **Triple ingestion — FIXED on this branch.** The store held 867 chunks =
   3 × 289; top-4 retrieval often returned the same (wrong) chunk multiple
   times. Both engines now call `_reset_chroma_collection()` before
   ingesting, so re-ingestion replaces rather than appends.

3. **Free-tier quotas (constraint, not a bug).**
   - `gemini-2.5-flash` generate: **20 requests/day** — the 38-case suite
     cannot complete in one day (17 cases hit 429). Run per-category across
     days (`--category easy` ≈ 10 requests), use `--retrieval-only`
     (embedding calls only) for retrieval regression checks, or use a paid key.
   - `gemini-embedding-001` embed: **100 requests/min**, and each chunk is
     one request — ingestion needs ≥60s between 50-chunk batches
     (`ingest_pdf(..., delay=60)`).

The baseline JSON (`2026-07-10-gemini-baseline.json`) records the *broken*
pre-fix state — keep it as the "before" snapshot.

### Post-fix results (2026-07-11/12)

| Metric | Pre-fix | Post-fix |
|---|---|---|
| easy accuracy | 0.278 | **0.900** |
| medium accuracy | 0.517 | **0.859** (13/14; MEDIUM-04 hit transient 429) |
| hard accuracy | 0.750 | **1.000** (2/8 so far; rest pending quota) |
| retrieval recall | 0.000 | **1.000** (all evaluable cases) |
| citation faithfulness | 1.000* | 1.000 |
| hallucination (decline) accuracy | n/a (429s) | **1.000** (6/6 true cases) |

\* pre-fix faithfulness was vacuously high — the model cited retrieved (wrong)
pages while declining.

Two original hallucination cases (HALL-02 Medicare levy 2%, HALL-06 late
lodgement penalty) were **reclassified to medium** after the run proved their
facts ARE in the document (pages 41 and 79) — the system answered both
grounded with correct citations. Both scored 1.000 as medium cases on
2026-07-12. Similarly, MEDIUM-08 was **recalibrated** — the corpus does not
contain D12 eligibility conditions (the document delegates to ato.gov.au),
so the system's honest partial answer was correct, not a failure. This is
expected eval-development iteration: the runs validate the system AND
correct the test set.

Still pending (one quota day): MEDIUM-04 re-run + HARD-03 through HARD-08
(7 generate calls). Note that failed requests also count against the 20/day
quota, and the sequential runner can trip the 10 requests/min limit
(MEDIUM-04's transient 429) — budget headroom accordingly.

---

## Interpreting results

### Baseline expectations for the current basic version

| Metric | Expected range | Notes |
|---|---|---|
| easy accuracy | 0.70 – 1.00 | Lower than 0.70 suggests retrieval or chunking is broken |
| medium accuracy | 0.50 – 0.80 | High variance; sensitive to `k` value |
| hard accuracy | 0.30 – 0.65 | Multi-hop is hard; use to set a floor, not a target |
| hallucination accuracy | 0.40 – 0.80 | The current system prompt is not strongly grounded; expect some failures |
| retrieval recall (easy/hard-01) | 0.70 – 1.00 | Only 3 cases have verified ground_truth_pages |
| citation faithfulness | 0.50 – 1.00 | Depends on whether LLM chooses to cite pages |

### What to do with low scores

**Low hallucination accuracy (most urgent):**
Update the system prompt to add an explicit refusal instruction:
```
If the answer is not present in the provided context, say exactly:
"I cannot find this information in the ATO document provided."
Do not use your general knowledge to fill gaps.
```

**Low easy accuracy:**
Check that `chroma_db/` is populated (run `src/rag_engine.py` to ingest),
and that the embedding model is the same one used during ingestion.

**Low medium accuracy with good easy:**
Increase `k` (try `--k 6` or `--k 8`). Each medium case may require 2-3
chunks from different pages.

**Low citation faithfulness:**
The LLM is citing pages that were not retrieved. This often happens when
the model confabulates page references from training data. Strengthen
the grounding instruction and consider adding `"Only cite pages from the context."`.

---

## Upgrade path

1. **Fill in `ground_truth_pages`** for medium/hard cases by reading the PDF.
2. **Add an LLM judge** as an optional `--llm-judge` flag for nuanced accuracy
   scoring (use Haiku to keep cost low).
3. **Activate SKIP-01** when Layer 3 (enforced grounding) is built; replace
   regex citation check with verbatim quote matching.
4. **Add cross-backend regression** via `--backend bedrock` once Layer 4 is
   deployed; compare against the Gemini baseline stored in `eval/results/`.
5. **Activate SKIP-03** (PII) when Layer 5 guardrails are built; the test
   validates that TFN patterns in input are refused, not echoed.
