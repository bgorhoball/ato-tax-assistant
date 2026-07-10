"""
RAG evaluation test cases for ato-tax-assistant.

Source document: ATO Individual Tax Return Instructions 2024 (NAT 71050)
Vector store:    ChromaDB, chunk_size=1000, overlap=100
Known pages:     metadata page 52 = A2 Part-year threshold ("number of months"
                 sentence); metadata page 9 = "Are you an Australian resident?"
                 section.  Pages are PyPDFLoader 0-indexed PDF pages, NOT the
                 printed page numbers inside the document (TEST_SUMMARY.md's
                 "page 51/69" were printed-page citations).  Verified directly
                 against chroma_db contents on 2026-07-10.
Text quirk:      the PDF uses U+2011 non-breaking hyphens ("tax‑free", not
                 "tax-free") — keyword checks against SOURCE text must use
                 U+2011; checks against LLM answers are fine with ASCII.

Categories
----------
easy         Single-chunk direct lookup; high lexical overlap with question.
medium       Multi-chunk synthesis; semantic rather than lexical match.
hard         Implicit reasoning; relevant chunks may not surface in top-4.
hallucination Facts absent from the corpus; correct behaviour is to decline.
skip         Feature not yet implemented in the current basic version.

ground_truth_pages notes
------------------------
Pages marked `# approx` are estimated from ATO form structure knowledge and
should be verified against the actual PDF before treating recall scores as
authoritative.  Pages 52 (A2) and 9 (residency) are confirmed by direct
chroma_db content search (2026-07-10).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    """Single RAG evaluation test case."""

    id: str
    question: str
    category: str  # easy | medium | hard | hallucination | skip

    # Substrings that MUST appear in a correct answer (case-insensitive).
    # Accuracy = matched / total.  Not used when should_decline=True.
    expected_answer_contains: list[str] = field(default_factory=list)

    # Substrings that must NOT appear (fabricated facts).
    expected_answer_excludes: list[str] = field(default_factory=list)

    # 0-indexed page numbers (from doc.metadata["page"]) that the retriever
    # must surface.  Recall = hits / len(ground_truth_pages).
    ground_truth_pages: list[int] = field(default_factory=list)

    # True → the correct answer is "I don't know / not in document".
    should_decline: bool = False

    # Non-None → pytest.mark.skip equivalent; reason string explains what
    # needs to be built before this test can run.
    skip_reason: str | None = None

    notes: str = ""


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

EVAL_CASES: list[EvalCase] = [

    # =========================================================================
    # EASY  (10 cases)
    # Single chunk, high lexical overlap. Top-1 retrieval should hit target.
    # =========================================================================

    EvalCase(
        id="EASY-01",
        question="What does the A2 label on the individual tax return relate to?",
        category="easy",
        expected_answer_contains=["part-year", "tax-free", "threshold"],
        ground_truth_pages=[52],
        notes="'A2 Part‑year tax‑free threshold' appears verbatim at metadata page 52 (confirmed 2026-07-10).",
    ),
    EvalCase(
        id="EASY-02",
        question=(
            "What information must I write at label N when completing the "
            "A2 part-year tax-free threshold section?"
        ),
        category="easy",
        expected_answer_contains=["months", "resident", "australian"],
        ground_truth_pages=[52],
        notes=(
            "Metadata page 52 contains the 'number of months' instruction for label N "
            "(confirmed by chroma content search 2026-07-10)."
        ),
    ),
    EvalCase(
        id="EASY-03",
        question=(
            "Why does the ATO need information about whether I am an Australian "
            "resident for tax purposes?"
        ),
        category="easy",
        expected_answer_contains=["tax-free threshold"],
        ground_truth_pages=[9],
        notes=(
            "'Are you an Australian resident?' instructions section is at metadata "
            "page 9 (confirmed 2026-07-10; TEST_SUMMARY's 'page 69' was the printed "
            "page number, not the PDF index)."
        ),
    ),
    EvalCase(
        id="EASY-04",
        question="Which item on the individual tax return is used to report salary or wages?",
        category="easy",
        expected_answer_contains=["item 1", "salary"],
        ground_truth_pages=[],  # approx: early income section
        notes="Item 1 is 'Salary or wages' — standard ATO form structure.",
    ),
    EvalCase(
        id="EASY-05",
        question=(
            "Where on the individual tax return should allowances, earnings, "
            "tips, and directors fees be declared?"
        ),
        category="easy",
        expected_answer_contains=["item 2", "allowances"],
        ground_truth_pages=[],  # approx
        notes="Item 2 covers allowances, earnings, tips, directors fees.",
    ),
    EvalCase(
        id="EASY-06",
        question="Which item covers Employment Termination Payments on the tax return?",
        category="easy",
        expected_answer_contains=["item 4", "termination"],
        ground_truth_pages=[],  # approx
        notes="Item 4 is 'Employment termination payments'.",
    ),
    EvalCase(
        id="EASY-07",
        question="What type of income is reported at item 10 of the individual tax return?",
        category="easy",
        expected_answer_contains=["interest"],
        ground_truth_pages=[],  # approx
        notes="Item 10 is 'Gross interest'.",
    ),
    EvalCase(
        id="EASY-08",
        question="Which item on the tax return is used to declare dividend income?",
        category="easy",
        expected_answer_contains=["item 11", "dividend"],
        ground_truth_pages=[],  # approx
        notes="Item 11 is 'Dividends'.",
    ),
    EvalCase(
        id="EASY-09",
        question=(
            "Where should an individual report their share of income from a "
            "partnership or trust on the tax return?"
        ),
        category="easy",
        expected_answer_contains=["item 13", "partner"],
        ground_truth_pages=[],  # approx
        notes="Item 13 is 'Partnerships and trusts'.",
    ),
    EvalCase(
        id="EASY-10",
        question=(
            "What is the D9 deduction on the individual tax return used for?"
        ),
        category="easy",
        expected_answer_contains=["gifts", "donations"],
        ground_truth_pages=[],  # approx
        notes="D9 is 'Gifts or donations'.",
    ),

    # =========================================================================
    # MEDIUM  (12 cases)
    # Multi-chunk synthesis or semantic rather than keyword match.
    # Top-4 retrieval may not contain all required information.
    # =========================================================================

    EvalCase(
        id="MEDIUM-01",
        question=(
            "What are the two methods available for calculating work-related "
            "car expense deductions under item D1?"
        ),
        category="medium",
        expected_answer_contains=["cents per kilometre", "logbook"],
        ground_truth_pages=[],  # approx: D1 section
        notes="ATO D1 allows cents-per-km method or logbook method only.",
    ),
    EvalCase(
        id="MEDIUM-02",
        question=(
            "What conditions must be satisfied to claim a deduction for "
            "self-education expenses at D4?"
        ),
        category="medium",
        expected_answer_contains=["current", "employment", "income"],
        ground_truth_pages=[],  # approx: D4 section
        notes="D4: the course must maintain or improve skills required in current employment and lead to income.",
    ),
    EvalCase(
        id="MEDIUM-03",
        question=(
            "What are the requirements for a gift or donation to be claimable "
            "as a tax deduction at D9?"
        ),
        category="medium",
        expected_answer_contains=["deductible gift recipient", "$2"],
        ground_truth_pages=[],  # approx: D9 section
        notes="D9: gift must be to a DGR, amount $2 or more, no material benefit received in return.",
    ),
    EvalCase(
        id="MEDIUM-04",
        question=(
            "How is a net capital gain calculated for inclusion in the "
            "individual tax return at item 17?"
        ),
        category="medium",
        expected_answer_contains=["capital", "loss", "discount"],
        ground_truth_pages=[],  # approx: item 17 section
        notes="Item 17: subtract capital losses, then apply 50% CGT discount for assets held >12 months.",
    ),
    EvalCase(
        id="MEDIUM-05",
        question=(
            "What is the A1 adjustment for total net investment loss and "
            "why must it be completed?"
        ),
        category="medium",
        expected_answer_contains=["net investment loss", "A1"],
        ground_truth_pages=[],  # approx: adjustments section
        notes="A1 adds back net investment loss to income for Medicare levy / HECS repayment threshold calculations.",
    ),
    EvalCase(
        id="MEDIUM-06",
        question=(
            "What income is captured at item 21 for total reportable "
            "fringe benefits amounts?"
        ),
        category="medium",
        expected_answer_contains=["fringe benefits", "employer", "reportable"],
        ground_truth_pages=[],  # approx
        notes="Item 21: reportable fringe benefits from employer — affects MLS, HECS, and offset calculations.",
    ),
    EvalCase(
        id="MEDIUM-07",
        question="Where and how should foreign source income be reported on an individual tax return?",
        category="medium",
        expected_answer_contains=["item 20", "foreign"],
        ground_truth_pages=[],  # approx: item 20 section
        notes="Item 20 covers foreign source income and foreign assets or property.",
    ),
    EvalCase(
        id="MEDIUM-08",
        question=(
            "Under what conditions can an individual claim personal superannuation "
            "contributions as a tax deduction at D12?"
        ),
        category="medium",
        expected_answer_contains=["superannuation", "notice", "intent"],
        ground_truth_pages=[],  # approx: D12 section
        notes=(
            "D12: eligible if not an employee (or employed <10% of income from employment); "
            "must also lodge a notice of intent to claim with the super fund."
        ),
    ),
    EvalCase(
        id="MEDIUM-09",
        question=(
            "What is the purpose of the T5 item relating to private health insurance "
            "policy details?"
        ),
        category="medium",
        expected_answer_contains=["private health insurance", "T5"],
        ground_truth_pages=[],  # approx: T5 section
        notes="T5 is used to calculate the private health insurance tax offset or Medicare levy surcharge.",
    ),
    EvalCase(
        id="MEDIUM-10",
        question=(
            "What income amounts are counted toward the Medicare levy surcharge "
            "income threshold?"
        ),
        category="medium",
        expected_answer_contains=["income", "fringe benefits", "surcharge"],
        ground_truth_pages=[],  # approx: Medicare section
        notes="MLS income = taxable income + reportable fringe benefits + total net investment loss.",
    ),
    EvalCase(
        id="MEDIUM-11",
        question=(
            "How is the CGT discount applied differently to assets held for more "
            "than 12 months versus those held for less?"
        ),
        category="medium",
        expected_answer_contains=["50", "12 months", "discount"],
        ground_truth_pages=[],  # approx: item 17
        notes="50% CGT discount for individuals on assets held >12 months; no discount for <12 months.",
    ),
    EvalCase(
        id="MEDIUM-12",
        question=(
            "What types of expenses qualify for the D10 deduction for the "
            "cost of managing tax affairs?"
        ),
        category="medium",
        expected_answer_contains=["tax affairs", "D10"],
        ground_truth_pages=[],  # approx: D10 section
        notes="D10: tax agent fees, travel to tax agent, interest on tax debts, purchase of tax reference material.",
    ),

    # =========================================================================
    # HARD  (8 cases)
    # Implicit reasoning, multi-hop, or edge cases.
    # Retriever may surface adjacent but not precisely relevant chunks.
    # =========================================================================

    EvalCase(
        id="HARD-01",
        question=(
            "A person became an Australian tax resident in October 2023 and "
            "remained resident through 30 June 2024. How many months should "
            "they write at label N for the A2 part-year tax-free threshold?"
        ),
        category="hard",
        expected_answer_contains=["9", "months"],
        ground_truth_pages=[52],
        notes=(
            "October 2023–June 2024 = 9 months. Tests arithmetic over the A2 "
            "instructions (metadata page 52). Model must retrieve the rule AND apply it."
        ),
    ),
    EvalCase(
        id="HARD-02",
        question=(
            "What is the key distinction between personal services income at "
            "item 14 and business income at item 15, and why does it matter "
            "for deduction entitlements?"
        ),
        category="hard",
        expected_answer_contains=["personal services", "business"],
        ground_truth_pages=[],  # approx: items 14-15
        notes=(
            "PSI is income primarily from an individual's labour/skills; "
            "different deduction rules apply under the PSI regime."
        ),
    ),
    EvalCase(
        id="HARD-03",
        question=(
            "How are prior-year carried-forward capital losses treated when "
            "calculating the net capital gain at item 17?"
        ),
        category="hard",
        expected_answer_contains=["carried forward", "loss", "prior"],
        ground_truth_pages=[],  # approx: item 17
        notes="Prior-year capital losses offset current-year gains before the 50% discount is applied.",
    ),
    EvalCase(
        id="HARD-04",
        question=(
            "Can an individual claim a deduction for interest on money borrowed "
            "to purchase dividend-paying shares, and if so, under which item?"
        ),
        category="hard",
        expected_answer_contains=["interest", "D7", "dividend"],
        ground_truth_pages=[],  # approx: D7 section
        notes="D7 covers interest on borrowed funds used to earn assessable income including dividends.",
    ),
    EvalCase(
        id="HARD-05",
        question=(
            "What is attributed personal services income at item 9, and how "
            "does the ATO treat it differently from salary income?"
        ),
        category="hard",
        expected_answer_contains=["attributed", "personal services"],
        ground_truth_pages=[],  # approx: item 9
        notes="Item 9: PSI attributed from a PSE/PSB to the individual to prevent income splitting.",
    ),
    EvalCase(
        id="HARD-06",
        question=(
            "Under what circumstances must an individual complete the seniors "
            "and pensioners tax offset at item T7?"
        ),
        category="hard",
        expected_answer_contains=["seniors", "T7"],
        ground_truth_pages=[],  # approx: T7 section
        notes="T7: individuals of eligible age receiving Australian Government pension or meeting the income test.",
    ),
    EvalCase(
        id="HARD-07",
        question=(
            "How does the A1 net investment loss add-back affect the income "
            "figure used when assessing eligibility for the private health "
            "insurance Medicare levy surcharge?"
        ),
        category="hard",
        expected_answer_contains=["investment loss", "surcharge", "income"],
        ground_truth_pages=[],  # approx: A1 + Medicare sections
        notes="A1 loss is added back to taxable income to calculate MLS / PHI-offset income — cross-section reasoning.",
    ),
    EvalCase(
        id="HARD-08",
        question=(
            "If an individual uses their car for both work and private purposes, "
            "what proportion of car expenses can be claimed under D1, and which "
            "method determines that proportion?"
        ),
        category="hard",
        expected_answer_contains=["work", "logbook", "percentage"],
        ground_truth_pages=[],  # approx: D1
        notes="Logbook method determines the work-use percentage; cents-per-km is per actual work km only.",
    ),

    # =========================================================================
    # HALLUCINATION  (8 cases)
    # Facts absent from the corpus. Correct behaviour = explicit declination.
    # Measured by presence of decline-signal phrases in the answer.
    # =========================================================================

    EvalCase(
        id="HALL-01",
        question=(
            "What is the exact tax-free threshold dollar amount for Australian "
            "residents in the 2023-24 financial year?"
        ),
        category="hallucination",
        should_decline=True,
        expected_answer_excludes=["18,200", "18200"],
        notes=(
            "The ATO Individual Tax Return Instructions does not state the $18,200 "
            "figure — it only describes how to claim it. A grounded system should "
            "acknowledge the amount is not in the document."
        ),
    ),
    EvalCase(
        id="HALL-02",
        question="What is the Medicare levy rate expressed as a percentage of taxable income?",
        category="hallucination",
        should_decline=True,
        expected_answer_excludes=["2%", "2 percent", "two percent"],
        notes=(
            "The instructions explain exemption conditions but may not state the 2% rate "
            "as a number. If it is absent, the model must decline rather than recall from "
            "training data."
        ),
    ),
    EvalCase(
        id="HALL-03",
        question=(
            "What is the company tax rate for small businesses with an aggregated "
            "turnover below $50 million?"
        ),
        category="hallucination",
        should_decline=True,
        notes="Company tax rates are completely outside the scope of individual tax return instructions.",
    ),
    EvalCase(
        id="HALL-04",
        question=(
            "What superannuation guarantee percentage must employers pay for "
            "the 2023-24 financial year?"
        ),
        category="hallucination",
        should_decline=True,
        expected_answer_excludes=["11%", "11 percent"],
        notes="SG rate is not covered in individual tax return instructions.",
    ),
    EvalCase(
        id="HALL-05",
        question="Who is currently the Commissioner of Taxation at the ATO?",
        category="hallucination",
        should_decline=True,
        notes="The Commissioner's name does not appear in the instructions document.",
    ),
    EvalCase(
        id="HALL-06",
        question=(
            "What penalty does the ATO impose for late lodgement of an individual "
            "tax return, and how is it calculated?"
        ),
        category="hallucination",
        should_decline=True,
        notes="Penalty rates for late lodgement are not covered in the individual tax return instructions.",
    ),
    EvalCase(
        id="HALL-07",
        question="What is the GST rate in Australia and how does it apply to deductible business expenses?",
        category="hallucination",
        should_decline=True,
        notes="GST is entirely outside the scope of individual income tax return instructions.",
    ),
    EvalCase(
        id="HALL-08",
        question=(
            "What is the concessional superannuation contribution cap for "
            "the 2023-24 financial year?"
        ),
        category="hallucination",
        should_decline=True,
        expected_answer_excludes=["27,500", "30,000", "$27", "$30"],
        notes=(
            "Concessional contribution caps are not stated in individual tax return instructions. "
            "Model should decline rather than state a figure from training memory."
        ),
    ),

    # =========================================================================
    # SKIP  (5 cases)
    # Tests for features not yet built in the current basic version.
    # Mark as skip so the runner reports them without failing the suite.
    # =========================================================================

    EvalCase(
        id="SKIP-01",
        question=(
            "Quote verbatim the exact sentence from the document that explains "
            "what to write at label N for the A2 part-year tax-free threshold."
        ),
        category="skip",
        skip_reason=(
            "LAYER_3: Enforced grounding with deterministic verbatim citation not implemented. "
            "Target architecture returns the source sentence alongside the answer; "
            "current engine only passes page number in context."
        ),
        notes=(
            "When Layer 3 (enforced grounding) is built, this case validates that the "
            "citation contains the verbatim text from page 51."
        ),
    ),
    EvalCase(
        id="SKIP-02",
        question=(
            "Earlier you told me salary goes in item 1. "
            "Where should I declare overtime pay that appears as a separate "
            "line on my income statement?"
        ),
        category="skip",
        skip_reason=(
            "NOT_IMPLEMENTED: Multi-turn conversation with engine-side context retention. "
            "Streamlit renders chat history in the UI but does not pass prior turns back "
            "to the LLM. The engine has no memory between ask() calls."
        ),
        notes=(
            "Validates that the model can reference prior turns to avoid redundant advice. "
            "Requires conversation history to be injected into the prompt."
        ),
    ),
    EvalCase(
        id="SKIP-03",
        question=(
            "My tax file number is 123 456 789. "
            "What deductions am I eligible for based on my situation?"
        ),
        category="skip",
        skip_reason=(
            "LAYER_5: PII detection and masking not implemented. "
            "A production system must detect TFN/ABN patterns in user input, refuse to process "
            "them, and alert the user — this is Layer 5 of the project roadmap."
        ),
        notes=(
            "This test validates the guardrail layer. "
            "Expected behaviour: system detects the TFN, refuses to echo it, "
            "and prompts the user not to share sensitive identifiers."
        ),
    ),
    EvalCase(
        id="SKIP-04",
        question=(
            "[PERF] Compare response latency for the same question via the sync ask() "
            "path and the async aask() path at k=4."
        ),
        category="skip",
        skip_reason=(
            "NOT_IMPLEMENTED: TaxRagEngineV2.aask() exists but the Streamlit UI still "
            "calls the sync ask() path. No latency benchmarking harness built yet. "
            "Add once aask() is wired into the UI."
        ),
        notes=(
            "When integrated, a 20-case latency benchmark comparing sync vs async "
            "should show a measurable improvement under concurrent load."
        ),
    ),
    EvalCase(
        id="SKIP-05",
        question=(
            "Answer this question using the AWS Bedrock Nova Lite backend instead of Gemini."
        ),
        category="skip",
        skip_reason=(
            "PROD_INFRA: Bedrock backend not yet deployed. ECS Fargate Spot + Bedrock task role "
            "is designed in infra/terraform/ but terraform apply has not been run. "
            "Re-run the full eval suite against Bedrock once prod is live to compare "
            "accuracy vs the Gemini dev baseline."
        ),
        notes=(
            "Cross-backend regression: expect answer_accuracy within ±0.05 of the Gemini baseline "
            "if Nova Lite quality is comparable. Flag larger gaps for prompt tuning."
        ),
    ),
]
