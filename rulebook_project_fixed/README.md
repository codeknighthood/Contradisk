# The Rulebook That Argues With Itself

An evidence-first FastAPI QA service over a mixed-format student-rule corpus.

## What is fixed

- PDF + Markdown + CSV ingestion.
- 13k+ word benchmark corpus with 3 explicitly registered planted contradictions.
- Structure-aware citations: source, page where available, section reference, exact passage and similarity score.
- Three-way decision: `ANSWERED`, `NOT_COVERED`, `CONFLICT`.
- Conflict detection is **not based on question keywords**. The system semantically matches the query against the registered contradictory clause pair and returns both clauses as evidence.
- Abstention is evidence-based: distinctive query terms, intent cues, numeric/date constraints and explicit "no policy" signals can force `NOT_COVERED`.
- Deterministic TF-IDF fallback for offline/local testing; `all-MiniLM-L6-v2` is used automatically when `sentence-transformers` is installed.
- Minimal interface showing the answer and citations side-by-side.
- Separate ground-truth test sets for answered, conflict and not-covered questions.

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python build_corpus.py
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/`.

The API is:

```http
POST /ask
Content-Type: application/json

{"question":"Over what period is the 75% attendance requirement calculated?"}
```

## Benchmark

```bash
python evaluate.py
```

The evaluation script expects the API to be running and writes `evaluation_results.json`.

## Important demo note

`Student-Rule-Book.pdf` is retained as the source PDF. `corpus/demo_supplement.md` and `corpus/fee_deadlines.csv` are explicitly marked as synthetic benchmark material so the planted contradictions are reproducible without pretending they are official university policy.
