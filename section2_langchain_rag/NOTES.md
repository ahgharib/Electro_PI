# Section 2 — Notes

## Document choice

Real, unmodified, government-published PDFs — not authored, not
summarized, not paraphrased. This is the "or you may use your own domain
docs — state your choice" option, taken literally: the actual original
publications, byte-for-byte. 3 files in `docs/`, all PDF:

| File | Real source | Verify at |
|---|---|---|
| `01_nist_800-145_cloud_computing.pdf` | NIST Special Publication 800-145, "The NIST Definition of Cloud Computing" (Mell & Grance, Sept. 2011) | [nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf) |
| `02_nasa_our_solar_system.pdf` | NASA "Our Solar System" lithograph (JPL 400-1489B, July 2013) | [solarsystem.nasa.gov] |
| `03_cdc_flu_vaccine_vis.pdf` | CDC Vaccine Information Statement, Inactivated/Recombinant Influenza Vaccine (rev. 1/31/2025) | [immunize.org/vis](https://www.immunize.org/vis/) |

---

## Write-up: chunking / retrieval on longer documents

**Current strategy.** Structure-aware recursive splitting: markdown is
split on headers first (coarse, structural boundaries), then each section
is recursively split by paragraph → sentence → character down to ~600
tokens with ~80 tokens (13%) of overlap. PDFs lack reliable structural
markers, so they skip straight to the same recursive splitter, with each
chunk mapped back to its source page via character-offset tracking.

**Trade-offs / limitations.** Overlap improves boundary recall (a sentence
split across a chunk boundary isn't orphaned) but increases storage and can
surface near-duplicate chunks in the same top-k, wasting context budget.
Header-splitting only gives coarse section boundaries — it can still cut a
table, a multi-step procedure, or a long explanation in half. Mechanically,
this strategy has no problem with long documents or many documents — it
just produces more chunks, and chunk metadata keeps per-document provenance
regardless of how many source files are indexed. What it does **not**
handle well is retrieval *quality* on long or numerous documents: flat
dense top-k search over hundreds/thousands of similarly-worded chunks
starts missing the genuinely relevant one, and a single global relevance
threshold gets harder to tune as the score distribution across many
documents gets noisier.

**What I'd change if answer quality on longer documents were poor**, in
priority order:

1. **Hybrid search (dense + BM25/keyword), fused with something likereciprocal rank fusion. or Reciprocal Rank Fusion(RRF).**
   Dense embeddings blur on exact terms — model
   numbers, order-policy numbers, error codes — that keyword search catches
   precisely. This is usually the highest-leverage single change and is
   why the `Retriever` is already structured so a keyword-search stage can
   be inserted as an additional step without touching the rest of the
   pipeline.
2. **Re-ranking.** Retrieve a wider candidate set (top 20–30) cheaply with
   the bi-encoder, then re-score with a cross-encoder before keeping the
   top-k that actually goes to the LLM. This recovers precision at the top
   without needing k to be small at retrieval time, which matters more as
   corpus size grows.
3. **Hierarchical/parent-child indexing.** Embed and search over
   short, precise child chunks, but pass the containing (larger) parent
   section to the LLM as context. This directly targets the "cut a table
   in half" failure mode without needing bigger chunks at index time
   (which would hurt retrieval precision).
4. **Smaller, semantically-bounded chunks with query-time expansion**
   instead of large fixed-size chunks — retrieve tight, precise units, and
   let neighbor-expansion (already implemented, config-flagged, see below)
   pull in surrounding context only for chunks that actually scored well.
5. **Neighbor-chunk expansion** (pulling in the adjacent chunk, same source,
   index ±1, around each retrieved chunk) is implemented and config-flagged
   (`ENABLE_NEIGHBOR_EXPANSION`, off by default) rather than always-on, since
   it trades additional token cost for recovered boundary context and isn't
   free — neighbors are marked distinctly and don't count toward the relevance
   gate or get auto-cited unless the model actually cites them.
6. **Small LLM before retrival** Using a fast and small LLM to make a desicion wither to return "I don't know"
   or pass the questions to the pipeline, this step is a good practice when using a larger and more time and money consuming LLM **(Used Depending on the Type and Domain of the Documents)**.
7. **Small LLM for query tunning** for better retrival quality It is best to use an extra LLM for tunning the 
   User query to a more RAG friendly query **(Used Depending on the Type and Domain of the Documents)**.

---

## Example questions and actual answers

5 questions, one of each required case, run against the 3 real PDFs above
— see `scripts/eval.py` for the full reasoning behind trimming this down
from a larger sweep.

- **Easy** (NIST doc) — single direct fact.
- **Medium** (CDC doc) — a specific fact plus its named reporting process.
- **Hard** (NASA doc) — a *grounding* test, not a synthesis test
- **Edge case, below threshold** — genuinely unrelated to all 3 documents
  (a truck's towing capacity). Should never reach the LLM at all.
- **Edge case, passes but should self-refuse** — a compound question
  splicing real vocabulary from two different documents together
  ("moons," which is all over the NASA doc, and "the cloud," which is all
  over the NIST doc). Verified empirically that this
  *does* clear the relevance threshold — so unlike the previous edge case,
  this one is a test of the model's own grounding instruction ("say so if
  the context doesn't answer the question"), not of retrieval precision.
  Kept in deliberately to demonstrate that second layer of defense.

> **To populate this section:** run `python scripts/ingest.py` then
> `python scripts/eval.py --out logs/eval_results.md` with a real
> `GOOGLE_API_KEY` in `.env`, then paste each Q/A/gate/citation block from
> `logs/eval_results.md` below.

```
[EASY]
Q: How many essential characteristics, service models, and deployment
models does the NIST definition of cloud computing have?
A: The NIST definition of cloud computing is composed of five essential characteristics, three service models, 
   and four deployment models.
Sources: - `docs\nistspecialpublication800-145.pdf` — chunks: nistspecialpublication800-145::chunk::5

[MEDIUM]
Q: What is VAERS, and how do you report a reaction to it?
A: VAERS stands for the Vaccine Adverse Event Reporting System. It is a system used for reporting adverse 
   reactions to vaccines. To report a reaction, you can visit the VAERS website at www.vaers.hhs.gov or call 
   1-800-822-7967. Typically, your health care provider will file the report, but you may also do it yourself.
Sources: - `docs\flu.pdf` — chunks: flu::chunk::1

[HARD -- grounding test]
Q: Is the moon and distance data in this document current, and how
should an answer handle that?
A: The document specifies that the moon counts provided are current as of July 2013. It notes that moon counts 
   can change because new moons are frequently discovered and some moons are awaiting official confirmation. The 
   document also indicates that the dwarf planet moons are not included in the provided list, nor are asteroid 
   moons.
Sources: - `docs\our_solar_system_lithograph.pdf` — chunks: our_solar_system_lithograph::chunk::2, our_solar_system_lithograph::chunk::3

[EDGE -- should be below_threshold]
Q: What is the maximum towing capacity of a diesel pickup truck?
A: I don't have enough information in the provided documents to answer that question.
Sources: none

[EDGE -- passes the gate, model should self-refuse]
Q: How many moons does the cloud have, according to these documents?
A: The provided documents do not contain information regarding a "cloud" having moons. The documents discuss 
   planetary moons and the Oort Cloud, which is described as a vast spherical shell of icy bodies surrounding the solar system, but they do not attribute moons to it.
Sources:- `docs\our_solar_system_lithograph.pdf` — chunks: our_solar_system_lithograph::chunk::3 
  our_solar_system_lithograph::chunk::2
```

For more details please Check **logs/eval_results.md**

## Additional design decisions

These extend past the two required deliverables above, covering points
worth being explicit about for a production-shaped MVP.

**"No relevant context found" is split into three outcomes, not one.**
`EMPTY_INDEX` (nothing has been ingested — a setup problem), `BELOW_THRESHOLD`
(chunks retrieved, none relevant enough — the expected, normal case), and
`RETRIEVAL_FAILED` (the embedder/vector store call itself errored — a system
problem handled by the reliability layer). All three skip the LLM call
entirely, but are logged and messaged differently, since they call for
different responses (a user question vs. an operator bug vs. a transient
outage).

**Citations are structured and schema-constrained, not inline, and come from two independent sources.**
The model's output is constrained server-side to `{"answer": ..., "cited_chunk_ids": [...]}` via each provider's native structured-output mode (`with_structured_output`, `method="json_schema"`) — not just requested via prompt text. A fallback to parsing raw text still exists for the rare case where even schema- constrained decoding doesn't return cleanly, and that fallback firing is now recorded on `QueryMetrics.structured_output_failed` rather than being invisible. Separately, `retrieved_sources` reports every chunk retrieval actually handed the model as context, computed straight from retrieval with zero dependency on the model's output — so a source list is always available even in the fallback case. Both are grouped by source file (not one row per chunk), so an answer drawing on two documents reads as two source entries each.

**Token-limit handling is asymmetric on purpose.** An overly long
*question* is rejected up front with a clear message, before any embedding
call — silently trimming a question can change its meaning and produce a
confidently wrong answer, which is exactly the failure mode this system is
built to avoid. An overly large assembled *context* (question + retrieved
chunks) is trimmed automatically by dropping the lowest-scored chunks
first, since this is normal, expected prompt assembly, not a user error.

**Component failures get retry → circuit breaker → optional fallback provider**
at the embedder/generator boundary (`src/application/reliability.py`),
so the graph never needs to know how a component recovers — only whether it eventually succeeded. A configured fallback provider (`FALLBACK_GENERATION_PROVIDER=openai`, etc.) is tried automatically once the primary's retries are exhausted; monitoring records which provider actually served a given call.

**Vector store: Chroma over FAISS/MongoDB/Pinecone, for this task specifically.**
Chroma stores vectors and metadata together with zero
server setup (`pip install chromadb`, persist to a folder) — since
citation/metadata is core to this deliverable, that removes a whole
separate id→metadata store FAISS would otherwise require. FAISS wins on
raw speed/footprint at large scale, which isn't relevant with 5 documents.
MongoDB/pgvector make sense when a Postgres/Mongo-based app already exists
and you'd rather not run a second store. Pinecone/Weaviate/Qdrant make
sense at real scale, multi-tenancy, or when built-in hybrid search is
wanted — all reasonable choices this task's size doesn't call for.

**Monitoring is a query log, not just diagnostics.** The per-query log
(question, retrieved scores, gate outcome, latency, tokens) doubles
as an eval dataset: `no_context_rate` tells you whether
`RELEVANCE_THRESHOLD` is miscalibrated (too high → rejecting answerable
questions; too low → letting weak matches through), which you can't tune
blind. Console output is intentionally partial (one summary line per
query) — full detail goes to the JSONL file, not the terminal.


## Trade-offs and limitations
 
Honest gaps and deliberate simplifications, not covered above.
 
**Chunking strategy**
(The main chunking/retrieval trade-offs — overlap cost, header-splitting
cutting content in half, dense-retrieval degrading at scale — are covered
in the required write-up earlier in this file. These are the ones that
aren't.)
- Chunk size is measured by an approximate word-based heuristic, not each
  provider's real tokenizer — actual chunk
  sizes drift somewhat from the configured 600-token target depending on
  word length, so `CHUNK_SIZE_TOKENS` is a target, not a guarantee.
- PDF chunks are now forced to break at page boundaries (specifically to make page-citation exact) — the trade-off
  that fix accepted: a sentence, table row, or procedure straddling a
  real page break is now *always* split into two chunks, with no
  overlap carried across the page boundary. Exact citations, at the cost
  of slightly worse continuity right at page breaks.
- Chunk size is one global constant for the whole document, regardless of
  content density — a dense statistics table and a page of prose both
  target the same 600 tokens, which is part of why table extraction
   is a known rough edge, not a solved problem.
- Markdown structural chunking only helps if the source document actually
  uses headers. A markdown file with no `#`/`##`/`###` structure falls
  straight through to flat character-based splitting with no benefit
  from the header-aware pass at all.
- Overlap isn't deduplicated at retrieval time — two overlapping chunks
  from the same region can both land in the same top-k, spending two
  context-budget slots on largely redundant text instead of one.
**Architecture**
- Hexagonal ports-and-adapters is more indirection than a 4-adapter MVP
  strictly needs — the payoff (swap Chroma/Gemini/OpenAI with a config
  change, not a rewrite) is real.
- Everything is synchronous, single-process, in-memory (circuit breaker
  state, rate limiter, session stats). Fine for a CLI/eval tool; would
  need real work — shared state, async I/O, multi-worker coordination —
  before this serves concurrent users.
**Technologies**
- Chroma's local persistent mode assumes a single writer/process. It's
  the right call at this document-set size (see reasoning above), but it
  is not a production multi-client store as-is.
- pypdf has no OCR. A scanned/image-only PDF will silently extract to
  near-empty text — no error, no warning, just an under-chunked document.
  Never hit in the 3 real PDFs tested, but there's no detection for it.
- Dense embeddings only, one similarity metric, no keyword layer — this is
  the same limitation the chunking write-up covers for retrieval quality,
  restated here as a technology choice: nothing in this stack catches an
  exact model number or error code that phrasing doesn't paraphrase well.
**Strategies**
- The relevance-threshold gate is a single scalar cutoff across every
  query shape. It's cheap and it works for genuinely unrelated queries,
  but it demonstrably cannot separate "unrelated" from "a compound
  question blending two real topics in the corpus" — verified empirically
  : an edge-case question scored *higher* than a
  legitimate one. No threshold value fixes that; it needs a different
  mechanism (re-ranking / LLM-judge), not a different number.
- Structured JSON output for citations is more reliable than inline
  markers, but not risk-free — the model occasionally returns non-JSON
  text, caught by a fallback that degrades to an uncited raw answer rather
  than failing loudly. Acceptable rate in testing, worth monitoring at
  scale via the `generation_output_not_json` log line.
- The test suite (69 tests, all passing, no API key needed) proves the
  pipeline's *logic* — gating, trimming, retries, citation formatting —
  never its retrieval *quality*, since the fake embedder used throughout
  has no real semantic understanding. That gap is real and only closes
  with a live-key run against `eval.py`, which is why this file keeps
  flagging which numbers came from which kind of run.
