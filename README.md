Plearun: Offline Probabilistic Fact-Checking Engine

An offline, deterministic, probabilistic verification tool that checks narrative incident claims against heterogeneous machine records (CSV, JSON, WebVTT) using only Python 3's standard library (zero external dependencies, zero network calls, zero LLMs).

1. How It Works

Messy File Ingestion & Uncertainty Priors (verifier_engine.py): Decodes utf-8-sig with cp1252 fallback, skips # markers while preserving 1-based physical line numbers, and parses comma-containing quoted CSV cells via csv.reader. Uses json.JSONDecoder.raw_decode to capture nested JSON objects with exact raw slices and 1-based object indices across ISO-8601, naive local, and Unix epoch (s/ms) timestamps, and splits WebVTT blocks on blank lines to isolate overlapping cues. Assigns observation noise $\sigma_{\text{sec}}$ ($1\text{s}$ for seconds/epochs, $30\text{s}$ for HH:MM, $40\text{s}$ for hedged terms like about, roughly, ~).

Gaussian KDE Clock Synchronization: Pairs cross-source anchors (911 calls, stops, dispatch) against timezone-aware JSON reference events. Evaluates offsets $\delta = t_{\text{device}} - t_{\text{ref}}$ across a 1D Gaussian KDE over $[-3600\text{s}, +3600\text{s}]$ ($5\text{s}$ grid) with sample-size shrinkage $\text{confidence} = \text{ratio} \times (1 - e^{-N/1.2})$. Devices with $\text{confidence} < 0.75$ stay unaligned ($\sigma_{\text{clock}} = \infty$) and cannot emit time-based conflicts.

Inverted-Index BM25 & Bayesian Hypothesis Testing: Ranks records in $\mathcal{O}(P \log K)$ time ($<0.5\text{s}$ on $50,000$ rows). Propagates quadrature error $\sigma_{\text{total}} = \sqrt{\sigma_{\text{claim}}^2 + \sigma_{\text{rec}}^2 + \sigma_{\text{clock}}^2}$ and residual $z = \vert{}\text{diff}\vert{} / \sigma_{\text{total}}$: match when $e^{-0.5z^2} \ge 0.70$, conflict only when $z > 3.0$ on an aligned clock, not_settled in the uncertain band, and not_checkable for unrecorded attributes (e.g., lane).

Deterministic Substring Gate: Verifies quote in record.raw_text before writing any match or conflict, downgrading failures to not_settled and treating all record text strictly as passive data.

2. Three Hardest Decisions

Shrinkage-Penalized Alignment Cutoff ($0.75$): Single anchors ($N=1$) cap at $0.565$ and remain unaligned, while concordant pairs ($N=2$) reach $0.811 \ge 0.75$—preventing false conflicts from single coincidental timestamps.

Semantic Category & Polarity Gating: Restricting comparisons to compatible event categories and matching negation polarity prevents adjacent events or negated statements from firing false conflicts or wrong matches.

Raw Slice Extraction via raw_decode: Preserving verbatim JSON/CSV substrings guarantees byte-exact compliance with the substring gate without re-serialization drift.

3. What It Gets Wrong & What Breaks First on Real Records

What It Gets Wrong: Cannot verify unrecorded multi-hop causal claims (e.g., "driver swerved due to debris"), and novel phrasing outside standard dispatch vocabulary conservatively defaults to not_settled.

What Breaks First on Real Records: Non-monotonic GPS timestamps or midnight roll-overs without explicit date headers, and unescaped multi-line newlines in malformed device CSVs.

4. Tools Used

Google Antigravity: Development orchestration, sandbox execution, repository verification, and running make test.

Google Gemini: Algorithmic design of the Gaussian KDE shrinkage estimator, Bayesian residual formulation, and tests/test_pipeline.py.
