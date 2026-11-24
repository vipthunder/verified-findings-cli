# Plearun: Offline Probabilistic Fact-Checking Engine

An offline, deterministic, probabilistic verification engine designed to check incident narrative claims against heterogeneous machine records (CSV, JSON, WebVTT) using **only Python 3 standard library** (zero external dependencies, zero network calls, zero LLMs).

---

## 1. How It Works

- **Messy File Ingestion & Uncertainty Priors:** 
  - *CSV:* Decodes via `utf-8-sig` with fallback to `cp1252` (handling smart quotes `\x93`, `\x94` and en-dashes `\x96`). Skips `#` comments while retaining exact 1-based file line row numbers. Uses `csv.reader` with auto-detected delimiters to preserve comma-containing quoted cells.
  - *JSON:* Utilizes `json.JSONDecoder.raw_decode` to capture nested event objects alongside exact raw text slices and 1-based object indices. Normalizes ISO-8601 strings, naive local datetimes (inheriting sibling timezones), and Unix epochs (seconds vs. milliseconds).
  - *WebVTT:* Splits on blank lines to cleanly isolate overlapping cues, assigning 1-based cue numbers.
  - *Priors:* Assigns observation noise $\sigma_{\text{sec}}$ ($1\text{s}$ for exact seconds/epochs, $30\text{s}$ for HH:MM, $40\text{s}$ for hedged phrasing like *about*, *roughly*, *~*).

- **Gaussian KDE Clock Synchronization:** 
  Identifies reference events from timezone-aware JSON logs and pairs cross-source anchors (911 calls, stops, dispatch). Evaluates offsets $\delta = t_{\text{device}} - t_{\text{ref}}$ with combined variance across a 1D Gaussian KDE over $[-3600\text{s}, +3600\text{s}]$ ($5\text{s}$ grid). Posterior confidence incorporates sample-size shrinkage: $\text{confidence} = \text{in\_band\_weight\_ratio} \times (1 - e^{-N/1.2})$. Devices with $\text{confidence} < 0.75$ remain unaligned ($\sigma_{\text{clock}} = \infty$), strictly forbidding time-based conflict assertions.

- **BM25 Retrieval & Bayesian Hypothesis Testing:** 
  Maintains inverted-index posting lists ($O(R \cdot L)$ build, $O(P \log K)$ query via `heapq`, executing in $<0.5\text{s}$ on $50,000$ rows). Filters multi-row traffic logs by incident proximity and semantic event categories. Evaluates quantitative claims via quadrature error $\sigma_{\text{total}} = \sqrt{\sigma_{\text{claim}}^2 + \sigma_{\text{rec}}^2 + \sigma_{\text{clock}}^2}$ and Gaussian residual $z = |\text{diff}| / \sigma_{\text{total}}$:
  - **`match`**: $P(\text{match}) = \exp(-0.5 z^2) \ge 0.70$ ($z \le 0.84$).
  - **`conflict`**: $z > 3.0$ on a reliably aligned clock.
  - **`not_settled`**: Uncertain middle band ($0.84 < z \le 3.0$) or ambiguous evidence.
  - **`not_checkable`**: Claims querying attributes absent from all record schemas (e.g., `lane`).

- **Deterministic Substring Gate:** 
  Verifies `quote in record.raw_text` before writing any `match` or `conflict`. Any failure automatically downgrades the finding to `not_settled` with `cite` and `quote` omitted. Record text is processed strictly as passive data, providing immunity against prompt injections.

---

## 2. Three Hardest Decisions

1. **Shrinkage-Penalized Clock Alignment Cutoff (0.75):** Rejecting alignment on single anchors ($N=1$ yields max confidence $0.565$, while concordant $N=2$ reaches $0.811 \ge 0.75$). A false conflict severely penalizes verification; requiring multiple concordant anchors prevents jittery officer notes or unsynchronized cameras from declaring unwarranted conflicts.
2. **Semantic Category Compatibility Gating:** Restricting temporal difference checks to compatible event types (e.g., 911 calls vs. CAD call logs). Without this, BM25 returns adjacent scene arrivals whose expected multi-minute separation falsely triggered high-$z$ conflicts.
3. **Raw Slice Extraction via `raw_decode`:** Storing verbatim substrings from the source stream rather than re-serializing parsed dicts, ensuring byte-exact compliance with the deterministic substring gate across all JSON and CSV variations.

---

## 3. What It Gets Wrong

- **Multi-Hop Causal Assertions:** Validates explicit timestamps and speeds, but cannot verify unrecorded causal assertions (e.g., *"driver swerved due to debris"*).
- **Novel Unstructured Phrasing:** Unseen terminology diverging heavily from standard police dispatch vocabulary may yield lower BM25 retrieval scores, conservatively defaulting to `not_settled`.

---

## 4. What Would Break First on Real Records

- **Non-Monotonic Timestamps & Midnight Roll-overs:** Real CAD systems experiencing GPS re-locks or crossing local midnight without explicit date headers will corrupt delta calculations without complex multi-day track assembly.
- **Corrupt / Malformed Multi-Line CSVs:** Unescaped newlines or quotes in proprietary device exports breaking standard RFC-4180 parsing.

---

## 5. Tools Used

- **Google Antigravity:** Development orchestration, sandbox execution, directory verification, and test execution.
- **Google Gemini:** Algorithmic design of the Gaussian KDE shrinkage function, Bayesian residual formulations, and generation of the comprehensive test suite (`test_pipeline.py`).
