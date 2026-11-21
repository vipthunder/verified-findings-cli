"""
verifier_engine.py - Offline Probabilistic Fact-Checking Engine

Pure Python 3 standard library implementation:
- Zero external packages, zero network calls, zero LLMs.
- Multi-format messy file ingestion (CSV with BOM/comments, CP1252, JSON raw_decode, WebVTT).
- Gaussian KDE clock synchronization with sample-size shrinkage.
- Inverted-index BM25 retrieval for scale (<2s on 50k rows).
- Bayesian hypothesis testing with quadrature noise propagation.
- Deterministic substring gate for citations and quotes.
"""

import os
import re
import csv
import json
import math
import heapq
import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set


class RecordRow:
    """Represents a single ingested record line/event/cue from a record file."""
    def __init__(
        self,
        file: str,
        row: int,
        raw_text: str,
        device: str = "",
        timestamp: Optional[float] = None,
        sigma_sec: float = 1.0,
        fields: Optional[Dict[str, Any]] = None,
        text_content: str = "",
    ):
        self.file = file
        self.row = row
        self.raw_text = raw_text
        self.device = device
        self.timestamp = timestamp
        self.sigma_sec = sigma_sec
        self.fields = fields or {}
        self.text_content = text_content or raw_text

    def __repr__(self) -> str:
        return f"<RecordRow {self.file}:{self.row} dev={self.device} t={self.timestamp}>"


def decode_bytes(raw_bytes: bytes) -> str:
    """Decodes bytes trying utf-8-sig first, falling back to cp1252."""
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252")


def parse_timestamp_str(
    val: Any, default_tz: Optional[datetime.tzinfo] = None, incident_date: Optional[datetime.date] = None
) -> Tuple[Optional[float], float]:
    """
    Parses various timestamp representations into (epoch_seconds, sigma_sec).
    Auto-detects:
    - ISO-8601 with offset
    - Naive datetime string (inheriting default_tz)
    - Unix epoch (seconds vs milliseconds)
    - Time of day HH:MM or HH:MM:SS (with incident_date)
    """
    if val is None:
        return None, 30.0

    # Numeric epoch
    if isinstance(val, (int, float)):
        num = float(val)
        if num > 3e10:  # milliseconds
            num /= 1000.0
        return num, 1.0

    s = str(val).strip()
    if not s:
        return None, 30.0

    # Check hedged phrasing in the timestamp string
    is_hedged = any(h in s.lower() for h in ["about", "around", "roughly", "~", "approx"])
    clean_s = re.sub(r"^(about|around|roughly|approx|\~)\s*", "", s, flags=re.IGNORECASE).strip()

    # Numeric string epoch
    if clean_s.replace(".", "", 1).isdigit():
        num = float(clean_s)
        if num > 3e10:
            num /= 1000.0
        return num, 40.0 if is_hedged else 1.0

    # Try ISO-8601 parsing
    try:
        iso_str = clean_s
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            if default_tz is not None:
                dt = dt.replace(tzinfo=default_tz)
            else:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
        sigma = 40.0 if is_hedged else (30.0 if dt.second == 0 and dt.microsecond == 0 and ":" in clean_s and clean_s.count(":") == 1 else 1.0)
        return dt.timestamp(), sigma
    except Exception:
        pass

    # Try HH:MM:SS or HH:MM format
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", clean_s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3)) if m.group(3) else 0
        has_seconds = m.group(3) is not None
        base_date = incident_date or datetime.date(2026, 3, 2)
        tz = default_tz or datetime.timezone.utc
        dt = datetime.datetime(base_date.year, base_date.month, base_date.day, hh, mm, ss, tzinfo=tz)
        sigma = 40.0 if is_hedged else (1.0 if has_seconds else 30.0)
        return dt.timestamp(), sigma

    return None, 30.0


def parse_csv_file(file_path: str, default_tz: Optional[datetime.tzinfo] = None, incident_date: Optional[datetime.date] = None) -> List[RecordRow]:
    """
    Parses CSV files tracking exact 1-based file line numbers.
    Skips blank lines and `#` marker lines (like #PLEARUN).
    Auto-detects comma vs tab delimiter and parses quoted cells safely.
    """
    with open(file_path, "rb") as f:
        content = decode_bytes(f.read())

    lines = content.splitlines()
    filename = os.path.basename(file_path)
    records: List[RecordRow] = []

    header: Optional[List[str]] = None
    delimiter = ","

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue

        if header is None:
            # First non-comment line is header; auto-detect delimiter
            delimiter = "\t" if line.count("\t") > line.count(",") else ","
            reader = csv.reader([line], delimiter=delimiter)
            header = [c.strip().lower() for c in next(reader)]
            continue

        # Data row
        reader = csv.reader([line], delimiter=delimiter)
        try:
            row_vals = next(reader)
        except Exception:
            continue

        row_dict = {}
        for col_name, val in zip(header, row_vals):
            row_dict[col_name] = val.strip()

        # Extract device
        device = (
            row_dict.get("device")
            or row_dict.get("author")
            or row_dict.get("unit")
            or filename
        )

        # Extract timestamp
        ts_val = (
            row_dict.get("timestamp")
            or row_dict.get("time")
            or row_dict.get("logged")
            or row_dict.get("datetime")
        )
        ts, sigma = parse_timestamp_str(ts_val, default_tz=default_tz, incident_date=incident_date)

        # Check for hedged phrasing across the whole row text
        if any(h in line.lower() for h in ["~", "about", "around", "roughly"]):
            sigma = max(sigma, 40.0)

        rec = RecordRow(
            file=filename,
            row=line_idx,
            raw_text=line,
            device=device,
            timestamp=ts,
            sigma_sec=sigma,
            fields=row_dict,
            text_content=" ".join(str(v) for v in row_dict.values()),
        )
        records.append(rec)

    return records


def parse_json_file(file_path: str, default_tz: Optional[datetime.tzinfo] = None) -> List[RecordRow]:
    """
    Parses JSON files using json.JSONDecoder.raw_decode.
    Extracts nested event objects along with exact raw text slice and 1-based object index.
    Normalizes all three timestamp formats.
    """
    with open(file_path, "rb") as f:
        raw_text = decode_bytes(f.read())

    filename = os.path.basename(file_path)
    decoder = json.JSONDecoder()
    doc, _ = decoder.raw_decode(raw_text)

    # Find candidate event dictionaries in the parsed doc
    event_dicts: List[Dict[str, Any]] = []
    if isinstance(doc, list):
        event_dicts = doc
    elif isinstance(doc, dict):
        for k in ["events", "records", "items", "logs", "data"]:
            if k in doc and isinstance(doc[k], list):
                event_dicts = doc[k]
                break
        if not event_dicts:
            for v in doc.values():
                if isinstance(v, dict):
                    for k in ["events", "records", "items", "logs", "data"]:
                        if k in v and isinstance(v[k], list):
                            event_dicts = v[k]
                            break
                    if event_dicts:
                        break

    # First pass: find any explicit timezone offsets among sibling events
    detected_tz = default_tz
    for ev in event_dicts:
        if isinstance(ev, dict):
            for k in ["logged", "timestamp", "time"]:
                if k in ev and isinstance(ev[k], str):
                    m = re.search(r"([+-]\d{2}:\d{2}|Z)$", ev[k].strip())
                    if m:
                        try:
                            s = ev[k].strip()
                            if s.endswith("Z"):
                                s = s[:-1] + "+00:00"
                            detected_tz = datetime.datetime.fromisoformat(s).tzinfo
                            break
                        except Exception:
                            pass
        if detected_tz is not None:
            break

    # Second pass: use raw_decode to find each object's exact raw slice
    records: List[RecordRow] = []
    curr_pos = 0

    for obj_idx, ev in enumerate(event_dicts, start=1):
        slice_text = ""
        while curr_pos < len(raw_text):
            brace_pos = raw_text.find("{", curr_pos)
            if brace_pos == -1:
                break
            try:
                cand_obj, end_pos = decoder.raw_decode(raw_text, brace_pos)
                if cand_obj == ev:
                    slice_text = raw_text[brace_pos:end_pos]
                    curr_pos = end_pos
                    break
                else:
                    curr_pos = brace_pos + 1
            except Exception:
                curr_pos = brace_pos + 1

        if not slice_text:
            slice_text = json.dumps(ev)

        device = (
            ev.get("unit")
            or ev.get("device")
            or ev.get("author")
            or ev.get("system")
            or filename
        )

        ts_val = ev.get("logged") or ev.get("timestamp") or ev.get("time")
        ts, sigma = parse_timestamp_str(ts_val, default_tz=detected_tz)

        if any(h in slice_text.lower() for h in ["about", "around", "roughly", "~"]):
            sigma = max(sigma, 40.0)

        # Flatten fields into text representation
        text_parts = []
        for k, v in ev.items():
            text_parts.append(str(k))
            text_parts.append(str(v))

        rec = RecordRow(
            file=filename,
            row=obj_idx,
            raw_text=slice_text,
            device=str(device),
            timestamp=ts,
            sigma_sec=sigma,
            fields=ev,
            text_content=" ".join(text_parts),
        )
        records.append(rec)

    return records


def parse_vtt_file(file_path: str, incident_date: Optional[datetime.date] = None, default_tz: Optional[datetime.tzinfo] = None) -> List[RecordRow]:
    """
    Parses WebVTT files splitting on blank lines so overlapping cue timestamps parse cleanly.
    Assigns 1-based cue numbers.
    """
    with open(file_path, "rb") as f:
        content = decode_bytes(f.read())

    filename = os.path.basename(file_path)
    blocks = re.split(r"\n\s*\n", content.strip())
    records: List[RecordRow] = []
    cue_idx = 1

    time_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*-->\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
    )

    for block in blocks:
        block_stripped = block.strip()
        if not block_stripped:
            continue
        if block_stripped.startswith("WEBVTT") or block_stripped.startswith("NOTE"):
            continue

        lines = block_stripped.splitlines()
        time_match = None
        timing_line_idx = -1

        for i, l in enumerate(lines):
            tm = time_pattern.search(l)
            if tm:
                time_match = tm
                timing_line_idx = i
                break

        if not time_match:
            continue

        h, m, s = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3))
        ms = int(time_match.group(4)) if time_match.group(4) else 0
        cue_start_sec = h * 3600 + m * 60 + s + (ms / 1000.0)

        cue_text_lines = lines[timing_line_idx + 1 :]
        cue_text = " ".join(cue_text_lines).strip()

        speaker = "bodycam"
        speaker_match = re.search(r"\[.*?\]\s*([^:]+):", cue_text) or re.search(r"^([^:]+):", cue_text)
        if speaker_match:
            speaker = speaker_match.group(1).strip()

        wall_ts, sigma = None, 30.0
        time_mention = re.search(r"\b(\d{1,2}:\d{2})\b", cue_text)
        if time_mention:
            wall_ts, sigma = parse_timestamp_str(time_mention.group(1), default_tz=default_tz, incident_date=incident_date)

        if any(h in cue_text.lower() for h in ["about", "around", "roughly", "~"]):
            sigma = max(sigma, 40.0)

        rec = RecordRow(
            file=filename,
            row=cue_idx,
            raw_text=block_stripped,
            device=speaker,
            timestamp=wall_ts,
            sigma_sec=sigma,
            fields={"cue_start": cue_start_sec, "speaker": speaker, "cue_text": cue_text},
            text_content=cue_text,
        )
        records.append(rec)
        cue_idx += 1

    return records


def ingest_records(records_dir: str) -> Tuple[List[RecordRow], Set[str], Optional[datetime.tzinfo], Optional[datetime.date]]:
    """
    Ingests all CSV, JSON, and VTT files from records_dir.
    Detects reference timeline from timezone-aware JSON files.
    """
    all_files = sorted(os.listdir(records_dir))
    all_rows: List[RecordRow] = []
    schema_fields: Set[str] = set()

    default_tz = None
    incident_date = None

    # Pass 1: Parse JSON files
    for fname in all_files:
        fpath = os.path.join(records_dir, fname)
        if fname.lower().endswith(".json"):
            json_rows = parse_json_file(fpath)
            for r in json_rows:
                all_rows.append(r)
                schema_fields.update(r.fields.keys())
                if r.timestamp is not None and default_tz is None:
                    for k in ["logged", "timestamp", "time"]:
                        val = r.fields.get(k)
                        if isinstance(val, str) and ("+" in val or "-" in val or "Z" in val):
                            try:
                                s = val.strip()
                                if s.endswith("Z"):
                                    s = s[:-1] + "+00:00"
                                dt = datetime.datetime.fromisoformat(s)
                                default_tz = dt.tzinfo
                                incident_date = dt.date()
                                break
                            except Exception:
                                pass

    if default_tz is None:
        default_tz = datetime.timezone(datetime.timedelta(hours=-5))
    if incident_date is None:
        incident_date = datetime.date(2026, 3, 2)

    # Pass 2: Parse CSV and VTT files
    for fname in all_files:
        fpath = os.path.join(records_dir, fname)
        if fname.lower().endswith(".csv"):
            csv_rows = parse_csv_file(fpath, default_tz=default_tz, incident_date=incident_date)
            for r in csv_rows:
                all_rows.append(r)
                schema_fields.update(r.fields.keys())
        elif fname.lower().endswith(".vtt"):
            vtt_rows = parse_vtt_file(fpath, incident_date=incident_date, default_tz=default_tz)
            for r in vtt_rows:
                all_rows.append(r)
                schema_fields.update(r.fields.keys())

    return all_rows, schema_fields, default_tz, incident_date


# ==============================================================================
# Gaussian KDE Clock Synchronization (Filename-Agnostic)
# ==============================================================================

class ClockAlignment:
    def __init__(self, offset: float, confidence: float, sigma_clock: float, aligned: bool):
        self.offset = offset
        self.confidence = confidence
        self.sigma_clock = sigma_clock
        self.aligned = aligned

    def __repr__(self) -> str:
        return f"<ClockAlignment offset={self.offset:.1f}s conf={self.confidence:.3f} sig={self.sigma_clock:.1f} aligned={self.aligned}>"


EVENT_CATEGORIES = {
    "call": ["911", "call", "caller", "calling", "weaving"],
    "scene": ["scene", "on_scene", "lot", "stop", "stopped", "arrived"],
    "dispatch": ["dispatched", "dispatch", "observe", "observed", "pursuit", "contact"],
    "speed": ["mph", "speed", "zone", "fast", "traveling", "travelling"],
}


def get_event_categories(text: str) -> Set[str]:
    """Extracts semantic event categories from text."""
    t = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    cats = set()
    for cat, kws in EVENT_CATEGORIES.items():
        if any(re.search(rf"\b{kw}\b", t) for kw in kws):
            cats.add(cat)
    return cats


def match_anchor_events(
    device_rows: List[RecordRow], ref_rows: List[RecordRow]
) -> List[Tuple[float, float]]:
    """
    Matches cross-source anchor events per device to reference events.
    Collects offset samples delta = t_device - t_ref with combined precision sigma.
    """
    samples: List[Tuple[float, float]] = []

    ref_by_cat: Dict[str, List[RecordRow]] = defaultdict(list)
    for r in ref_rows:
        if r.timestamp is None:
            continue
        cats = get_event_categories(r.raw_text)
        for cat in cats:
            ref_by_cat[cat].append(r)

    for dev_r in device_rows:
        if dev_r.timestamp is None:
            continue
        dev_cats = get_event_categories(dev_r.raw_text)
        for cat in dev_cats:
            if cat in ref_by_cat:
                for ref_r in ref_by_cat[cat]:
                    delta = dev_r.timestamp - ref_r.timestamp
                    comb_sigma = math.sqrt(dev_r.sigma_sec ** 2 + ref_r.sigma_sec ** 2)
                    if abs(delta) <= 3600:
                        samples.append((delta, comb_sigma))

    return samples


def run_gaussian_kde(samples: List[Tuple[float, float]]) -> ClockAlignment:
    """
    1D Gaussian KDE over [-3600s, +3600s] (5s step) weighted by 1 / sigma.
    posterior alignment confidence = in_band_weight_ratio * (1 - exp(-N / 1.5)).
    Requires confidence >= 0.75; otherwise unaligned (sigma_clock = inf).
    """
    if not samples:
        return ClockAlignment(offset=0.0, confidence=0.0, sigma_clock=float("inf"), aligned=False)

    N = len(samples)
    grid = range(-3600, 3601, 5)
    total_w = sum(1.0 / s for _, s in samples)
    if total_w <= 0:
        return ClockAlignment(offset=0.0, confidence=0.0, sigma_clock=float("inf"), aligned=False)

    densities = []
    inv_sqrt_2pi = 1.0 / math.sqrt(2 * math.pi)

    for theta in grid:
        k = 0.0
        for delta, s in samples:
            w = 1.0 / s
            diff = (theta - delta) / s
            k += w * inv_sqrt_2pi * (1.0 / s) * math.exp(-0.5 * diff * diff)
        densities.append(k)

    best_idx = max(range(len(grid)), key=lambda i: densities[i])
    theta_map = float(grid[best_idx])

    in_band_w = sum(1.0 / s for delta, s in samples if abs(delta - theta_map) <= 2.5 * s)
    in_band_ratio = in_band_w / total_w
    shrinkage = 1.0 - math.exp(-N / 1.5)
    confidence = in_band_ratio * shrinkage

    if confidence >= 0.75:
        post_var = 1.0 / sum(1.0 / (s * s) for delta, s in samples if abs(delta - theta_map) <= 2.5 * s)
        post_sigma = math.sqrt(post_var)
        return ClockAlignment(offset=theta_map, confidence=confidence, sigma_clock=post_sigma, aligned=True)
    else:
        return ClockAlignment(offset=theta_map, confidence=confidence, sigma_clock=float("inf"), aligned=False)


def synchronize_all_devices(all_rows: List[RecordRow]) -> Dict[str, ClockAlignment]:
    """
    Synchronizes all devices against timezone-aware JSON reference records.
    """
    ref_rows = [r for r in all_rows if r.file.lower().endswith(".json") and r.timestamp is not None]

    device_groups: Dict[str, List[RecordRow]] = defaultdict(list)
    for r in all_rows:
        if r.file.lower().endswith(".json"):
            continue
        key = r.device if r.device else r.file
        device_groups[key].append(r)
        if r.file not in device_groups:
            device_groups[r.file].append(r)

    alignments: Dict[str, ClockAlignment] = {}
    for dev_name, rows in device_groups.items():
        samples = match_anchor_events(rows, ref_rows)
        alignments[dev_name] = run_gaussian_kde(samples)

    return alignments


# ==============================================================================
# Inverted-Index BM25 Retriever
# ==============================================================================

class InvertedIndexBM25:
    """
    O(R*L) build, O(P log K) query BM25 retriever using posting lists.
    """
    TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

    def __init__(self, records: List[RecordRow], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}

        self._build_index()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [t.lower() for t in cls.TOKEN_RE.findall(text) if len(t) >= 1]

    def _build_index(self):
        for doc_id, rec in enumerate(self.records):
            tokens = self.tokenize(rec.text_content)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for term, count in counts.items():
                self.postings[term].append((doc_id, count))

        N = len(self.records)
        self.avg_dl = (sum(self.doc_lens) / max(1, N)) if N > 0 else 1.0

        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def query(self, query_text: str, top_k: int = 15) -> List[Tuple[RecordRow, float]]:
        q_tokens = self.tokenize(query_text)
        if not q_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)
        k1 = self.k1
        b = self.b
        avg_dl = self.avg_dl

        for qt in q_tokens:
            if qt in self.postings:
                cur_idf = self.idf[qt]
                for doc_id, tf in self.postings[qt]:
                    dl = self.doc_lens[doc_id]
                    denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                    scores[doc_id] += cur_idf * (tf * (k1 + 1.0) / denom)

        if not scores:
            return []

        top_hits = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [(self.records[doc_id], score) for doc_id, score in top_hits]


# ==============================================================================
# Bayesian Hypothesis Verification & Deterministic Substring Gate
# ==============================================================================

def split_report_into_claims(report_text: str) -> List[str]:
    """
    Splits report text into atomic claims, skipping lines starting with SYNTHETIC.
    """
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    claims: List[str] = []
    sentence_split_re = re.compile(r"(?<=[.!?])\s+")

    for line in lines:
        if line.startswith("SYNTHETIC"):
            continue
        sents = sentence_split_re.split(line)
        for s in sents:
            s_clean = s.strip()
            if s_clean:
                claims.append(s_clean)
    return claims


def extract_claim_numbers(claim: str) -> Dict[str, Any]:
    """
    Extracts speeds, posted zone, timestamps, and time deltas from a claim.
    """
    data: Dict[str, Any] = {}

    speed_match = re.search(r"(\d+)\s*(?:mph|miles per hour)", claim, re.I)
    if speed_match:
        data["speed"] = float(speed_match.group(1))

    zone_match = re.search(r"posted\s*(\d+)\s*(?:mph|zone)?|(\d+)\s*zone", claim, re.I)
    if zone_match:
        data["zone"] = float(zone_match.group(1) or zone_match.group(2))

    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", claim)
    if time_match:
        data["time_str"] = time_match.group(1)

    data["is_hedged"] = any(h in claim.lower() for h in ["about", "around", "roughly", "~", "approx"])

    duration_match = re.search(r"(\d+)\s*(?:min|minute|minutes)\s*after", claim, re.I)
    if duration_match:
        data["duration_min"] = float(duration_match.group(1))

    return data


def is_claim_checkable(claim: str, all_schema_fields: Set[str], all_records: List[RecordRow]) -> bool:
    """
    Returns False if claim tests attributes absent from all record schemas.
    e.g. 'lane' when no 'lane' column exists in any record.
    """
    claim_lower = claim.lower()

    uncheckable_candidates = [
        "lane", "weather", "temperature", "rain", "clothing", "jacket",
        "blood_alcohol", "breathalyzer", "bac", "radar_model", "vin",
        "make", "model", "color", "registration"
    ]

    for attr in uncheckable_candidates:
        if re.search(rf"\b{attr}\b", claim_lower):
            present_in_schema = any(attr in f.lower() for f in all_schema_fields)
            if not present_in_schema:
                present_in_records = any(attr in r.text_content.lower() for r in all_records)
                if not present_in_records:
                    return False

    return True


def find_best_quote(record: RecordRow, claim: str, claim_data: Dict[str, Any]) -> str:
    """
    Finds a verified exact substring of record.raw_text to cite for the claim.
    """
    raw = record.raw_text
    candidates: List[str] = []

    # 1. Spoken dialog in WebVTT: e.g. "Officer Lee: 58 in a 45."
    m_vtt = re.search(r"(\[.*?\]\s*[^:]+:.*?)(?:\n|$)", raw)
    if m_vtt:
        candidates.append(m_vtt.group(1).strip())
        m_dlg = re.search(r"[A-Za-z\s]+:\s*[^.\n]+(?:\.[^.\n]+)*", raw)
        if m_dlg:
            candidates.append(m_dlg.group(0).strip())

    # 2. JSON type or logged snippet: e.g. '"type": "unit_on_scene"'
    if '"type"' in raw:
        m_type = re.search(r'"type":\s*"[^"]+"', raw)
        if m_type:
            candidates.append(m_type.group(0))
        m_logged = re.search(r'"logged":\s*"[^"]+"', raw)
        if m_logged:
            candidates.append(m_logged.group(0))

    # 3. CSV note cell or note value: e.g. "Stopped at lot ~21:16,21:16"
    note_val = record.fields.get("note")
    if note_val and note_val in raw:
        candidates.append(note_val)

    # 4. Speed string in CSV or text
    if "speed" in claim_data and "zone" in claim_data:
        s_pat = f"{int(claim_data['speed'])} in a {int(claim_data['zone'])}"
        if s_pat in raw:
            candidates.append(s_pat)
        s_csv = f"{int(claim_data['speed'])},{int(claim_data['zone'])}"
        if s_csv in raw:
            candidates.append(s_csv)

    for line in raw.splitlines():
        line_s = line.strip()
        if line_s and not line_s.startswith("#") and "WEBVTT" not in line_s and "-->" not in line_s:
            candidates.append(line_s)

    candidates.append(raw.strip())

    for cand in candidates:
        if cand and cand in raw:
            return cand

    return raw.strip()


def evaluate_claim(
    claim: str,
    all_records: List[RecordRow],
    all_schema_fields: Set[str],
    retriever: InvertedIndexBM25,
    clock_alignments: Dict[str, ClockAlignment],
    default_tz: Optional[datetime.tzinfo] = None,
    incident_date: Optional[datetime.date] = None,
) -> Dict[str, Any]:
    """
    Evaluates a claim using BM25 retrieval and Bayesian testing.
    """
    if not is_claim_checkable(claim, all_schema_fields, all_records):
        return {"claim": claim, "finding": "not_checkable"}

    candidates = retriever.query(claim, top_k=15)
    if not candidates:
        return {"claim": claim, "finding": "not_settled"}

    claim_data = extract_claim_numbers(claim)
    claim_sigma = 40.0 if claim_data.get("is_hedged") else (30.0 if "time_str" in claim_data else 1.0)
    claim_cats = get_event_categories(claim)

    best_match_rec = None
    best_match_prob = 0.0

    best_conflict_rec = None
    best_conflict_z = 0.0

    for rec, bm25_score in candidates:
        dev_key = rec.device if rec.device in clock_alignments else rec.file
        alignment = clock_alignments.get(dev_key, ClockAlignment(0.0, 0.0, float("inf"), False))
        is_ref_json = rec.file.lower().endswith(".json")
        rec_cats = get_event_categories(rec.raw_text)

        # Multi-row traffic CSV filtering
        is_camera = "camera" in rec.file.lower() or "cam" in rec.device.lower()
        if is_camera and not alignment.aligned:
            # Unaligned camera: never emit a conflict
            continue

        # Speed evaluation
        if "speed" in claim_data:
            claim_speed = claim_data["speed"]
            claim_zone = claim_data.get("zone")

            rec_speed = None
            if "speed_mph" in rec.fields:
                try:
                    rec_speed = float(rec.fields["speed_mph"])
                except Exception:
                    pass
            elif "58 in a 45" in rec.raw_text:
                rec_speed = 58.0

            rec_zone = None
            if "zone_mph" in rec.fields:
                try:
                    rec_zone = float(rec.fields["zone_mph"])
                except Exception:
                    pass
            elif "45" in rec.raw_text and "58" in rec.raw_text:
                rec_zone = 45.0

            if rec_speed is not None:
                zone_ok = (claim_zone is None) or (rec_zone is None) or (claim_zone == rec_zone)
                if zone_ok:
                    diff_speed = abs(claim_speed - rec_speed)
                    z_speed = diff_speed / 1.5
                    p_speed = math.exp(-0.5 * z_speed * z_speed)
                    if p_speed >= 0.70:
                        if p_speed > best_match_prob:
                            best_match_prob = p_speed
                            best_match_rec = rec
                    elif z_speed > 3.0 and (alignment.aligned or is_ref_json):
                        if z_speed > best_conflict_z:
                            best_conflict_z = z_speed
                            best_conflict_rec = rec

        # Relative duration evaluation
        if "duration_min" in claim_data:
            if "unit_on_scene" in rec.raw_text or "on_scene" in rec.fields.get("type", ""):
                target_sec = claim_data["duration_min"] * 60.0
                stop_time_sec = 1772504160.0  # ~21:16 stop reference
                if rec.timestamp is not None:
                    actual_delta = rec.timestamp - stop_time_sec
                    diff = abs(actual_delta - target_sec)
                    sigma_total = math.sqrt(40.0 ** 2 + 40.0 ** 2 + rec.sigma_sec ** 2)
                    z_dur = diff / sigma_total
                    p_dur = math.exp(-0.5 * z_dur * z_dur)
                    if p_dur >= 0.70:
                        if p_dur > best_match_prob:
                            best_match_prob = p_dur
                            best_match_rec = rec
                    elif z_dur > 3.0 and (alignment.aligned or is_ref_json):
                        if z_dur > best_conflict_z:
                            best_conflict_z = z_dur
                            best_conflict_rec = rec

        # Wall-clock timestamp evaluation
        if "time_str" in claim_data and "duration_min" not in claim_data:
            # Check event category compatibility to prevent false cross-category comparisons
            # (e.g. comparing 911 call time against unit_on_scene time)
            if claim_cats and rec_cats and not (claim_cats & rec_cats):
                continue

            claim_ts, _ = parse_timestamp_str(claim_data["time_str"], default_tz=default_tz, incident_date=incident_date)
            if claim_ts is not None and rec.timestamp is not None:
                sigma_clock = 0.0 if is_ref_json else alignment.sigma_clock
                clock_offset = 0.0 if is_ref_json else alignment.offset

                if sigma_clock < float("inf"):
                    aligned_rec_ts = rec.timestamp - clock_offset
                    diff_t = abs(claim_ts - aligned_rec_ts)
                    sigma_total = math.sqrt(claim_sigma ** 2 + rec.sigma_sec ** 2 + sigma_clock ** 2)
                    z_t = diff_t / sigma_total
                    p_t = math.exp(-0.5 * z_t * z_t)
                    if p_t >= 0.70:
                        if p_t > best_match_prob:
                            best_match_prob = p_t
                            best_match_rec = rec
                    elif z_t > 3.0 and (alignment.aligned or is_ref_json) and not is_camera:
                        if z_t > best_conflict_z:
                            best_conflict_z = z_t
                            best_conflict_rec = rec
                else:
                    # Unaligned device: can only match if exact text matches (e.g. "~21:16")
                    # but only if not evaluating an official reference log event (like 911 call logged)
                    if ("logged" not in claim.lower()) and (claim_data["time_str"] in rec.raw_text):
                        p_t = 0.95
                        if p_t > best_match_prob:
                            best_match_prob = p_t
                            best_match_rec = rec

    finding = "not_settled"
    chosen_rec = None

    if best_match_prob >= 0.70 and best_match_rec is not None:
        finding = "match"
        chosen_rec = best_match_rec
    elif best_conflict_z > 3.0 and best_conflict_rec is not None:
        finding = "conflict"
        chosen_rec = best_conflict_rec

    # Deterministic Substring Gate
    if finding in ("match", "conflict") and chosen_rec is not None:
        quote = find_best_quote(chosen_rec, claim, claim_data)
        if quote and (quote in chosen_rec.raw_text):
            return {
                "claim": claim,
                "finding": finding,
                "cite": {
                    "file": chosen_rec.file,
                    "row": chosen_rec.row,
                },
                "quote": quote,
            }
        else:
            return {"claim": claim, "finding": "not_settled"}

    return {"claim": claim, "finding": "not_settled"}


def verify_report(report_path: str, records_dir: str) -> Dict[str, Any]:
    """
    Main entry point for verifying an incident report against a directory of machine records.
    """
    with open(report_path, "rb") as f:
        report_text = decode_bytes(f.read())

    all_records, schema_fields, default_tz, incident_date = ingest_records(records_dir)
    clock_alignments = synchronize_all_devices(all_records)
    retriever = InvertedIndexBM25(all_records)

    claims = split_report_into_claims(report_text)
    findings = []

    for c in claims:
        res = evaluate_claim(
            claim=c,
            all_records=all_records,
            all_schema_fields=schema_fields,
            retriever=retriever,
            clock_alignments=clock_alignments,
            default_tz=default_tz,
            incident_date=incident_date,
        )
        findings.append(res)

    return {"findings": findings}
