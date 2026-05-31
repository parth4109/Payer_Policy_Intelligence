"""
==============================================================
Payer Policy Intelligence Pipeline — v3 FIXED
==============================================================
Inherits ALL v1 and v2 fixes unchanged. Three new fixes in v3:

  FIX G — assemble_dataframe: phototherapy "N/A" → "No"
           pandas read_csv silently converts the string "N/A" to
           NaN (it is in pandas default na_values list). So writing
           photo_display="N/A" produced NaN on every re-read. Since
           business rules treat N/A (no criteria at all) and No
           (phototherapy not required) identically for scoring, we
           now always output "No" for that case. "Yes" is unchanged.

  FIX H — stage5_output / to_csv: write with keep_default_na guard
           df.to_csv(...) now uses na_rep="" so any residual Python
           None values that slipped through become empty string
           rather than the literal "nan" text. validate_dataframe
           already fills blanks with "Not specified", so the end
           result is a fully populated CSV with no NaN cells.

  FIX I — assemble_dataframe Fix B broadened: phototherapy generic
           overcounting correction no longer requires a numbered-
           list pattern (\\d+). It now also fires when ≥3 AND
           conjunctions appear in the step text (indicating an
           AND-list written in prose, e.g. Oregon Medicaid Q11).
           The OR-phototherapy guard is unchanged so false positives
           are prevented.

  No new API calls. All fixes are pure Python post-processing.
  Nothing from v1/v2 was removed or weakened.
"""

import os, sys, json, re, time, hashlib, warnings
from pathlib import Path

import pandas as pd
import pdfplumber
import fitz
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "Data"
OUTPUT_DIR      = BASE_DIR / "outputs"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"
BUSINESS_RULES  = BASE_DIR / "PA_Business_rules_pc.xlsx"

OUTPUT_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

CACHE_FILE     = OUTPUT_DIR / "extraction_log.json"
RESULT_CSV     = OUTPUT_DIR / "result.csv"
SCORE_CSV      = OUTPUT_DIR / "score_breakdown.csv"
ERROR_LOG      = OUTPUT_DIR / "ingestion_errors.log"
DASHBOARD_HTML = OUTPUT_DIR / "dashboard.html"
RAW_TEXTS_CACHE = OUTPUT_DIR / "raw_texts_cache.json"

MODEL_FAST   = "llama-3.1-8b-instant"
MODEL_STRONG = "llama-3.3-70b-versatile"

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set.")

client = Groq(api_key=GROQ_API_KEY)

BRAND_KEYWORDS = {
    "TREMFYA":   ["TREMFYA", "guselkumab", "tremfya"],
    "STELARA":   ["STELARA", "ustekinumab", "stelara"],
    "AMJEVITA":  ["AMJEVITA", "adalimumab-atto", "amjevita"],
    "COSENTYX":  ["COSENTYX", "secukinumab", "cosentyx"],
    "ENBREL":    ["ENBREL", "etanercept", "enbrel"],
    "REMICADE":  ["REMICADE", "infliximab", "remicade"],
    "SILIQ":     ["SILIQ", "brodalumab", "siliq"],
    "CIMZIA":    ["CIMZIA", "certolizumab", "cimzia"],
    "BIMZELX":   ["BIMZELX", "bimekizumab", "bimzelx"],
    "SKYRIZI":   ["SKYRIZI", "risankizumab", "skyrizi"],
    "OTEZLA":    ["OTEZLA", "apremilast", "otezla"],
    "YESINTEK":  ["YESINTEK", "ustekinumab-kfce", "yesintek"],
    "OTULFI":    ["OTULFI", "ustekinumab-aauz", "otulfi"],
    "ILUMYA":    ["ILUMYA", "tildrakizumab", "ilumya"],
    "ACITRETIN": ["ACITRETIN", "acitretin", "soriatane"],
}

ALL_BRAND_NAMES = [kw for kws in BRAND_KEYWORDS.values() for kw in kws]

OUTPUT_COLUMNS = [
    "Filename", "Brand", "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required", "Quantity Limits", "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]

# ─── Payer-type hints ─────────────────────────────────────────
# Payers known to NOT specify auth durations inline (CPB-style)
CPB_STYLE_PAYERS = ["aetna", "anthem", "cigna", "humana"]

def detect_payer_type(text: str) -> str:
    """Return lowercase payer name hint from document text."""
    t = text[:3000].lower()
    for p in ["aetna", "anthem", "cigna", "humana", "hmsa", "bcbs",
              "united", "medicaid", "medicare", "caremore", "molina",
              "centene", "magellan", "priority", "horizon"]:
        if p in t:
            return p
    return "unknown"

def is_omnibus_doc(text: str) -> bool:
    """True if doc covers many drugs (state Medicaid, omnibus PA criteria)."""
    t = text[:5000].lower()
    indicators = ["prior authorization criteria", "pharmaceutical services",
                  "fee-for-service", "preferred drug list", "pdl",
                  "table of contents", "oregon medicaid", "medicaid pa criteria"]
    return sum(1 for i in indicators if i in t) >= 2


# ════════════════════════════════════════════════════════════════
# STAGE 1 — PDF INGESTION  (unchanged)
# ════════════════════════════════════════════════════════════════

def extract_text_pdfplumber(pdf_path):
    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if not text.strip():
                    words = page.extract_words()
                    text = " ".join(w["text"] for w in words)
                pages.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
        return "\n".join(pages)
    except Exception:
        return None

def extract_text_fitz(pdf_path):
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
        doc.close()
        return "\n".join(pages)
    except Exception:
        return None

def stage1_ingest_pdfs():
    print("\n" + "="*60)
    print("STAGE 1 — PDF Ingestion")
    print("="*60)
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    pdf_names = {f.name for f in pdf_files}
    print(f"Found {len(pdf_files)} PDFs in {DATA_DIR}")
    raw_texts = {}
    errors = []
    if RAW_TEXTS_CACHE.exists():
        try:
            cached = json.loads(RAW_TEXTS_CACHE.read_text(encoding="utf-8"))
            cached = {k: v.replace("\r\n", "\n") for k, v in cached.items()}
            if all(n in cached for n in pdf_names):
                print(f"  Loaded all {len(pdf_names)} from cache")
                return {n: cached[n] for n in pdf_names}
            raw_texts = {n: cached[n] for n in pdf_names if n in cached}
        except Exception:
            raw_texts = {}
    for pdf_path in tqdm(pdf_files, desc="  Ingesting"):
        fname = pdf_path.name
        if fname in raw_texts:
            continue
        text = extract_text_pdfplumber(pdf_path)
        if text is None or len(text.strip()) < 200:
            text = extract_text_fitz(pdf_path)
        if text is None or len(text.strip()) < 50:
            errors.append(fname)
            text = ""
        raw_texts[fname] = text.replace("\r\n", "\n")
    if errors:
        ERROR_LOG.write_text("\n".join(errors), encoding="utf-8")
    try:
        RAW_TEXTS_CACHE.write_text(
            json.dumps(raw_texts, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    print(f"  Stage 1 complete: {len(raw_texts)-len(errors)}/{len(pdf_files)} PDFs extracted")
    return raw_texts


# ════════════════════════════════════════════════════════════════
# STAGE 2 — BRAND SEGMENTATION  ← KEY FIX #1
# ════════════════════════════════════════════════════════════════

# Keywords that mark the START of a shared biologic approval section
# in omnibus Medicaid docs (no brand-specific header exists)
OMNIBUS_BIOLOGIC_HEADERS = [
    "biologic",
    "approval criteria",
    "plaque psoriasis",
    "targeted immune modulator",
    "biologic agent",
    "systemic therapy",
    "prior authorization criteria",
]

def segment_text_for_brand(full_text: str, brand: str, max_chars: int = 9000) -> str:
    """
    Extract the most brand-relevant portion of the document text.

    FIX vs original:
    - For omnibus Medicaid-style docs (many drugs, no brand-specific section),
      extract the SHARED biologic approval criteria block (plaque psoriasis
      section) instead of relying purely on brand keyword proximity.
    - This prevents the model from reading unrelated drug criteria.
    """
    keywords = [k.lower() for k in BRAND_KEYWORDS.get(brand, [brand])]
    other_brand_kws = {
        k.lower()
        for b, kws in BRAND_KEYWORDS.items()
        if b != brand
        for k in kws
    }

    lines = full_text.split("\n")
    n = len(lines)

    # ── Detect omnibus doc early ──────────────────────────────
    omnibus = is_omnibus_doc(full_text)

    # ── Step 1: Find explicit brand section headers ───────────
    header_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        line_lower = stripped.lower()
        if any(kw in line_lower for kw in keywords) and len(stripped) < 120:
            if len(stripped) > 5:
                header_indices.append(i)

    if header_indices and not omnibus:
        # Standard (non-omnibus) brand-section extraction
        start = header_indices[0]
        end = n
        for j in range(start + 5, n):
            stripped = lines[j].strip()
            line_lower = stripped.lower()
            if (
                len(stripped) < 100
                and any(kw in line_lower for kw in other_brand_kws)
                and not any(kw in line_lower for kw in keywords)
                and (stripped.isupper() or stripped.istitle())
            ):
                end = j
                break
        segment = "\n".join(lines[start:end])
        if len(segment) > max_chars:
            segment = segment[:max_chars]
        if len(segment.strip()) > 200:
            return segment

    # ── Step 2 (NEW): Omnibus doc — grab the shared biologic   ──
    # criteria section (plaque psoriasis approval flow).
    # We look for the biologic-approval table/section and take it
    # along with surrounding context including renewal criteria.
    if omnibus:
        # Find the plaque psoriasis approval block
        pso_block_start = -1
        for i, line in enumerate(lines):
            ll = line.strip().lower()
            # Look for question-flow markers near plaque psoriasis
            if ("plaque psoriasis" in ll and len(line.strip()) < 80) or \
               ("biologic" in ll and "plaque" in ll):
                pso_block_start = max(0, i - 5)
                break

        # Also find the renewal/continuation criteria
        renewal_start = -1
        for i, line in enumerate(lines):
            ll = line.strip().lower()
            if any(x in ll for x in ["renewal criteria", "reauthorization", "continuation of therapy"]):
                renewal_start = i
                break

        # Also include any universal criteria (TB, etc.) near top of approval section
        universal_start = -1
        for i, line in enumerate(lines):
            ll = line.strip().lower()
            if any(x in ll for x in ["tuberculosis", "tb test", "tb screen",
                                       "annual screen", "universal", "all indications"]):
                universal_start = max(0, i - 2)
                break

        segments = []

        # Include brand mention context for identity confirmation
        for hi in header_indices[:2]:
            seg = "\n".join(lines[max(0, hi-2): hi+30])
            if len(seg.strip()) > 20:
                segments.append(f"[Brand context]\n{seg}")

        if pso_block_start >= 0:
            end = min(n, pso_block_start + 200)  # ~200 lines covers approval flow
            seg = "\n".join(lines[pso_block_start:end])
            segments.append(f"[Plaque psoriasis approval criteria]\n{seg}")

        if renewal_start >= 0:
            end = min(n, renewal_start + 60)
            seg = "\n".join(lines[renewal_start:end])
            segments.append(f"[Renewal/reauth criteria]\n{seg}")

        if universal_start >= 0:
            end = min(n, universal_start + 20)
            seg = "\n".join(lines[universal_start:end])
            segments.append(f"[Universal/TB criteria]\n{seg}")

        combined = "\n\n".join(segments)
        if len(combined.strip()) > 300:
            return combined[:max_chars]

    # ── Step 3: Paragraph scoring by keyword density ─────────
    paragraphs = re.split(r"\n{2,}", full_text)
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(para_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, para))
    if scored:
        scored.sort(key=lambda x: -x[0])
        collected, total_len = [], 0
        for _, para in scored:
            if total_len + len(para) > max_chars:
                break
            collected.append(para)
            total_len += len(para)
        if collected:
            return "\n\n".join(collected)

    # ── Step 4: Full text fallback ────────────────────────────
    return full_text[:max_chars]


def load_jobs_from_submissions():
    df = pd.read_excel(BUSINESS_RULES, sheet_name="Submissions")
    jobs = []
    for _, row in df.iterrows():
        fname = str(row["Filename"]).strip()
        brand = str(row["Brand"]).strip()
        if fname and brand and fname != "nan" and brand != "nan":
            jobs.append((fname, brand))
    return jobs


def stage2_build_jobs(raw_texts: dict):
    print("\n" + "="*60)
    print("STAGE 2 — Brand Detection & Job Segmentation")
    print("="*60)
    try:
        jobs_raw = load_jobs_from_submissions()
        print(f"  Loaded {len(jobs_raw)} jobs from Submissions sheet")
    except Exception as e:
        print(f"  WARNING: {e}. Auto-detect mode.")
        jobs_raw = []

    existing_jobs = {}
    for fname, brand in jobs_raw:
        existing_jobs.setdefault(fname, set()).add(brand)

    new_jobs = []
    for fname, text in raw_texts.items():
        if fname not in existing_jobs:
            text_lower = text.lower()
            detected = [b for b, kws in BRAND_KEYWORDS.items()
                        if any(k.lower() in text_lower for k in kws)]
            if detected:
                print(f"  Auto-detected '{fname}': {detected}")
                for brand in detected:
                    new_jobs.append((fname, brand))

    all_jobs_raw = jobs_raw + new_jobs
    jobs = []
    for fname, brand in all_jobs_raw:
        if fname not in raw_texts:
            jobs.append((fname, brand, ""))
            continue
        segment = segment_text_for_brand(raw_texts[fname], brand)
        jobs.append((fname, brand, segment))

    brand_counts = {}
    for _, brand, _ in jobs:
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    print("\n  Brand distribution:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"    {brand:15s}: {count}")
    print(f"\n  Stage 2 complete: {len(jobs)} jobs")
    return jobs


# ════════════════════════════════════════════════════════════════
# STAGE 3 — EXTRACTION PROMPT  ← KEY FIX #2
# ════════════════════════════════════════════════════════════════

_cache: dict = {}

def _load_cache():
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            print(f"  Loaded {len(_cache)} cached responses")
        except Exception:
            _cache = {}

def _save_cache():
    CACHE_FILE.write_text(json.dumps(_cache, indent=2, ensure_ascii=False), encoding="utf-8")

def _cache_key(model, prompt):
    h = hashlib.sha256(prompt.replace("\r\n", "\n").encode()).hexdigest()[:20]
    return f"{model}::{h}"

_call_counts = {MODEL_FAST: 0, MODEL_STRONG: 0}
RPD_LIMITS   = {MODEL_FAST: 14400, MODEL_STRONG: 1000}

# ── TPM (Tokens Per Minute) throttle ─────────────────────────
# Free Groq tier: 6,000 TPM for llama-3.1-8b-instant
# Paid / Dev tier: 30,000 TPM. Adjust TPM_LIMIT if you upgrade.
TPM_LIMIT    = 6000          # tokens per minute ceiling (free tier)
TPM_MARGIN   = 0.85          # target 85 % of limit to leave headroom
TPM_WINDOW   = 60.0          # rolling window in seconds
CHARS_PER_TOKEN = 4          # rough chars→tokens estimate (conservative)

# Rolling token-usage log: list of (timestamp, token_count)
_tpm_log: list = []

def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters (safe over-estimate)."""
    return max(1, len(text) // CHARS_PER_TOKEN)

def _tpm_throttle(estimated_tokens: int) -> None:
    """
    Block until there is enough TPM headroom to send `estimated_tokens`.
    Purges log entries older than 60 s, then sleeps in small increments
    until the rolling sum + estimated_tokens fits under the threshold.
    """
    global _tpm_log
    threshold = int(TPM_LIMIT * TPM_MARGIN)  # 5 100 on free tier

    while True:
        now = time.time()
        # Drop entries outside the rolling window
        _tpm_log = [(t, n) for t, n in _tpm_log if now - t < TPM_WINDOW]
        used = sum(n for _, n in _tpm_log)

        if used + estimated_tokens <= threshold:
            break  # safe to proceed

        # How long until the oldest entry rolls off?
        if _tpm_log:
            oldest = _tpm_log[0][0]
            wait   = max(0.1, TPM_WINDOW - (now - oldest) + 0.2)
        else:
            wait = 1.0

        print(f"\n  [TPM throttle] {used}/{threshold} tokens used — "
              f"waiting {wait:.1f}s before next call...")
        time.sleep(wait)

def _record_tpm(token_count: int) -> None:
    """Log a completed call's token usage."""
    _tpm_log.append((time.time(), token_count))


def route_model(text_length, is_retry=False):
    if is_retry or text_length > 8000:
        return MODEL_STRONG
    return MODEL_FAST


def call_groq(prompt, model=None, is_retry=False, _retry_count=0):
    """
    Call Groq API with:
    - Disk response cache (no duplicate calls)
    - TPM-aware pre-call throttle (prevents 429 token/min errors)
    - Exponential back-off on 429 (handles burst spikes the throttle misses)
    - Model fallback: 70B → 8B on 429
    """
    if model is None:
        model = route_model(len(prompt), is_retry)

    key = _cache_key(model, prompt)
    if key in _cache:
        return _cache[key], True

    # Warn at RPD threshold
    used = _call_counts[model]
    if used >= RPD_LIMITS[model] * 0.85:
        print(f"\n  WARNING: {model} at {used}/{RPD_LIMITS[model]} RPD")

    # Pre-call TPM throttle
    est_tokens = _estimate_tokens(prompt)
    _tpm_throttle(est_tokens)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content

        # Record actual token usage if available, else use estimate
        actual_tokens = getattr(
            getattr(response, "usage", None), "total_tokens", None
        ) or (est_tokens + _estimate_tokens(result))
        _record_tpm(actual_tokens)

        _cache[key] = result
        _save_cache()
        _call_counts[model] += 1
        time.sleep(0.3)   # small courtesy gap between calls
        return result, False

    except Exception as e:
        err = str(e)
        is_rate_limit = "429" in err or "rate limit" in err.lower() or "rate_limit_exceeded" in err

        if is_rate_limit:
            # Parse wait time from error message if available
            wait_match = re.search(r"try again in (\d+(?:\.\d+)?)\s*m?s", err, re.I)
            if wait_match:
                raw_wait = float(wait_match.group(1))
                # Groq returns milliseconds in the message text
                wait_secs = raw_wait / 1000 if raw_wait > 10 else raw_wait
                wait_secs = max(1.0, wait_secs + 0.5)
            else:
                # Exponential back-off: 2s, 4s, 8s …  capped at 60s
                wait_secs = min(60.0, 2.0 ** (_retry_count + 1))

            print(f"\n  [429] {model} rate limited — waiting {wait_secs:.1f}s "
                  f"(retry {_retry_count + 1})...")
            time.sleep(wait_secs)

            # Model fallback: strong → fast on first 429
            if model == MODEL_STRONG and _retry_count == 0:
                print(f"  [429 fallback] switching to {MODEL_FAST}")
                return call_groq(prompt, model=MODEL_FAST,
                                 is_retry=is_retry, _retry_count=_retry_count + 1)

            # Retry same model (up to 3 times total)
            if _retry_count < 3:
                return call_groq(prompt, model=model,
                                 is_retry=is_retry, _retry_count=_retry_count + 1)

        raise e


# ── FIXED EXTRACTION PROMPT ───────────────────────────────────
# Key changes vs original:
#  A) Step counting: explicit OR-path minimum logic with disease-severity bypass
#  B) Phototherapy: explicit AND-list detection
#  C) Auth duration: "Unspecified" instruction for CPB-style payers
#  D) Quantity limits: tighter guard against dosing schedules
EXTRACTION_PROMPT = """\
You are a pharmaceutical policy analyst. Extract parameters for {BRAND} (Plaque Psoriasis/PsO) from the PA policy below.

Return ONLY valid JSON with exactly these keys:

{{
  "age": "Minimum age. Format '>= X'. If policy says 'FDA labelled age' or gives no number, write 'FDA labelled age'.",
  "step_therapy_requirements": "ALL step therapy language verbatim, including universal/across-all-indications criteria AND PsO-specific criteria. Include phototherapy language if present. If moderate-to-severe vs severe PsO differ, use moderate-to-severe only. Write 'None required' if none.",
  "num_steps_biologic": 0,
  "num_steps_conventional": 0,
  "phototherapy_required": "CRITICAL RULE — read carefully: 'Yes' ONLY if phototherapy is listed as a REQUIRED step in an AND-list (every item must be satisfied). 'No' if phototherapy is in an OR-list (patient picks one option from several) OR not mentioned. 'N/A' if drug excluded.",
  "tb_test_required": "Yes or No. Yes if TB test/screening is explicitly required.",
  "quantity_limits": "Only if the policy uses the exact phrase 'quantity limit' or 'quantity level limit'. Do NOT capture dosing schedules (mg at weeks, every N weeks). Write 'Per FDA label' if not explicitly stated.",
  "specialist_types": "Prescriber specialty restriction. 'Not specified' if absent.",
  "initial_auth_duration_months": "Number only (e.g. 6, 12). IMPORTANT: If the policy does NOT explicitly state an authorization duration in months, write 'Unspecified'. Do NOT infer or assume 12 months. CRITICAL DISTINCTION: a duration that appears INSIDE step therapy language (e.g., 'biologic for at least 3 months', 'trial of 3 months', '56-day trial') is a TRIAL duration, NOT the authorization period — do NOT use it here. Auth duration is stated separately as 'approved for X months' or 'authorization of X months'.",
  "reauth_duration_months": "Number only. 'Same as initial' if same. 'Unspecified' if not stated.",
  "reauth_required": "Yes or No.",
  "reauth_clinical_criteria": "Clinical criteria for renewal. 'Unspecified' if not stated.",
  "confidence": "High, Medium, or Low",
  "exclusion_flag": false
}}

STEP COUNTING RULES (read carefully before counting):
1. Identify ALL pathways that lead to approval (connected by OR at the top level).
2. For EACH pathway, count its branded steps and generic steps separately.
3. Choose the pathway with the FEWEST TOTAL STEPS — this is the minimum path.
4. Report branded and generic counts from that minimum path only.
5. KEY INSIGHT: If ANY pathway requires only disease severity documentation (e.g., ≥10% BSA, crucial body areas affected) with NO prior therapy requirement, then num_steps_biologic=0 AND num_steps_conventional=0 for that path. Use 0.
6. Phototherapy steps are NEVER counted in num_steps_biologic or num_steps_conventional regardless.
7. If the same step appears in ALL pathways (AND), it must be counted.

PHOTOTHERAPY RULE:
- AND-list = each item in a bullet/numbered list where ALL bullets must be satisfied → phototherapy_required = "Yes"
- OR-list = patient must satisfy ONE of several options → phototherapy_required = "No"
- Example of AND: "Patient must have failed: (1) topical agent AND (2) phototherapy AND (3) systemic" → Yes
- Example of OR: "Patient must have failed: phototherapy OR methotrexate OR cyclosporine" → No

EXCLUSION: If {BRAND} is explicitly non-covered/excluded: exclusion_flag=true, all text fields="N/A - Drug Excluded", all ints=0.

Focus ONLY on Plaque Psoriasis criteria. Ignore PsA, CD, UC unless stated to apply to all indications.
tb_test_required=Yes if TB screening is required or referenced via prescribing information.
num_steps_biologic and num_steps_conventional must be INTEGER values.
exclusion_flag must be boolean.

Policy text (brand: {BRAND}):
{TEXT}"""


FAILED_PARAMS = {
    "age": "Extraction Failed",
    "step_therapy_requirements": "Extraction Failed",
    "num_steps_biologic": 0,
    "num_steps_conventional": 0,
    "phototherapy_required": "No",
    "tb_test_required": "No",
    "quantity_limits": "Per FDA label",
    "specialist_types": "Not specified",
    "initial_auth_duration_months": "Unspecified",
    "reauth_duration_months": "Unspecified",
    "reauth_required": "No",
    "reauth_clinical_criteria": "Unspecified",
    "confidence": "Low",
    "exclusion_flag": False,
}

def parse_json_response(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.MULTILINE)
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def extract_parameters(fname, brand, text):
    if not text.strip():
        return dict(FAILED_PARAMS, confidence="Low")
    prompt = EXTRACTION_PROMPT.format(BRAND=brand, TEXT=text)
    model  = route_model(len(text))
    try:
        raw, from_cache = call_groq(prompt, model=model)
        params = parse_json_response(raw)
        if params is not None:
            params["_model"]      = model
            params["_from_cache"] = from_cache
            params["num_steps_biologic"]     = int(params.get("num_steps_biologic", 0) or 0)
            params["num_steps_conventional"] = int(params.get("num_steps_conventional", 0) or 0)
            return params
    except Exception as e:
        print(f"\n    API error ({fname}, {brand}): {e}")
    print(f"\n    Retrying with {MODEL_STRONG} for ({fname}, {brand})")
    try:
        raw, from_cache = call_groq(prompt, model=MODEL_STRONG, is_retry=True)
        params = parse_json_response(raw)
        if params is not None:
            params["_model"]      = MODEL_STRONG
            params["_from_cache"] = from_cache
            params["num_steps_biologic"]     = int(params.get("num_steps_biologic", 0) or 0)
            params["num_steps_conventional"] = int(params.get("num_steps_conventional", 0) or 0)
            return params
    except Exception as e2:
        print(f"\n    BOTH models failed ({fname}, {brand}): {e2}")
    return dict(FAILED_PARAMS)


def stage3_extract_all(jobs):
    print("\n" + "="*60)
    print("STAGE 3 — Groq API Parameter Extraction")
    print("="*60)
    _load_cache()
    extracted = {}
    cache_hits = 0
    for fname, brand, text in tqdm(jobs, desc="  Extracting"):
        params = extract_parameters(fname, brand, text)
        extracted[(fname, brand)] = params
        if params.get("_from_cache"):
            cache_hits += 1
    api_calls = sum(_call_counts.values())
    print(f"\n  Stage 3: {len(jobs)} jobs, {cache_hits} cache hits, {api_calls} API calls")
    return extracted


# ════════════════════════════════════════════════════════════════
# STAGE 4 — ACCESS QUALITY SCORE  ← KEY FIX #3
# ════════════════════════════════════════════════════════════════

def parse_months(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"not specified", "n/a - drug excluded", "extraction failed",
             "same as initial", "unspecified", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    for pat, divisor in [(r"(\d+(?:\.\d+)?)\s*month", 1),
                         (r"(\d+(?:\.\d+)?)\s*mo\b", 1),
                         (r"(\d+(?:\.\d+)?)\s*week", 4.33),
                         (r"(\d+(?:\.\d+)?)\s*day", 30.0),
                         (r"(\d+(?:\.\d+)?)\s*year", 1/12)]:
        m = re.search(pat, s)
        if m:
            return float(m.group(1)) / divisor
    return None


def _is_false_exclusion(params: dict) -> bool:
    """
    FIX F: Detect when LLM set exclusion_flag=True incorrectly.
    A genuine exclusion has ALL key text fields set to 'N/A - Drug Excluded'.
    If only secondary fields (e.g. reauth_criteria) say N/A while core fields
    (qty_limits, step_therapy, age) are normal, the exclusion flag is wrong.
    """
    qty  = str(params.get("quantity_limits", "")).lower()
    step = str(params.get("step_therapy_requirements", "")).lower()
    age  = str(params.get("age", "")).lower()
    na_marker = "n/a - drug excluded"
    na_count = sum(1 for v in [qty, step, age] if na_marker in v)
    return na_count == 0  # none of the three core fields are excluded → false exclusion


def compute_access_score(params):
    BD = {}

    excl = params.get("exclusion_flag", False)
    if (excl is True or str(excl).lower() == "true") and not _is_false_exclusion(params):
        return 0, {k: 0 for k in ["age","conv_steps","bio_steps",
                                    "step_duration","tb_test",
                                    "prescriber","initial_auth","reauth"]}
    if str(params.get("age", "")).startswith("N/A"):
        return 0, {k: 0 for k in ["age","conv_steps","bio_steps",
                                    "step_duration","tb_test",
                                    "prescriber","initial_auth","reauth"]}

    # 1. Age (10 pts)
    age_raw = str(params.get("age", ">= 18")).lower().replace(" ", "")
    if   any(x in age_raw for x in [">=6",">= 6"]):    BD["age"] = 6
    elif any(x in age_raw for x in [">=12",">= 12"]):  BD["age"] = 8
    elif any(x in age_raw for x in [">=18",">= 18"]):  BD["age"] = 10
    elif any(x in age_raw for x in [">18",">= 21",">=21",">21"]): BD["age"] = 5
    elif "not specified" in age_raw or "extraction" in age_raw: BD["age"] = 10
    else: BD["age"] = 7

    # 2. Conventional step therapy (20 pts)
    conv = int(params.get("num_steps_conventional", 0) or 0)
    BD["conv_steps"] = {0: 20, 1: 15, 2: 8}.get(conv, 0 if conv >= 3 else 15)

    # 3. Biologic step therapy (20 pts)
    bio = int(params.get("num_steps_biologic", 0) or 0)
    BD["bio_steps"] = {0: 20, 1: 12, 2: 5}.get(bio, 0 if bio >= 3 else 12)

    # 4. Step therapy duration (10 pts)
    # FIX: if no steps at all, award full 10 (no restriction)
    step_text = str(params.get("step_therapy_requirements", "")).lower()
    has_steps = (conv + bio) > 0
    if not has_steps:
        BD["step_duration"] = 10
    else:
        dur_months = None
        for pat, divisor in [(r"(\d+(?:\.\d+)?)\s*month", 1),
                              (r"(\d+(?:\.\d+)?)\s*mo\b", 1),
                              (r"(\d+(?:\.\d+)?)\s*week", 4.33)]:
            m = re.search(pat, step_text)
            if m:
                dur_months = float(m.group(1)) / divisor
                break
        if dur_months is None:    BD["step_duration"] = 10
        elif dur_months <= 3:     BD["step_duration"] = 10
        elif dur_months <= 6:     BD["step_duration"] = 5
        else:                     BD["step_duration"] = 0

    # 5. TB test (10 pts)
    tb = str(params.get("tb_test_required", "No")).strip().lower()
    BD["tb_test"] = 5 if tb == "yes" else 10

    # 6. Prescriber restriction (10 pts)
    presc = str(params.get("specialist_types", "Not specified")).lower()
    if any(x in presc for x in ["no restriction","not specified","any provider",
                                  "not mentioned","extraction failed"]):
        BD["prescriber"] = 10
    elif any(x in presc for x in ["preferred","recommended","consult"]):
        BD["prescriber"] = 7
    elif any(x in presc for x in ["only","required","must be","restricted to"]):
        BD["prescriber"] = 3
    else:
        BD["prescriber"] = 7

    # 7. Initial auth duration (10 pts)
    init_months = parse_months(params.get("initial_auth_duration_months", "Unspecified"))
    if   init_months is None:  BD["initial_auth"] = 7
    elif init_months >= 12:    BD["initial_auth"] = 10
    elif init_months >= 6:     BD["initial_auth"] = 5
    else:                      BD["initial_auth"] = 0

    # 8. Reauth duration (10 pts)
    reauth_val = str(params.get("reauth_duration_months", "Unspecified"))
    if "same as initial" in reauth_val.lower():
        BD["reauth"] = BD["initial_auth"]
    else:
        reauth_months = parse_months(reauth_val)
        if   reauth_months is None:  BD["reauth"] = 7
        elif reauth_months >= 12:    BD["reauth"] = 10
        elif reauth_months >= 6:     BD["reauth"] = 5
        else:                        BD["reauth"] = 0

    total = max(0, min(100, sum(BD.values())))
    return total, BD


def stage4_score_all(extracted):
    print("\n" + "="*60)
    print("STAGE 4 — Access Quality Score")
    print("="*60)
    scores = {}
    for (fname, brand), params in extracted.items():
        score, breakdown = compute_access_score(params)
        scores[(fname, brand)] = (score, breakdown)
    all_scores = [s for s, _ in scores.values()]
    print(f"  {len(scores)} jobs scored | mean={sum(all_scores)/len(all_scores):.1f} | "
          f"min={min(all_scores)} max={max(all_scores)}")
    return scores


# ════════════════════════════════════════════════════════════════
# STAGE 5 — OUTPUT ASSEMBLY  (unchanged except minor guard)
# ════════════════════════════════════════════════════════════════

def _normalise_age(age_str: str) -> str:
    """FIX C: Canonicalise age to '>= N' format, replacing unicode ≥ and stripping trailing words."""
    if not age_str or str(age_str).strip() in {"", "None", "nan", "Not specified", "Unspecified", "Extraction Failed"}:
        return "FDA labelled age"
    s = str(age_str).strip()
    s = s.replace("≥", ">=").replace("≤", "<=")
    m = re.search(r"([><=]+)\s*(\d+)", s)
    if m:
        op, num = m.group(1), m.group(2)
        if ">=" in op:
            return f">= {num}"
        elif ">" in op:
            return f"> {num}"
        elif "<=" in op:
            return f"<= {num}"
    if "fda" in s.lower() or "label" in s.lower():
        return "FDA labelled age"
    return s


def _is_trial_context_duration(step_text: str, auth_months) -> bool:
    """
    FIX E: Returns True if auth_months appears ONLY as a trial/failure
    duration inside step therapy text (e.g., 'for at least 3 months'),
    meaning it was NOT the authorization period.
    Only fires for short durations (<=4 months) to be conservative.
    """
    if auth_months is None or auth_months > 4:
        return False
    n = int(auth_months)
    step_lower = str(step_text).lower()
    trial_patterns = [
        rf"for\s+at\s+least\s+{n}\s*months?",
        rf"trial\s+of\s+(?:at\s+least\s+)?{n}\s*months?",
        rf"{n}\s*months?\s+trial",
        rf"{n}\s*months?\s+of\s+(?:therapy|treatment)",
        rf"(?:minimum|min\.?)\s+{n}\s*months?",
        rf"{n}\s*months?\s*\?",          # Oregon Medicaid Q-style "3 months?"
        rf"\(\s*{n}\s*\)\s*months?",     # "(3) months"
        rf"{n}\s*[-–]\s*month\s+trial",  # "3-month trial"
        rf"56\s*[-–]\s*day",             # 56-day = ~2 months
    ]
    # Only flag if at least one trial pattern matches AND no explicit auth statement
    auth_patterns = [
        rf"approv(?:al|ed)\s+(?:of|for)\s+(?:up\s+to\s+)?{n}\s*months?",
        rf"authorization\s+of\s+{n}\s*months?",
        rf"authorized\s+for\s+{n}\s*months?",
        rf"grant(?:ed)?\s+for\s+{n}\s*months?",
    ]
    has_trial = any(re.search(p, step_lower) for p in trial_patterns)
    has_auth  = any(re.search(p, step_lower) for p in auth_patterns)
    return has_trial and not has_auth


def assemble_dataframe(jobs, extracted, scores):
    rows = []
    for fname, brand, _ in jobs:
        params = extracted.get((fname, brand), FAILED_PARAMS)
        score, _ = scores.get((fname, brand), (0, {}))

        # FIX C: normalised age
        age_val = _normalise_age(params.get("age", "FDA labelled age"))

        step_req = params.get("step_therapy_requirements", "Unspecified")
        if not step_req or str(step_req).strip() in {"", "None", "nan", "Not specified"}:
            step_req = "Unspecified"

        num_brands   = int(params.get("num_steps_biologic", 0) or 0)
        num_generics = int(params.get("num_steps_conventional", 0) or 0)

        # FIX B + FIX I: phototherapy step overcounting correction.
        # Per business rules, phototherapy is NEVER counted in generic steps.
        # If photo=Yes AND the step text shows an AND-list structure where
        # phototherapy was one of several mandatory items, subtract 1 generic.
        # FIX I broadens the AND-list detection beyond numbered lists ((\d+))
        # to also catch prose AND-lists (e.g. Oregon Medicaid Q11 style):
        # condition fires if ≥3 AND conjunctions appear AND phototherapy is
        # not in an OR clause.
        photo_raw = str(params.get("phototherapy_required", "No")).strip().lower()
        if photo_raw in {"yes", "y"} and num_generics > 0:
            step_lower = str(step_req).lower()
            has_bullet_or_numbered = bool(re.search(r"\(\d+\)|•|\d+\.", step_lower))
            has_prose_and_list     = len(re.findall(r"\band\b", step_lower)) >= 3
            photo_not_in_or        = not re.search(
                r"phototherapy\s+or\b|\bor\s+phototherapy|\bor\b[^.]{0,30}phototherapy",
                step_lower
            )
            photo_and_generics = (
                "phototherapy" in step_lower
                and (has_bullet_or_numbered or has_prose_and_list)
                and photo_not_in_or
            )
            if photo_and_generics:
                num_generics = max(0, num_generics - 1)

        num_brands_display   = "NA" if num_brands == 0 else num_brands
        num_generics_display = "NA" if num_generics == 0 else num_generics

        # FIX G: pandas read_csv converts the string "N/A" to NaN by default
        # (it is in pandas built-in na_values list). To avoid NaN in the output
        # CSV, we never write "N/A" for this column. Business rules treat
        # "N/A" (no criteria at all) and "No" (not required) identically for
        # scoring, so "No" is the safe universal fallback.
        if photo_raw in {"yes", "y"}:
            photo_display = "Yes"
        else:
            photo_display = "No"   # covers "no", "n", "n/a", "na", None, NaN, anything else

        tb = str(params.get("tb_test_required", "No")).strip()
        tb_display = "Y" if tb.lower() in {"yes","y","true"} else "N"

        def clean_qty_limit(val):
            s = str(val).strip() if val is not None else ""
            # FIX D: NaN / None → default
            if not s or s.lower() in {"nan", "none", ""}:
                return "Per FDA label"
            s_lower = s.lower()
            if any(x in s_lower for x in ["per fda label","fda label",
                                           "not specified","unspecified","none"]):
                return "Per FDA label"
            dosing_signals = ["mg per dose","dose limit","maintenance dose",
                              "dosing is","dosing:","mg at weeks","mg every",
                              "every 12 weeks","every 8 weeks","at week"]
            qty_signals    = ["syringe","vial","pack","qty","quantity level",
                              "max","maximum","per day","per month","per 28","per 84"]
            if any(d in s_lower for d in dosing_signals):
                if not any(q in s_lower for q in qty_signals):
                    return "Per FDA label"
            return s
        qty_display = clean_qty_limit(params.get("quantity_limits", "Per FDA label"))

        spec_display = str(params.get("specialist_types", "Not specified")).strip()
        if not spec_display or spec_display in {"", "None", "nan"}:
            spec_display = "Not specified"

        # FIX E: trial-context auth duration guard
        init_raw = params.get("initial_auth_duration_months", "Unspecified")
        im = parse_months(str(init_raw))
        if im is not None and _is_trial_context_duration(step_req, im):
            im = None  # reset — this was a trial duration, not auth duration
        init_display = (f"{int(im)} Months" if im and im == int(im)
                        else f"{im:.1f} Months" if im else "Unspecified")

        reauth_raw = str(params.get("reauth_duration_months", "Unspecified"))
        if "same as initial" in reauth_raw.lower():
            reauth_display = "Same as initial"
        else:
            rm = parse_months(reauth_raw)
            reauth_display = (f"{int(rm)} Months" if rm and rm == int(rm)
                              else f"{rm:.1f} Months" if rm else "Unspecified")

        reauth_crit = str(params.get("reauth_clinical_criteria", "Unspecified")).strip()
        if not reauth_crit or reauth_crit in {"", "None", "nan", "Not specified"}:
            reauth_crit = "Unspecified"

        reauth_req = str(params.get("reauth_required", "No")).strip().lower()
        reauth_req_display = "Yes" if reauth_req in {"yes","y","true"} else "No"
        has_reauth_dur  = reauth_display.lower() not in {"unspecified","not specified","n/a","na","","none"}
        has_reauth_crit = reauth_crit.lower() not in {"unspecified","not specified","n/a","na","","none","failed"}
        if has_reauth_dur or has_reauth_crit:
            reauth_req_display = "Yes"

        rows.append({
            "Filename":   fname,
            "Brand":      brand,
            "Age":        age_val,
            "Step Therapy Requirements Documented in Policy": step_req,
            "Number of Steps through Brands":  num_brands_display,
            "Number of Steps through Generic": num_generics_display,
            "Step through-Phototherapy":       photo_display,
            "TB Test required":                tb_display,
            "Quantity Limits":                 qty_display,
            "Specialist Types":                spec_display,
            "Initial Authorization Duration(in-months)": init_display,
            "Reauthorization Duration(in-months)":       reauth_display,
            "Reauthorization Required":                  reauth_req_display,
            "Reauthorization Requirements Documented in Policy": reauth_crit,
            "Access Score": score,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def validate_dataframe(df, jobs):
    print("\n  Running validation checks...")
    passed = True
    if len(df) < len(jobs):
        print(f"  FAIL: Expected {len(jobs)} rows, got {len(df)}")
        passed = False
    else:
        print(f"  PASS: Row count {len(df)}")
    dupes = df.duplicated(subset=["Filename","Brand"])
    if dupes.any():
        print(f"  FAIL: {dupes.sum()} duplicate (Filename, Brand) pairs")
        passed = False
    else:
        print(f"  PASS: No duplicates")
    blank = df.isnull().sum().sum()
    if blank > 0:
        df.fillna("Not specified", inplace=True)
        print(f"  WARN: {blank} blank cells filled")
    bad = ((df["Access Score"] < 0) | (df["Access Score"] > 100)).sum()
    if bad:
        print(f"  FAIL: {bad} scores outside [0,100]")
        passed = False
    else:
        print(f"  PASS: All scores in [0,100]")
    return passed


def build_score_breakdown(jobs, scores):
    rows = []
    for fname, brand, _ in jobs:
        score, bd = scores.get((fname, brand), (0, {}))
        row = {"Filename": fname, "Brand": brand, "Total_Score": score}
        row.update(bd)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_dashboard(df):
    brand_avg = df.groupby("Brand")["Access Score"].mean().round(1).to_dict()
    score_dist = {}
    for s in df["Access Score"]:
        bucket = f"{(s//10)*10}-{(s//10)*10+9}"
        score_dist[bucket] = score_dist.get(bucket, 0) + 1

    tremfya_df = df[df["Brand"] == "TREMFYA"].sort_values("Access Score", ascending=False)
    stelara_df = df[df["Brand"] == "STELARA"].sort_values("Access Score", ascending=False)

    brand_colours = {
        "TREMFYA":"#6366f1","STELARA":"#06b6d4","AMJEVITA":"#f59e0b",
        "COSENTYX":"#10b981","ENBREL":"#ef4444","REMICADE":"#8b5cf6",
        "SILIQ":"#ec4899","CIMZIA":"#f97316","BIMZELX":"#14b8a6",
        "SKYRIZI":"#84cc16","OTEZLA":"#a78bfa","YESINTEK":"#fb923c",
        "OTULFI":"#38bdf8","ILUMYA":"#4ade80","ACITRETIN":"#f472b6",
    }

    brands_json     = json.dumps(list(brand_avg.keys()))
    brand_avgs_json = json.dumps(list(brand_avg.values()))
    brand_cols_json = json.dumps([brand_colours.get(b,"#94a3b8") for b in brand_avg.keys()])
    dist_labels     = json.dumps(sorted(score_dist.keys()))
    dist_values     = json.dumps([score_dist[k] for k in sorted(score_dist.keys())])

    heatmap_rows = ""
    for _, row in df.sort_values(["Brand","Access Score"],ascending=[True,False]).iterrows():
        score = int(row["Access Score"])
        colour = ("#22c55e" if score>=75 else "#f59e0b" if score>=50
                  else "#f97316" if score>=25 else "#ef4444")
        fname_short = str(row["Filename"]).replace(".pdf","")
        bc = brand_colours.get(str(row["Brand"]),"#94a3b8")
        heatmap_rows += f"""
        <tr>
          <td><span style="font-family:monospace;font-size:11px">{fname_short}</span></td>
          <td><span class="brand-badge" style="background:{bc}20;color:{bc};border:1px solid {bc}40">{row['Brand']}</span></td>
          <td>{row['Age']}</td>
          <td style="text-align:center">{row['Number of Steps through Generic']}</td>
          <td style="text-align:center">{row['Number of Steps through Brands']}</td>
          <td>{row['TB Test required']}</td>
          <td style="font-size:11px">{str(row['Initial Authorization Duration(in-months)'])[:20]}</td>
          <td>
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{score}%;background:{colour}"></div>
              <span class="score-label" style="color:{colour}">{score}</span>
            </div>
          </td>
        </tr>"""

    merged = pd.merge(
        tremfya_df[["Filename","Access Score"]].rename(columns={"Access Score":"TREMFYA_Score"}),
        stelara_df[["Filename","Access Score"]].rename(columns={"Access Score":"STELARA_Score"}),
        on="Filename", how="outer").fillna("-")
    comparison_rows = ""
    for _, r in merged.iterrows():
        t = r["TREMFYA_Score"]; s = r["STELARA_Score"]
        t_d = f"{int(t)}" if t != "-" else "-"
        s_d = f"{int(s)}" if s != "-" else "-"
        fn = str(r["Filename"]).replace(".pdf","")
        comparison_rows += f"<tr><td style='font-size:11px;font-family:monospace'>{fn}</td><td style='text-align:center;color:#6366f1;font-weight:600'>{t_d}</td><td style='text-align:center;color:#06b6d4;font-weight:600'>{s_d}</td></tr>"

    total_rows  = len(df)
    avg_score   = df["Access Score"].mean()
    excl_count  = (df["Access Score"] == 0).sum()
    high_access = (df["Access Score"] >= 75).sum()

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Payer Policy Intelligence Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0f1117;--surface:#1a1d2e;--surface2:#232640;--border:rgba(255,255,255,0.07);--text:#e2e8f0;--muted:#94a3b8;--primary:#6366f1;--accent:#06b6d4;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:0 24px 60px}}
.header{{padding:48px 0 32px;border-bottom:1px solid var(--border);margin-bottom:40px}}
.header h1{{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,var(--primary),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:40px}}
.kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px}}
.kpi-label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
.kpi-value{{font-size:2.4rem;font-weight:800;line-height:1}}
.charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:40px}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px}}
.chart-title{{font-size:14px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:20px}}
.chart-wrap{{position:relative;height:280px}}
.table-card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:32px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:10px 14px;color:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle}}
.brand-badge{{padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}}
.score-bar-wrap{{display:flex;align-items:center;gap:10px;min-width:120px}}
.score-bar{{height:8px;border-radius:999px;flex-shrink:0}}
.score-label{{font-weight:700;font-size:13px;white-space:nowrap}}
input.search-box{{background:var(--surface2);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px 14px;font-size:13px;width:260px;outline:none;margin-bottom:12px}}
.filter-btn{{padding:6px 16px;border-radius:999px;border:1px solid var(--border);background:var(--surface2);color:var(--muted);font-size:12px;font-weight:600;cursor:pointer;margin-right:8px;margin-bottom:12px}}
.filter-btn.active,.filter-btn:hover{{background:var(--primary);color:#fff;border-color:var(--primary)}}
</style></head><body>
<div class="header"><h1>Payer Policy Intelligence</h1><p style="color:var(--muted);margin-top:8px">Access Quality Score Dashboard — Pipeline v3</p></div>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Policies Processed</div><div class="kpi-value" style="color:#6366f1">{total_rows}</div></div>
  <div class="kpi-card"><div class="kpi-label">Avg Access Score</div><div class="kpi-value" style="color:#06b6d4">{avg_score:.1f}</div></div>
  <div class="kpi-card"><div class="kpi-label">High Access (≥75)</div><div class="kpi-value" style="color:#22c55e">{high_access}</div></div>
  <div class="kpi-card"><div class="kpi-label">Excluded / Zero</div><div class="kpi-value" style="color:#ef4444">{excl_count}</div></div>
</div>
<div class="charts-grid">
  <div class="chart-card"><div class="chart-title">Avg Score by Brand</div><div class="chart-wrap"><canvas id="brandChart"></canvas></div></div>
  <div class="chart-card"><div class="chart-title">Score Distribution</div><div class="chart-wrap"><canvas id="distChart"></canvas></div></div>
</div>
<div class="charts-grid">
  <div class="chart-card"><div class="chart-title">TREMFYA Scores</div><div class="chart-wrap"><canvas id="tremfyaChart"></canvas></div></div>
  <div class="chart-card"><div class="chart-title">STELARA Scores</div><div class="chart-wrap"><canvas id="stelaraChart"></canvas></div></div>
</div>
<div class="table-card">
  <div class="chart-title">Policy Heatmap</div>
  <input class="search-box" id="searchBox" placeholder="Search filename or brand..." oninput="filterTable()">
  <button class="filter-btn active" onclick="filterBrand('ALL',this)">All</button>
  <button class="filter-btn" onclick="filterBrand('TREMFYA',this)">TREMFYA</button>
  <button class="filter-btn" onclick="filterBrand('STELARA',this)">STELARA</button>
  <button class="filter-btn" onclick="filterBrand('OTHER',this)">Other</button>
  <table id="heatmapTable">
    <thead><tr><th>Filename</th><th>Brand</th><th>Age</th><th>Conv.Steps</th><th>Bio.Steps</th><th>TB</th><th>Init.Auth</th><th style="min-width:160px">Score</th></tr></thead>
    <tbody id="heatmapBody">{heatmap_rows}</tbody>
  </table>
</div>
<div class="table-card">
  <div class="chart-title">TREMFYA vs STELARA</div>
  <table><thead><tr><th>Filename</th><th style="color:#6366f1;text-align:center">TREMFYA</th><th style="color:#06b6d4;text-align:center">STELARA</th></tr></thead>
  <tbody>{comparison_rows}</tbody></table>
</div>
<script>
const cd={{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.04)'}}}},y:{{ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.04)'}},beginAtZero:true}}}}}};
new Chart(document.getElementById('brandChart'),{{type:'bar',data:{{labels:{brands_json},datasets:[{{data:{brand_avgs_json},backgroundColor:{brand_cols_json},borderRadius:6}}]}},options:{{...cd,scales:{{...cd.scales,y:{{...cd.scales.y,max:100}}}}}}}});
new Chart(document.getElementById('distChart'),{{type:'doughnut',data:{{labels:{dist_labels},datasets:[{{data:{dist_values},backgroundColor:['#ef4444','#f97316','#f59e0b','#84cc16','#22c55e','#06b6d4','#6366f1','#8b5cf6','#ec4899','#14b8a6'],borderWidth:0}}]}},options:{{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{{legend:{{display:true,position:'right',labels:{{color:'#94a3b8',font:{{size:11}}}}}}}}}}}});
new Chart(document.getElementById('tremfyaChart'),{{type:'bar',data:{{labels:{json.dumps(tremfya_df['Access Score'].tolist())}.map((_,i)=>`Doc ${{i+1}}`),datasets:[{{data:{json.dumps(tremfya_df['Access Score'].tolist())},backgroundColor:'#6366f1',borderRadius:4}}]}},options:{{...cd,scales:{{...cd.scales,y:{{...cd.scales.y,max:100}}}}}}}});
new Chart(document.getElementById('stelaraChart'),{{type:'bar',data:{{labels:{json.dumps(stelara_df['Access Score'].tolist())}.map((_,i)=>`Doc ${{i+1}}`),datasets:[{{data:{json.dumps(stelara_df['Access Score'].tolist())},backgroundColor:'#06b6d4',borderRadius:4}}]}},options:{{...cd,scales:{{...cd.scales,y:{{...cd.scales.y,max:100}}}}}}}});
let activeBrand='ALL';
function filterBrand(b,btn){{activeBrand=b;document.querySelectorAll('.filter-btn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');filterTable();}}
function filterTable(){{const q=document.getElementById('searchBox').value.toLowerCase();document.querySelectorAll('#heatmapBody tr').forEach(row=>{{const txt=row.textContent.toLowerCase();const bc=row.cells[1]?.textContent.trim();const bm=activeBrand==='ALL'||(activeBrand==='OTHER'?!['TREMFYA','STELARA'].includes(bc):bc===activeBrand);row.style.display=(bm&&txt.includes(q))?'':'none';}});}}
</script></body></html>"""

    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    print(f"  Dashboard -> {DASHBOARD_HTML}")


def stage5_output(jobs, extracted, scores):
    print("\n" + "="*60)
    print("STAGE 5 — Output Assembly")
    print("="*60)
    df    = assemble_dataframe(jobs, extracted, scores)
    df_bd = build_score_breakdown(jobs, scores)
    validate_dataframe(df, jobs)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8", na_rep="")  # FIX H: na_rep prevents Python None → "nan" text
    df_bd.to_csv(SCORE_CSV, index=False, encoding="utf-8", na_rep="")
    print(f"\n  result.csv  -> {RESULT_CSV}")
    print(f"  score.csv   -> {SCORE_CSV}")
    generate_dashboard(df)
    print(f"\n  Brand summary:")
    bs = df.groupby("Brand")["Access Score"].agg(["mean","min","max","count"])
    for brand, r in bs.sort_values("mean", ascending=False).iterrows():
        print(f"    {brand:15s}: mean={r['mean']:.1f} min={r['min']} max={r['max']} n={int(r['count'])}")
    return df


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def run_pipeline():
    print("\n" + "="*60)
    print("  PAYER POLICY INTELLIGENCE PIPELINE — v3 FIXED")
    print("="*60)
    raw_texts = stage1_ingest_pdfs()
    jobs      = stage2_build_jobs(raw_texts)
    extracted = stage3_extract_all(jobs)
    scores    = stage4_score_all(extracted)
    result_df = stage5_output(jobs, extracted, scores)
    print("\n" + "="*60)
    print(f"  DONE | result.csv -> {RESULT_CSV}")
    print("="*60 + "\n")
    return result_df

if __name__ == "__main__":
    run_pipeline()
