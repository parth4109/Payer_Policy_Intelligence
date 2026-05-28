"""
==============================================================
Payer Policy Intelligence Pipeline
Hackathon Submission — Full End-to-End Script
==============================================================

Stages:
  1. PDF Ingestion (pdfplumber + PyMuPDF fallback)
  2. Job list from Submissions sheet (ground-truth template)
  3. Brand text segmentation
  4. Groq API extraction (8B bulk / 70B retry) + caching
  5. Access Quality Score computation
  6. Output assembly → result.csv + dashboard.html

Models (final submission):
  - llama-3.1-8b-instant   (bulk, 14,400 RPD)
  - llama-3.3-70b-versatile (complex/retry, 1,000 RPD)
"""

import os, sys, json, re, time, hashlib, warnings
from pathlib import Path

import pandas as pd
import pdfplumber
import fitz          # PyMuPDF
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR        = Path(r"C:\Users\Parth Chauhan\Desktop\RAG_Project")
DATA_DIR        = BASE_DIR / "Data"
OUTPUT_DIR      = BASE_DIR / "outputs"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"
BUSINESS_RULES  = BASE_DIR / "PA_Business_rules_pc.xlsx"

OUTPUT_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

CACHE_FILE      = OUTPUT_DIR / "extraction_log.json"
RESULT_CSV      = OUTPUT_DIR / "result.csv"
SCORE_CSV       = OUTPUT_DIR / "score_breakdown.csv"
ERROR_LOG       = OUTPUT_DIR / "ingestion_errors.log"
DASHBOARD_HTML  = OUTPUT_DIR / "dashboard.html"

# ─── Models ──────────────────────────────────────────────────
MODEL_FAST   = "llama-3.1-8b-instant"      # 14,400 RPD
MODEL_STRONG = "llama-3.3-70b-versatile"   # 1,000 RPD

# ─── API Key ─────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Run: $env:GROQ_API_KEY='gsk_...'"
        " or add it to a .env file in the project root."
    )

client = Groq(api_key=GROQ_API_KEY)

# ─── Brand keyword registry ───────────────────────────────────
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

# ─── Output column names (match Submissions sheet exactly) ───
OUTPUT_COLUMNS = [
    "Filename",
    "Brand",
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]


# ════════════════════════════════════════════════════════════════
# STAGE 1 — PDF INGESTION
# ════════════════════════════════════════════════════════════════

def extract_text_pdfplumber(pdf_path: Path) -> str | None:
    """Extract full text using pdfplumber (handles tables, columns)."""
    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if not text.strip():
                    # Try extracting words directly as fallback within pdfplumber
                    words = page.extract_words()
                    text = " ".join(w["text"] for w in words)
                pages.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
        return "\n".join(pages)
    except Exception:
        return None


def extract_text_fitz(pdf_path: Path) -> str | None:
    """Extract text using PyMuPDF (better for scanned/complex layouts)."""
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


def stage1_ingest_pdfs() -> dict:
    """
    Stage 1: Ingest all PDFs from DATA_DIR.
    Returns raw_texts: {filename: full_text}
    """
    print("\n" + "="*60)
    print("STAGE 1 — PDF Ingestion")
    print("="*60)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {DATA_DIR}")

    raw_texts = {}
    errors = []

    for pdf_path in tqdm(pdf_files, desc="  Ingesting"):
        fname = pdf_path.name

        # Primary: pdfplumber
        text = extract_text_pdfplumber(pdf_path)

        # Fallback: PyMuPDF if text too short
        if text is None or len(text.strip()) < 200:
            text = extract_text_fitz(pdf_path)

        if text is None or len(text.strip()) < 50:
            errors.append(fname)
            text = ""
            print(f"  WARNING: Very short/empty extraction for {fname}")

        raw_texts[fname] = text

    if errors:
        ERROR_LOG.write_text("\n".join(errors), encoding="utf-8")
        print(f"\n  {len(errors)} PDFs had extraction issues -> {ERROR_LOG}")

    ok = len(raw_texts) - len(errors)
    print(f"\n  Stage 1 complete: {ok}/{len(pdf_files)} PDFs extracted cleanly")
    return raw_texts


# ════════════════════════════════════════════════════════════════
# STAGE 2 — JOB LIST + BRAND SEGMENTATION
# ════════════════════════════════════════════════════════════════

def load_jobs_from_submissions() -> list[tuple[str, str]]:
    """
    Load (Filename, Brand) pairs from the Submissions sheet.
    This is the ground-truth job template provided by the hackathon.
    """
    df = pd.read_excel(BUSINESS_RULES, sheet_name="Submissions")
    jobs = []
    for _, row in df.iterrows():
        fname = str(row["Filename"]).strip()
        brand = str(row["Brand"]).strip()
        if fname and brand and fname != "nan" and brand != "nan":
            jobs.append((fname, brand))
    return jobs


def segment_text_for_brand(full_text: str, brand: str, max_chars: int = 9000) -> str:
    """
    Extract the most brand-relevant portion of the document text.
    Strategy:
      1. Look for explicit brand section headers → extract that block
      2. If no clear section, score paragraphs by keyword density → top chunks
      3. Final fallback: first max_chars of full text
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

    # ── Step 1: Find section headers for this brand ──
    header_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        line_lower = stripped.lower()
        # A header: contains brand keyword, short line, possibly all-caps or title
        if any(kw in line_lower for kw in keywords) and len(stripped) < 120:
            # Not just a footnote reference (very short, just the word)
            if len(stripped) > 5:
                header_indices.append(i)

    if header_indices:
        # Extract text from first brand header to next major section or end
        start = header_indices[0]
        end = n
        for j in range(start + 5, n):
            stripped = lines[j].strip()
            line_lower = stripped.lower()
            # Stop if we hit a clear header for another brand
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
            # Take first max_chars chars of the segment (most relevant)
            segment = segment[:max_chars]
        if len(segment.strip()) > 200:
            return segment

    # ── Step 2: Paragraph scoring by keyword density ──
    paragraphs = re.split(r"\n{2,}", full_text)
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(para_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, para))

    if scored:
        scored.sort(key=lambda x: -x[0])
        # Take top paragraphs up to max_chars
        collected = []
        total_len = 0
        for _, para in scored:
            if total_len + len(para) > max_chars:
                break
            collected.append(para)
            total_len += len(para)
        if collected:
            return "\n\n".join(collected)

    # ── Step 3: Full text fallback ──
    return full_text[:max_chars]


def stage2_build_jobs(raw_texts: dict) -> list[tuple[str, str, str]]:
    """
    Stage 2: Load job list from Submissions sheet and segment text per brand.
    Returns jobs: [(filename, brand, text_segment), ...]
    """
    print("\n" + "="*60)
    print("STAGE 2 — Brand Detection & Job Segmentation")
    print("="*60)

    jobs_raw = load_jobs_from_submissions()
    print(f"  Loaded {len(jobs_raw)} (Filename, Brand) pairs from Submissions sheet")

    jobs = []
    missing_pdfs = []

    for fname, brand in jobs_raw:
        if fname not in raw_texts:
            print(f"  WARNING: PDF not found in raw_texts: {fname}")
            missing_pdfs.append((fname, brand))
            jobs.append((fname, brand, ""))
            continue

        full_text = raw_texts[fname]
        segment = segment_text_for_brand(full_text, brand)
        jobs.append((fname, brand, segment))

    if missing_pdfs:
        print(f"\n  {len(missing_pdfs)} PDFs referenced in Submissions but not found in Data/")

    # Print brand distribution
    brand_counts = {}
    for _, brand, _ in jobs:
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    print("\n  Brand distribution:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"    {brand:15s}: {count} job(s)")

    print(f"\n  Stage 2 complete: {len(jobs)} extraction jobs ready")
    return jobs


# ════════════════════════════════════════════════════════════════
# STAGE 3 — GROQ API EXTRACTION
# ════════════════════════════════════════════════════════════════

# ── Cache ────────────────────────────────────────────────────
_cache: dict = {}

def _load_cache():
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            print(f"  Loaded {len(_cache)} cached responses from {CACHE_FILE.name}")
        except Exception:
            _cache = {}
    else:
        _cache = {}

def _save_cache():
    CACHE_FILE.write_text(
        json.dumps(_cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
    return f"{model}::{h}"


# ── Model routing ────────────────────────────────────────────
_call_counts = {MODEL_FAST: 0, MODEL_STRONG: 0}
RPD_LIMITS   = {MODEL_FAST: 14400, MODEL_STRONG: 1000}

def route_model(text_length: int, is_retry: bool = False) -> str:
    """Route to 70B for long/complex docs or retries; 8B for everything else."""
    if is_retry or text_length > 8000:
        return MODEL_STRONG
    return MODEL_FAST


def call_groq(prompt: str, model: str = None, is_retry: bool = False) -> tuple[str, bool]:
    """
    Call Groq API with disk caching.
    Returns (response_text, from_cache).
    """
    if model is None:
        model = route_model(len(prompt), is_retry)

    key = _cache_key(model, prompt)
    if key in _cache:
        return _cache[key], True

    # Rate limit warning
    used = _call_counts[model]
    limit = RPD_LIMITS[model]
    if used >= limit * 0.85:
        print(f"\n  WARNING: {model} at {used}/{limit} RPD ({100*used/limit:.0f}%)")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    result = response.choices[0].message.content
    _cache[key] = result
    _save_cache()
    _call_counts[model] += 1
    time.sleep(0.5)   # stay within rate limits
    return result, False


# ── Extraction prompt ────────────────────────────────────────
EXTRACTION_PROMPT = """\
You are a pharmaceutical policy analyst specializing in payer Prior Authorization (PA) policies.

Extract clinical parameters for drug brand {BRAND} (indication: Plaque Psoriasis / PsO) from the PA policy text below.

Return ONLY a valid JSON object with EXACTLY these keys (no extra keys, no markdown fences):

{{
  "age": "Minimum age eligibility. Format: '>= X' (e.g., '>= 18', '>= 6'). Default '>= 18' if not stated.",
  "step_therapy_requirements": "Full description of ALL step therapy agents (conventional AND biologic) required before {BRAND} approval for PsO. Include agent names, count, AND trial duration. Example: '1 conventional systemic (methotrexate, cyclosporine, or acitretin) for >= 3 months AND trial and failure of 1 biologic'. Write 'None required' if no step therapy.",
  "num_steps_biologic": 0,
  "num_steps_conventional": 0,
  "phototherapy_required": "Yes or No — is phototherapy explicitly a required step therapy option?",
  "tb_test_required": "Yes or No ONLY. Yes if TB test is explicitly required OR referenced via prescribing information (e.g., 'per prescribing information').",
  "quantity_limits": "Any quantity/supply limits stated (e.g., '1 syringe per 56 days'). Write 'Per FDA label' if not stated.",
  "specialist_types": "Who can prescribe (e.g., 'Dermatologist', 'Dermatologist or Rheumatologist', 'No restriction'). Write 'Not specified' if not mentioned.",
  "initial_auth_duration_months": "Initial approval period as a number of months only (e.g., 6, 12). Write 'Not specified' if not stated.",
  "reauth_duration_months": "Renewal period as a number of months only (e.g., 12). Write 'Same as initial' if same. Write 'Not specified' if not stated.",
  "reauth_required": "Yes or No — is reauthorization required?",
  "reauth_clinical_criteria": "Key clinical criteria for renewal (e.g., 'Documentation of positive clinical response evidenced by reduction in BSA or improvement in symptoms'). Write 'Not specified' if not stated.",
  "confidence": "Your extraction confidence: High, Medium, or Low",
  "exclusion_flag": false
}}

Critical rules:
- num_steps_biologic and num_steps_conventional must be INTEGER values (0, 1, 2, 3 — not strings)
- exclusion_flag must be boolean true or false (not a string)
- If {BRAND} is explicitly excluded/non-covered: set exclusion_flag=true and all text fields to "N/A - Drug Excluded", all integers to 0
- Focus ONLY on Plaque Psoriasis (PsO) criteria; ignore PsA, CD, UC unless stated to apply to ALL indications
- For step therapy with OR conditions, count ONLY the minimum required path
- tb_test_required = Yes if text says "per prescribing information" for TB screening (STELARA/TREMFYA prescribing info requires TB testing)
- Return ONLY the JSON object — no explanation, no markdown, no preamble

Policy text (brand: {BRAND}):
{TEXT}"""


def parse_json_response(raw: str) -> dict | None:
    """Robustly parse JSON from LLM response, stripping markdown if present."""
    s = raw.strip()
    # Remove markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.MULTILINE)
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


FAILED_PARAMS = {
    "age": "Extraction Failed",
    "step_therapy_requirements": "Extraction Failed",
    "num_steps_biologic": 0,
    "num_steps_conventional": 0,
    "phototherapy_required": "No",
    "tb_test_required": "No",
    "quantity_limits": "Per FDA label",
    "specialist_types": "Not specified",
    "initial_auth_duration_months": "Not specified",
    "reauth_duration_months": "Not specified",
    "reauth_required": "No",
    "reauth_clinical_criteria": "Not specified",
    "confidence": "Low",
    "exclusion_flag": False,
}


def extract_parameters(fname: str, brand: str, text: str) -> dict:
    """Extract all parameters for a single (filename, brand) job."""
    if not text.strip():
        return dict(FAILED_PARAMS, confidence="Low")

    prompt = EXTRACTION_PROMPT.format(BRAND=brand, TEXT=text)
    model  = route_model(len(text))

    # First attempt
    try:
        raw, from_cache = call_groq(prompt, model=model)
        params = parse_json_response(raw)
        if params is not None:
            params["_model"]      = model
            params["_from_cache"] = from_cache
            # Enforce integer types
            params["num_steps_biologic"]    = int(params.get("num_steps_biologic",    0) or 0)
            params["num_steps_conventional"] = int(params.get("num_steps_conventional", 0) or 0)
            return params
    except Exception as e:
        print(f"\n    API error ({fname}, {brand}): {e}")

    # Retry with stronger model
    print(f"\n    Retrying with {MODEL_STRONG} for ({fname}, {brand})")
    try:
        raw, from_cache = call_groq(prompt, model=MODEL_STRONG, is_retry=True)
        params = parse_json_response(raw)
        if params is not None:
            params["_model"]      = MODEL_STRONG
            params["_from_cache"] = from_cache
            params["num_steps_biologic"]    = int(params.get("num_steps_biologic",    0) or 0)
            params["num_steps_conventional"] = int(params.get("num_steps_conventional", 0) or 0)
            return params
    except Exception as e2:
        print(f"\n    BOTH models failed for ({fname}, {brand}): {e2}")

    return dict(FAILED_PARAMS)


def stage3_extract_all(jobs: list) -> dict:
    """
    Stage 3: Run Groq extraction for all jobs.
    Returns extracted: {(filename, brand): params_dict}
    """
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

    total = len(jobs)
    api_calls = sum(_call_counts.values())
    print(f"\n  Stage 3 complete:")
    print(f"    Total jobs      : {total}")
    print(f"    Cache hits      : {cache_hits}")
    print(f"    API calls made  : {api_calls} ({_call_counts[MODEL_FAST]} fast / {_call_counts[MODEL_STRONG]} strong)")
    return extracted


# ════════════════════════════════════════════════════════════════
# STAGE 4 — ACCESS QUALITY SCORE
# ════════════════════════════════════════════════════════════════

def parse_months(value: str | int | float) -> float | None:
    """Convert any duration string to months. Returns None if unknown."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"not specified", "n/a - drug excluded", "extraction failed", "same as initial", ""}:
        return None
    try:
        return float(s)   # already a number (months)
    except ValueError:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*(month|mo\b)", s)
    if m: return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(week|wk)", s)
    if m: return float(m.group(1)) / 4.33
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day)", s)
    if m: return float(m.group(1)) / 30.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|yr)", s)
    if m: return float(m.group(1)) * 12.0
    return None


def compute_access_score(params: dict) -> tuple[int, dict]:
    """
    Compute Access Quality Score (0-100) per rubric.
    Returns (total_score, breakdown_dict).
    """
    BD = {}

    # ── Hard exclusion override ───────────────────────────────
    excl = params.get("exclusion_flag", False)
    if excl is True or str(excl).lower() == "true":
        return 0, {k: 0 for k in ["age", "conv_steps", "bio_steps",
                                    "step_duration", "tb_test",
                                    "prescriber", "initial_auth", "reauth"]}

    # Check if all fields are "N/A - Drug Excluded"
    if str(params.get("age", "")).startswith("N/A"):
        return 0, {k: 0 for k in ["age", "conv_steps", "bio_steps",
                                    "step_duration", "tb_test",
                                    "prescriber", "initial_auth", "reauth"]}

    # 1. Age (10 pts)
    age_raw = str(params.get("age", ">= 18")).lower().replace(" ", "")
    if   any(x in age_raw for x in [">=6", ">= 6"]):   BD["age"] = 6
    elif any(x in age_raw for x in [">=12", ">= 12"]):  BD["age"] = 8
    elif any(x in age_raw for x in [">=18", ">= 18"]): BD["age"] = 10
    elif any(x in age_raw for x in [">18", "> 18", ">=21", ">= 21", ">21", "> 21"]): BD["age"] = 5
    elif "not specified" in age_raw or "extraction" in age_raw: BD["age"] = 10
    else: BD["age"] = 7

    # 2. Conventional step therapy (20 pts)
    conv = int(params.get("num_steps_conventional", 0) or 0)
    BD["conv_steps"] = {0: 20, 1: 15, 2: 8}.get(conv, 0 if conv >= 3 else 15)

    # 3. Biologic step therapy (20 pts)
    bio = int(params.get("num_steps_biologic", 0) or 0)
    BD["bio_steps"] = {0: 20, 1: 12, 2: 5}.get(bio, 0 if bio >= 3 else 12)

    # 4. Step therapy duration (10 pts)
    # Derive from step_therapy_requirements string
    step_text = str(params.get("step_therapy_requirements", "Not specified")).lower()
    dur_months = None
    for pattern in [
        r"(\d+(?:\.\d+)?)\s*month", r"(\d+(?:\.\d+)?)\s*mo\b",
        r"(\d+(?:\.\d+)?)\s*week", r"(\d+(?:\.\d+)?)\s*wk",
    ]:
        m = re.search(pattern, step_text)
        if m:
            val = float(m.group(1))
            if "week" in pattern or "wk" in pattern:
                val /= 4.33
            dur_months = val
            break

    if dur_months is None:  BD["step_duration"] = 10   # no restriction stated
    elif dur_months <= 3:   BD["step_duration"] = 10
    elif dur_months <= 6:   BD["step_duration"] = 5
    else:                   BD["step_duration"] = 0

    # 5. TB test (10 pts)
    tb = str(params.get("tb_test_required", "No")).strip().lower()
    BD["tb_test"] = 5 if tb == "yes" else 10

    # 6. Prescriber restriction (10 pts)
    presc = str(params.get("specialist_types", "Not specified")).lower()
    if any(x in presc for x in ["no restriction", "not specified", "any provider",
                                  "not mentioned", "extraction failed"]):
        BD["prescriber"] = 10
    elif any(x in presc for x in ["preferred", "recommended", "consult"]):
        BD["prescriber"] = 7
    elif any(x in presc for x in ["only", "required", "must be", "restricted to"]):
        BD["prescriber"] = 3
    else:
        BD["prescriber"] = 7

    # 7. Initial auth duration (10 pts)
    init_val = params.get("initial_auth_duration_months", "Not specified")
    init_months = parse_months(init_val)
    if   init_months is None:  BD["initial_auth"] = 7
    elif init_months >= 12:    BD["initial_auth"] = 10
    elif init_months >= 6:     BD["initial_auth"] = 5
    else:                      BD["initial_auth"] = 0

    # 8. Reauth duration (10 pts)
    reauth_val = str(params.get("reauth_duration_months", "Not specified"))
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


def stage4_score_all(extracted: dict) -> dict:
    """
    Stage 4: Compute access scores for all extracted results.
    Returns scores: {(filename, brand): (score, breakdown)}
    """
    print("\n" + "="*60)
    print("STAGE 4 — Access Quality Score Computation")
    print("="*60)

    scores = {}
    for (fname, brand), params in extracted.items():
        score, breakdown = compute_access_score(params)
        scores[(fname, brand)] = (score, breakdown)

    all_scores = [s for s, _ in scores.values()]
    print(f"  Scores computed for {len(scores)} jobs")
    print(f"  Mean score : {sum(all_scores)/len(all_scores):.1f}")
    print(f"  Min / Max  : {min(all_scores)} / {max(all_scores)}")
    excl = sum(1 for s, _ in scores.values() if s == 0)
    print(f"  Excluded (score=0): {excl}")
    return scores


# ════════════════════════════════════════════════════════════════
# STAGE 5 — OUTPUT ASSEMBLY, VALIDATION & DASHBOARD
# ════════════════════════════════════════════════════════════════

def assemble_dataframe(jobs: list, extracted: dict, scores: dict) -> pd.DataFrame:
    """Build final result DataFrame with exact Submissions-sheet column names."""
    rows = []
    for fname, brand, _ in jobs:
        params = extracted.get((fname, brand), FAILED_PARAMS)
        score, _ = scores.get((fname, brand), (0, {}))

        # Reauth duration display
        reauth_raw = str(params.get("reauth_duration_months", "Not specified"))
        if "same as initial" in reauth_raw.lower():
            reauth_display = reauth_raw
        else:
            rm = parse_months(reauth_raw)
            reauth_display = f"{int(rm)} months" if rm and rm == int(rm) else (f"{rm:.1f} months" if rm else reauth_raw)

        # Initial auth display
        init_raw = params.get("initial_auth_duration_months", "Not specified")
        im = parse_months(str(init_raw))
        init_display = f"{int(im)} months" if im and im == int(im) else (f"{im:.1f} months" if im else str(init_raw))

        rows.append({
            "Filename":   fname,
            "Brand":      brand,
            "Age":        params.get("age", ">= 18"),
            "Step Therapy Requirements Documented in Policy":
                params.get("step_therapy_requirements", "Not specified"),
            "Number of Steps through Brands":
                params.get("num_steps_biologic", 0),
            "Number of Steps through Generic":
                params.get("num_steps_conventional", 0),
            "Step through-Phototherapy":
                params.get("phototherapy_required", "No"),
            "TB Test required":
                params.get("tb_test_required", "No"),
            "Quantity Limits":
                params.get("quantity_limits", "Per FDA label"),
            "Specialist Types":
                params.get("specialist_types", "Not specified"),
            "Initial Authorization Duration(in-months)": init_display,
            "Reauthorization Duration(in-months)":       reauth_display,
            "Reauthorization Required":
                params.get("reauth_required", "Yes"),
            "Reauthorization Requirements Documented in Policy":
                params.get("reauth_clinical_criteria", "Not specified"),
            "Access Score": score,
        })

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def validate_dataframe(df: pd.DataFrame, jobs: list) -> bool:
    """Run validation assertions before export."""
    print("\n  Running validation checks...")
    passed = True

    # 1. Row count >= jobs count
    if len(df) < len(jobs):
        print(f"  FAIL: Expected {len(jobs)} rows, got {len(df)}")
        passed = False
    else:
        print(f"  PASS: Row count {len(df)} >= {len(jobs)} jobs")

    # 2. No duplicate (Filename, Brand)
    dupes = df.duplicated(subset=["Filename", "Brand"])
    if dupes.any():
        print(f"  FAIL: {dupes.sum()} duplicate (Filename, Brand) pairs found")
        passed = False
    else:
        print(f"  PASS: No duplicate (Filename, Brand) pairs")

    # 3. No blank cells
    blank_counts = df.isnull().sum()
    total_blanks = blank_counts.sum()
    if total_blanks > 0:
        print(f"  WARN: {total_blanks} blank cells found — filling with defaults")
        df.fillna("Not specified", inplace=True)
    else:
        print(f"  PASS: No blank cells")

    # 4. Access Score in [0, 100]
    score_col = df["Access Score"]
    bad_scores = ((score_col < 0) | (score_col > 100)).sum()
    if bad_scores > 0:
        print(f"  FAIL: {bad_scores} scores outside [0, 100]")
        passed = False
    else:
        print(f"  PASS: All scores in [0, 100]")

    # 5. Summary
    print(f"\n  Validation {'PASSED' if passed else 'COMPLETED WITH WARNINGS'}")
    return passed


def build_score_breakdown(jobs: list, scores: dict) -> pd.DataFrame:
    """Build score breakdown DataFrame for transparency."""
    rows = []
    for fname, brand, _ in jobs:
        score, bd = scores.get((fname, brand), (0, {}))
        row = {"Filename": fname, "Brand": brand, "Total_Score": score}
        row.update(bd)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_dashboard(df: pd.DataFrame):
    """Generate a rich HTML dashboard with Chart.js visualizations."""
    # Prepare data for charts
    brand_avg = df.groupby("Brand")["Access Score"].mean().round(1).to_dict()
    score_dist = {}
    for s in df["Access Score"]:
        bucket = f"{(s//10)*10}-{(s//10)*10+9}"
        score_dist[bucket] = score_dist.get(bucket, 0) + 1

    tremfya_df  = df[df["Brand"] == "TREMFYA"].sort_values("Access Score", ascending=False)
    stelara_df  = df[df["Brand"] == "STELARA"].sort_values("Access Score", ascending=False)

    # Colour palette per brand
    brand_colours = {
        "TREMFYA": "#6366f1", "STELARA": "#06b6d4", "AMJEVITA": "#f59e0b",
        "COSENTYX": "#10b981", "ENBREL": "#ef4444", "REMICADE": "#8b5cf6",
        "SILIQ": "#ec4899", "CIMZIA": "#f97316", "BIMZELX": "#14b8a6",
        "SKYRIZI": "#84cc16", "OTEZLA": "#a78bfa", "YESINTEK": "#fb923c",
        "OTULFI": "#38bdf8", "ILUMYA": "#4ade80", "ACITRETIN": "#f472b6",
    }

    brands_json     = json.dumps(list(brand_avg.keys()))
    brand_avgs_json = json.dumps(list(brand_avg.values()))
    brand_cols_json = json.dumps([brand_colours.get(b, "#94a3b8") for b in brand_avg.keys()])
    dist_labels     = json.dumps(sorted(score_dist.keys()))
    dist_values     = json.dumps([score_dist[k] for k in sorted(score_dist.keys())])

    # Heatmap table rows
    heatmap_rows = ""
    for _, row in df.sort_values(["Brand", "Access Score"], ascending=[True, False]).iterrows():
        score = int(row["Access Score"])
        if   score >= 75: colour = "#22c55e"
        elif score >= 50: colour = "#f59e0b"
        elif score >= 25: colour = "#f97316"
        else:             colour = "#ef4444"
        fname_short = str(row["Filename"]).replace(".pdf", "")
        brand_col = brand_colours.get(str(row["Brand"]), "#94a3b8")
        heatmap_rows += f"""
        <tr>
          <td><span style="font-family:monospace;font-size:11px">{fname_short}</span></td>
          <td><span class="brand-badge" style="background:{brand_col}20;color:{brand_col};border:1px solid {brand_col}40">{row['Brand']}</span></td>
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

    # TREMFYA vs STELARA comparison
    comparison_rows = ""
    merged = pd.merge(
        tremfya_df[["Filename", "Access Score"]].rename(columns={"Access Score": "TREMFYA_Score"}),
        stelara_df[["Filename", "Access Score"]].rename(columns={"Access Score": "STELARA_Score"}),
        on="Filename", how="outer"
    ).fillna("-")
    for _, r in merged.iterrows():
        t = r["TREMFYA_Score"]
        s = r["STELARA_Score"]
        t_disp = f"{int(t)}" if t != "-" else "-"
        s_disp = f"{int(s)}" if s != "-" else "-"
        fname_short = str(r["Filename"]).replace(".pdf","")
        comparison_rows += f"<tr><td style='font-size:11px;font-family:monospace'>{fname_short}</td><td style='text-align:center;color:#6366f1;font-weight:600'>{t_disp}</td><td style='text-align:center;color:#06b6d4;font-weight:600'>{s_disp}</td></tr>"

    total_rows = len(df)
    avg_score  = df["Access Score"].mean()
    excl_count = (df["Access Score"] == 0).sum()
    high_access = (df["Access Score"] >= 75).sum()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Payer Policy Intelligence — Access Score Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --surface2: #232640;
    --border: rgba(255,255,255,0.07); --text: #e2e8f0; --muted: #94a3b8;
    --primary: #6366f1; --accent: #06b6d4; --green: #22c55e;
    --yellow: #f59e0b; --red: #ef4444; --orange: #f97316;
  }}
  body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 0 24px 60px; }}
  
  /* Header */
  .header {{ padding: 48px 0 32px; border-bottom: 1px solid var(--border); margin-bottom: 40px; }}
  .header h1 {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header p {{ color: var(--muted); margin-top: 8px; font-size: 0.95rem; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-right: 8px; margin-top: 12px; }}
  
  /* KPI Cards */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 40px; }}
  .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; position: relative; overflow: hidden; }}
  .kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent-grad, var(--primary)); }}
  .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
  .kpi-value {{ font-size: 2.4rem; font-weight: 800; line-height: 1; }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
  
  /* Charts grid */
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }}
  @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px; }}
  .chart-title {{ font-size: 14px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 20px; }}
  .chart-wrap {{ position: relative; height: 280px; }}
  
  /* Table */
  .table-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 32px; overflow-x: auto; }}
  .table-title {{ font-size: 14px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 10px 14px; color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  
  .brand-badge {{ padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap; }}
  
  /* Score bar */
  .score-bar-wrap {{ display: flex; align-items: center; gap: 10px; min-width: 120px; }}
  .score-bar {{ height: 8px; border-radius: 999px; transition: width 0.3s; flex-shrink: 0; }}
  .score-label {{ font-weight: 700; font-size: 13px; white-space: nowrap; }}
  
  /* Filter bar */
  .filter-bar {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .filter-btn {{ padding: 6px 16px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface2); color: var(--muted); font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }}
  .filter-btn:hover, .filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
  
  input.search-box {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 8px 14px; font-size: 13px; width: 260px; outline: none; }}
  input.search-box:focus {{ border-color: var(--primary); }}
  
  footer {{ color: var(--muted); font-size: 12px; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Payer Policy Intelligence</h1>
  <p>Access Quality Score Dashboard &mdash; Prior Authorization Policy Analysis</p>
  <span class="badge" style="background:#6366f120;color:#6366f1;border:1px solid #6366f140">llama-3.1-8b-instant</span>
  <span class="badge" style="background:#06b6d420;color:#06b6d4;border:1px solid #06b6d440">llama-3.3-70b-versatile</span>
  <span class="badge" style="background:#22c55e20;color:#22c55e;border:1px solid #22c55e40">Groq API</span>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi-card" style="--accent-grad: linear-gradient(90deg,#6366f1,#8b5cf6)">
    <div class="kpi-label">Total Policies Processed</div>
    <div class="kpi-value" style="color:#6366f1">{total_rows}</div>
    <div class="kpi-sub">Unique (Filename, Brand) pairs</div>
  </div>
  <div class="kpi-card" style="--accent-grad: linear-gradient(90deg,#06b6d4,#0ea5e9)">
    <div class="kpi-label">Average Access Score</div>
    <div class="kpi-value" style="color:#06b6d4">{avg_score:.1f}</div>
    <div class="kpi-sub">Out of 100</div>
  </div>
  <div class="kpi-card" style="--accent-grad: linear-gradient(90deg,#22c55e,#16a34a)">
    <div class="kpi-label">High Access Policies</div>
    <div class="kpi-value" style="color:#22c55e">{high_access}</div>
    <div class="kpi-sub">Score &ge; 75</div>
  </div>
  <div class="kpi-card" style="--accent-grad: linear-gradient(90deg,#ef4444,#dc2626)">
    <div class="kpi-label">Excluded / Zero Score</div>
    <div class="kpi-value" style="color:#ef4444">{excl_count}</div>
    <div class="kpi-sub">Explicitly non-covered</div>
  </div>
</div>

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-card">
    <div class="chart-title">Average Access Score by Brand</div>
    <div class="chart-wrap"><canvas id="brandChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Score Distribution</div>
    <div class="chart-wrap"><canvas id="distChart"></canvas></div>
  </div>
</div>

<!-- TREMFYA vs STELARA -->
<div class="charts-grid" style="grid-template-columns: 1fr 1fr;">
  <div class="chart-card">
    <div class="chart-title">TREMFYA Score Distribution</div>
    <div class="chart-wrap"><canvas id="tremfyaChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">STELARA Score Distribution</div>
    <div class="chart-wrap"><canvas id="stelaraChart"></canvas></div>
  </div>
</div>

<!-- Main Heatmap Table -->
<div class="table-card">
  <div class="table-title">Policy Access Heatmap</div>
  <div class="filter-bar">
    <input class="search-box" id="searchBox" placeholder="Search filename or brand..." oninput="filterTable()">
    <button class="filter-btn active" onclick="filterBrand('ALL', this)">All</button>
    <button class="filter-btn" onclick="filterBrand('TREMFYA', this)">TREMFYA</button>
    <button class="filter-btn" onclick="filterBrand('STELARA', this)">STELARA</button>
    <button class="filter-btn" onclick="filterBrand('OTHER', this)">Other Brands</button>
  </div>
  <table id="heatmapTable">
    <thead>
      <tr>
        <th>Filename</th><th>Brand</th><th>Age</th>
        <th>Conv. Steps</th><th>Bio. Steps</th><th>TB Test</th>
        <th>Init. Auth</th><th style="min-width:160px">Access Score</th>
      </tr>
    </thead>
    <tbody id="heatmapBody">
      {heatmap_rows}
    </tbody>
  </table>
</div>

<!-- TREMFYA vs STELARA side-by-side -->
<div class="table-card">
  <div class="table-title">TREMFYA vs STELARA — Cross-Brand Comparison</div>
  <table>
    <thead>
      <tr>
        <th>Filename</th>
        <th style="color:#6366f1;text-align:center">TREMFYA Score</th>
        <th style="color:#06b6d4;text-align:center">STELARA Score</th>
      </tr>
    </thead>
    <tbody>
      {comparison_rows}
    </tbody>
  </table>
</div>

<footer>
  Generated by Payer Policy Intelligence Pipeline &bull; Models: llama-3.1-8b-instant &amp; llama-3.3-70b-versatile via Groq API
</footer>

<script>
const brandLabels = {brands_json};
const brandAvgs   = {brand_avgs_json};
const brandCols   = {brand_cols_json};
const distLabels  = {dist_labels};
const distValues  = {dist_values};

const chartDefaults = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ backgroundColor: '#1a1d2e', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 }} }},
  scales: {{ x: {{ ticks: {{ color: '#64748b', font: {{ size: 11 }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}, y: {{ ticks: {{ color: '#64748b', font: {{ size: 11 }} }}, grid: {{ color: 'rgba(255,255,255,0.04)' }}, beginAtZero: true }} }}
}};

// Brand bar chart
new Chart(document.getElementById('brandChart'), {{
  type: 'bar',
  data: {{ labels: brandLabels, datasets: [{{ data: brandAvgs, backgroundColor: brandCols, borderRadius: 6, borderSkipped: false }}] }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, max: 100 }} }} }}
}});

// Distribution doughnut
new Chart(document.getElementById('distChart'), {{
  type: 'doughnut',
  data: {{ labels: distLabels, datasets: [{{ data: distValues, backgroundColor: ['#ef4444','#f97316','#f59e0b','#84cc16','#22c55e','#06b6d4','#6366f1','#8b5cf6','#ec4899','#14b8a6'], borderWidth: 0, hoverOffset: 8 }}] }},
  options: {{ responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: {{ legend: {{ display: true, position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}, tooltip: {{ backgroundColor: '#1a1d2e', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 }} }} }}
}});

// TREMFYA scores
const tremfyaScores = {json.dumps(tremfya_df['Access Score'].tolist())};
const tremfyaLabels = tremfyaScores.map((_, i) => `Doc ${{i+1}}`);
new Chart(document.getElementById('tremfyaChart'), {{
  type: 'bar',
  data: {{ labels: tremfyaLabels, datasets: [{{ data: tremfyaScores, backgroundColor: '#6366f1', borderRadius: 4 }}] }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, max: 100 }} }} }}
}});

// STELARA scores
const stelaraScores = {json.dumps(stelara_df['Access Score'].tolist())};
const stelaraLabels = stelaraScores.map((_, i) => `Doc ${{i+1}}`);
new Chart(document.getElementById('stelaraChart'), {{
  type: 'bar',
  data: {{ labels: stelaraLabels, datasets: [{{ data: stelaraScores, backgroundColor: '#06b6d4', borderRadius: 4 }}] }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, max: 100 }} }} }}
}});

// Filtering
let activeBrand = 'ALL';
function filterBrand(brand, btn) {{
  activeBrand = brand;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterTable();
}}
function filterTable() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  document.querySelectorAll('#heatmapBody tr').forEach(row => {{
    const text = row.textContent.toLowerCase();
    const brandCell = row.cells[1]?.textContent.trim();
    const brandMatch = activeBrand === 'ALL' || (activeBrand === 'OTHER' ? !['TREMFYA','STELARA'].includes(brandCell) : brandCell === activeBrand);
    row.style.display = (brandMatch && text.includes(q)) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    print(f"  Dashboard written -> {DASHBOARD_HTML}")


def stage5_output(jobs: list, extracted: dict, scores: dict):
    """
    Stage 5: Assemble, validate, export result.csv + score_breakdown.csv + dashboard.
    """
    print("\n" + "="*60)
    print("STAGE 5 — Output Assembly & Validation")
    print("="*60)

    df     = assemble_dataframe(jobs, extracted, scores)
    df_bd  = build_score_breakdown(jobs, scores)

    validate_dataframe(df, jobs)

    # Export
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    df_bd.to_csv(SCORE_CSV, index=False, encoding="utf-8")
    print(f"\n  Exported: {RESULT_CSV}")
    print(f"  Exported: {SCORE_CSV}")

    # Dashboard
    generate_dashboard(df)

    # Summary stats
    print("\n" + "-"*40)
    print("  SUMMARY STATISTICS")
    print("-"*40)
    print(f"  Total rows          : {len(df)}")
    print(f"  Unique brands       : {df['Brand'].nunique()}")
    print(f"  Mean Access Score   : {df['Access Score'].mean():.1f}")
    print(f"  Median Access Score : {df['Access Score'].median():.1f}")
    print(f"\n  Score by Brand:")
    brand_summary = df.groupby("Brand")["Access Score"].agg(["mean","min","max","count"])
    brand_summary.columns = ["Mean", "Min", "Max", "Count"]
    brand_summary = brand_summary.sort_values("Mean", ascending=False)
    for brand, row in brand_summary.iterrows():
        print(f"    {brand:15s}: mean={row['Mean']:.1f}  min={row['Min']}  max={row['Max']}  n={int(row['Count'])}")

    print(f"\n  Stage 5 complete.")
    return df


# ════════════════════════════════════════════════════════════════
# MAIN — Run all stages
# ════════════════════════════════════════════════════════════════

def run_pipeline():
    print("\n" + "="*60)
    print("  PAYER POLICY INTELLIGENCE PIPELINE")
    print("  Hackathon Submission | Groq API")
    print("="*60)

    raw_texts = stage1_ingest_pdfs()
    jobs      = stage2_build_jobs(raw_texts)
    extracted = stage3_extract_all(jobs)
    scores    = stage4_score_all(extracted)
    result_df = stage5_output(jobs, extracted, scores)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  result.csv   -> {RESULT_CSV}")
    print(f"  dashboard    -> {DASHBOARD_HTML}")
    print(f"  cache        -> {CACHE_FILE}")
    print("="*60 + "\n")
    return result_df


if __name__ == "__main__":
    run_pipeline()
