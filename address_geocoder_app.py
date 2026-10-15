import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import dotenv_values, load_dotenv

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("address_geocoder")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Address Geocoder | CGI",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling & Design System ──────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --cgi-purple: #5236AB;
  --cgi-purple-900: #200A58;
  --cgi-purple-700: #3A2679;
  --cgi-purple-50: #F2F1F9;
  --cgi-red: #E31937;
  --cgi-success: #1AB977;
  --cgi-gray-900: #151515;
  --cgi-gray-400: #5C5C5C;
  --cgi-white-700: #EBEEF2;
  --cgi-white-200: #F9FAFB;
  --cgi-gradient-h: linear-gradient(90deg, #E31937 0%, #A82465 60%, #5236AB 100%);
}

.cgi-brand-bar {
  height: 6px;
  width: 100%;
  background: var(--cgi-gradient-h);
  border-radius: 3px;
  margin-bottom: 1.2rem;
}

.cgi-standalone-logo {
  background: var(--cgi-red);
  color: #FFFFFF;
  font-weight: 900;
  font-size: 1.25rem;
  letter-spacing: 0.12em;
  padding: 0.35rem 0.9rem;
  border-radius: 6px;
  display: inline-block;
  margin-bottom: 0.85rem;
}

.cgi-header-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.1rem 1.5rem;
  background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
  border: 1px solid var(--cgi-white-700);
  border-left: 6px solid var(--cgi-purple);
  border-radius: 8px;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.cgi-header-banner h1 {
  margin: 0;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--cgi-purple-900);
}

.cgi-header-banner p {
  margin: 0.25rem 0 0 0;
  font-size: 0.9rem;
  color: var(--cgi-gray-400);
}
</style>
""", unsafe_allow_html=True)

# ── Top Bar & Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="cgi-brand-bar"></div>
<div class="cgi-standalone-logo">CGI</div>
<div class="cgi-header-banner">
  <div>
    <h1>Address Geocoder Assistant</h1>
    <p>Extract Latitude & Longitude using Gemini Enterprise Discovery Engine</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class GeocodeResult:
    latitude: Optional[float]
    longitude: Optional[float]
    resolved_location: Optional[str]
    match_type: Optional[str]
    geocode_status: str  # SUCCESS, UNRESOLVED, FAILED, SKIPPED_EMPTY
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


# ── Environment & Configuration Loader ───────────────────────────────────────
def resolve_env_config():
    """Loads variables from .env or EnvVariablesToBeSet.env in the script directory."""
    script_dir = Path(__file__).parent.resolve()
    env_file = script_dir / ".env"
    if not env_file.exists():
        env_file = script_dir / "EnvVariablesToBeSet.env"

    env_vars = {}
    if env_file.exists():
        env_vars = dotenv_values(env_file)
        load_dotenv(env_file)

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or env_vars.get("GOOGLE_CLOUD_PROJECT", "")
    engine = os.getenv("DISCOVERY_ENGINE_ID") or env_vars.get("DISCOVERY_ENGINE_ID", "")
    assistant = os.getenv("DISCOVERY_ASSISTANT_ID") or env_vars.get("DISCOVERY_ASSISTANT_ID", "default_assistant")

    raw_api_url = os.getenv("api_url") or env_vars.get("api_url", "")
    raw_api_url = re.sub(r'^[fF]?["\']|["\']$', '', raw_api_url.strip())

    if raw_api_url:
        endpoint = (
            raw_api_url.replace("{project}", project)
            .replace("{engine}", engine)
            .replace("{assistant}", assistant)
        )
    else:
        endpoint = (
            f"https://discoveryengine.googleapis.com/v1/projects/{project}/"
            f"locations/global/collections/default_collection/engines/{engine}/"
            f"assistants/{assistant}:streamAssist"
        )
    return project, engine, assistant, endpoint, env_file.name if env_file.exists() else "None"


project_id, engine_id, assistant_id, resolved_api_url, env_loaded_name = resolve_env_config()

# ── Authentication Helper ────────────────────────────────────────────────────
def get_google_access_token() -> Optional[str]:
    """Attempts to obtain an OAuth token using google-auth default credentials."""
    try:
        import google.auth
        from google.auth.transport.requests import Request
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token
    except Exception:
        return None


# ── Sidebar Configuration & Prompt Suggestions ───────────────────────────────
with st.sidebar:
    with st.expander("⚙️ Advanced API Settings", expanded=False):
        st.caption(f"Loaded config from: `{env_loaded_name}`")
        cfg_project = st.text_input("GCP Project ID", value=project_id)
        cfg_engine = st.text_input("Discovery Engine ID", value=engine_id)
        cfg_assistant = st.text_input("Assistant ID", value=assistant_id)
        cfg_url = st.text_input("API URL Endpoint", value=resolved_api_url)

    st.markdown("---")
    st.subheader("💡 Geocoding Prompts & Presets")
    st.caption("Tuned for US ZIP/ZIP+4, UK postcodes, Canadian postal codes, and partial addresses.")

    prompt_options = {
        "Global Postal Codes & Partial Addresses (Recommended)": (
            "You are an expert geographic geocoder. Geocode the location input: '{address}'.\n\n"
            "Instructions:\n"
            "1. US Addresses: Support full street addresses, 5-digit ZIP (e.g., 90210), and ZIP+4 (e.g., 10001-1234).\n"
            "2. UK Addresses: Support full UK postcodes (e.g., SW1A 1AA, EC1A 1BB, M1 1AE) or outward codes.\n"
            "3. Canadian Addresses: Support 6-character alphanumeric postal codes (e.g., K1A 0B1, M5V 2T6).\n"
            "4. Partial / Ambiguous Inputs: If only a street, neighborhood, city, county, or postal code is provided, "
            "compute the geographic centroid.\n"
            "5. Return strictly a single valid JSON object without markdown code blocks, preamble, or commentary:\n"
            '{"latitude": <float>, "longitude": <float>, "resolved_location": "<formatted address, region, country>", '
            '"match_type": "<exact_address|postal_code_centroid|partial_locality|interpolated>"}'
        ),
        "Strict JSON Coordinates Only": (
            "Extract latitude and longitude for: '{address}'. "
            "If it is a postal code or partial locality, return its geographic centroid coordinates. "
            'Respond strictly with valid JSON: {"latitude": <float>, "longitude": <float>, "resolved_location": "<string>"}. '
            "Do not include markdown or explanations."
        ),
        "Detailed Regional & Country Verification": (
            "Geocode this location: '{address}'. "
            "Resolve US, UK, Canadian, or global postcodes and partial addresses. "
            'Return strictly JSON: {"latitude": <float>, "longitude": <float>, "resolved_location": "<full_address>", '
            '"country": "<country_name>", "match_type": "<exact_address|postal_code_centroid|partial_locality>"}'
        ),
    }

    selected_prompt_key = st.selectbox("Prompt Template", list(prompt_options.keys()))
    custom_prompt = st.text_area(
        "Active Prompt (`{address}` placeholder is replaced per row)",
        value=prompt_options[selected_prompt_key],
        height=130,
    )

    st.markdown("---")
    st.subheader("⚡ Performance & Retries")
    cfg_max_retries = st.slider("Max Retries on Transient Errors", min_value=1, max_value=5, value=3)
    cfg_timeout = st.slider("Request Timeout (seconds)", min_value=15, max_value=90, value=45)


# ── Coordinate Validation Helper ─────────────────────────────────────────────
def is_valid_coordinate(lat: Optional[float], lon: Optional[float]) -> bool:
    """Validates that latitude and longitude fall within real-world geographic bounds."""
    if lat is None or lon is None:
        return False
    try:
        v_lat = float(lat)
        v_lon = float(lon)
        return -90.0 <= v_lat <= 90.0 and -180.0 <= v_lon <= 180.0 and not (v_lat == 0.0 and v_lon == 0.0)
    except (ValueError, TypeError):
        return False


def normalize_and_validate_coords(lat_raw: Any, lon_raw: Any) -> Tuple[Optional[float], Optional[float]]:
    """Converts raw values to float, detects and fixes swapped [lon, lat] order, and validates bounds."""
    if lat_raw is None or lon_raw is None:
        return None, None
    try:
        lat_clean = re.sub(r"[^\d+.-]", "", str(lat_raw)).strip()
        lon_clean = re.sub(r"[^\d+.-]", "", str(lon_raw)).strip()
        if not lat_clean or not lon_clean:
            return None, None
        v_lat = float(lat_clean)
        v_lon = float(lon_clean)

        # Auto-detect and correct swapped coordinates (GeoJSON [lon, lat] order)
        if abs(v_lat) > 90.0 and abs(v_lat) <= 180.0 and abs(v_lon) <= 90.0:
            v_lat, v_lon = v_lon, v_lat

        if is_valid_coordinate(v_lat, v_lon):
            return v_lat, v_lon
    except (ValueError, TypeError):
        pass
    return None, None


def extract_from_dict(d: Any) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Recursively searches a dictionary structure for coordinates and location metadata."""
    if not isinstance(d, dict):
        return None, None, None, None

    res_loc = d.get("resolved_location") or d.get("formatted_address") or d.get("address") or d.get("display_name")
    m_type = d.get("match_type") or d.get("type")

    # 1. Direct keys
    raw_lat = d.get("latitude") if d.get("latitude") is not None else d.get("lat", d.get("lat_deg"))
    raw_lon = d.get("longitude") if d.get("longitude") is not None else d.get("long", d.get("lng", d.get("lon")))
    lat, lon = normalize_and_validate_coords(raw_lat, raw_lon)
    if lat is not None and lon is not None:
        return lat, lon, res_loc, m_type or "json_direct"

    # 2. Nested location/geometry/coordinates dicts or arrays
    for sub_key in ("location", "coordinates", "coords", "geometry", "centroid", "point", "position"):
        sub = d.get(sub_key)
        if isinstance(sub, dict):
            s_lat, s_lon, s_loc, s_type = extract_from_dict(sub)
            if s_lat is not None and s_lon is not None:
                return s_lat, s_lon, res_loc or s_loc, m_type or s_type or f"json_{sub_key}"
        elif isinstance(sub, (list, tuple)) and len(sub) >= 2:
            c_lat, c_lon = normalize_and_validate_coords(sub[0], sub[1])
            if c_lat is not None and c_lon is not None:
                return c_lat, c_lon, res_loc, m_type or f"json_{sub_key}_array"

    return None, None, res_loc, m_type


# ── Geocoder Engine ──────────────────────────────────────────────────────────
def parse_geocoding_result(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Extracts latitude, longitude, resolved location, and match type from model text output."""
    if not text:
        return None, None, None, None

    # 1. Check markdown code blocks (```json ... ``` or ``` ... ```)
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        try:
            data = json.loads(block.strip())
            lat, lon, res_loc, m_type = extract_from_dict(data)
            if lat is not None and lon is not None:
                return lat, lon, res_loc, m_type
        except Exception:
            pass

    # 2. Direct JSON decode
    try:
        data = json.loads(text.strip())
        lat, lon, res_loc, m_type = extract_from_dict(data)
        if lat is not None and lon is not None:
            return lat, lon, res_loc, m_type
    except Exception:
        pass

    # 3. Extract embedded JSON using JSONDecoder.raw_decode
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            data, end_pos = decoder.raw_decode(text, pos)
            lat, lon, res_loc, m_type = extract_from_dict(data)
            if lat is not None and lon is not None:
                return lat, lon, res_loc, m_type
            pos = text.find("{", end_pos)
        except Exception:
            pos = text.find("{", pos + 1)

    # 4. Key-value regex parsing with proper precedence and cardinal sign support
    lat_match = re.search(
        r"['\"]?lat(?:itude)?['\"]?\s*[:=]\s*['\"]?([+-]?\d+(?:\.\d+)?)\s*°?\s*([NS])?",
        text,
        re.IGNORECASE,
    )
    lon_match = re.search(
        r"['\"]?(?:longitude|long|lng|lon)['\"]?\s*[:=]\s*['\"]?([+-]?\d+(?:\.\d+)?)\s*°?\s*([EW])?",
        text,
        re.IGNORECASE,
    )
    if lat_match and lon_match:
        try:
            raw_lat_val = float(lat_match.group(1)) * (-1 if (lat_match.group(2) or "").upper() == "S" else 1)
            raw_lon_val = float(lon_match.group(1)) * (-1 if (lon_match.group(2) or "").upper() == "W" else 1)
            lat, lon = normalize_and_validate_coords(raw_lat_val, raw_lon_val)
            if lat is not None and lon is not None:
                loc_match = re.search(r"['\"]?(?:resolved_location|formatted_address)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
                res_loc = loc_match.group(1) if loc_match else None
                return lat, lon, res_loc, "regex_key_value"
        except Exception:
            pass

    # 5. Cardinal coordinates (e.g., 51.5074° N, 0.1278° W)
    cardinal_match = re.search(
        r"(\d+(?:\.\d+)?)\s*°?\s*([NS])\s*[,;]?\s*(\d+(?:\.\d+)?)\s*°?\s*([EW])",
        text,
        re.IGNORECASE,
    )
    if cardinal_match:
        try:
            v_lat = float(cardinal_match.group(1)) * (-1 if cardinal_match.group(2).upper() == "S" else 1)
            v_lon = float(cardinal_match.group(3)) * (-1 if cardinal_match.group(4).upper() == "W" else 1)
            lat, lon = normalize_and_validate_coords(v_lat, v_lon)
            if lat is not None and lon is not None:
                return lat, lon, None, "cardinal_degrees"
        except Exception:
            pass

    # 6. Standard coordinate pair fallback (e.g., 37.7749, -122.4194)
    for pair_match in re.finditer(r"([+-]?\d{1,2}(?:\.\d+)?)\s*[,;\s]\s*([+-]?\d{1,3}(?:\.\d+)?)", text):
        try:
            lat, lon = normalize_and_validate_coords(pair_match.group(1), pair_match.group(2))
            if lat is not None and lon is not None:
                return lat, lon, None, "coordinate_pair"
        except Exception:
            pass

    return None, None, None, None


def parse_stream_assist_response(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses Discovery Engine streamAssist response text.
    Filters out 'IN_PROGRESS' heartbeats and extracts the final answer text.
    Returns (answer_text, error_message).
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty response from API"

    text = raw_text.strip()
    parsed_chunks = []

    # 1. Direct JSON parse (handles JSON arrays or single JSON objects)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            parsed_chunks.extend(data)
        elif isinstance(data, dict):
            parsed_chunks.append(data)
    except Exception:
        # 2. SSE parsing if lines start with 'data:'
        for line in text.splitlines():
            line_str = line.strip()
            if line_str.startswith("data:"):
                payload_str = line_str[5:].strip()
                if payload_str:
                    try:
                        parsed_chunks.append(json.loads(payload_str))
                    except Exception:
                        pass

        # 3. Concatenated JSON decoding fallback
        if not parsed_chunks:
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(text):
                while pos < len(text) and text[pos] in " \t\r\n,[]":
                    pos += 1
                if pos >= len(text):
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text, pos)
                    parsed_chunks.append(obj)
                    pos = end_idx
                except Exception:
                    next_nl = text.find("\n", pos)
                    if next_nl == -1:
                        break
                    pos = next_nl + 1

    # Check for any API level error chunks
    for chunk in parsed_chunks:
        if isinstance(chunk, dict) and "error" in chunk:
            err_info = chunk["error"]
            err_msg = err_info.get("message", str(err_info)) if isinstance(err_info, dict) else str(err_info)
            return None, f"Discovery Engine Error: {err_msg}"

    # Prioritize terminal chunk with state == SUCCEEDED
    for chunk in reversed(parsed_chunks):
        if isinstance(chunk, dict):
            if any(k in chunk for k in ("latitude", "lat", "coordinates")):
                return json.dumps(chunk), None
            ans = chunk.get("answer", {})
            if isinstance(ans, dict) and ans.get("state") == "SUCCEEDED" and ans.get("answerText"):
                return ans["answerText"], None

    # Accumulate answerText or grounded replies across chunks
    collected = []
    for chunk in parsed_chunks:
        if isinstance(chunk, dict):
            if any(k in chunk for k in ("latitude", "lat", "coordinates")):
                return json.dumps(chunk), None
            ans = chunk.get("answer", {})
            if isinstance(ans, dict):
                txt = ans.get("answerText")
                if txt and (not collected or txt != collected[-1]):
                    collected.append(txt)
                for reply in ans.get("replies", []):
                    content_obj = reply.get("groundedContent", {}).get("content", {})
                    if isinstance(content_obj, dict):
                        if content_obj.get("text") and (not collected or content_obj["text"] != collected[-1]):
                            collected.append(content_obj["text"])
                        for part in content_obj.get("parts", []):
                            if isinstance(part, dict) and part.get("text") and (not collected or part["text"] != collected[-1]):
                                collected.append(part["text"])

    if collected:
        # Cumulative chunks check
        if len(collected) > 1 and collected[-1].startswith(collected[0]) and len(collected[-1]) >= len(collected[0]):
            return collected[-1], None
        if len(collected) > 1 and all(collected[-1].startswith(c[:min(len(c), 10)]) for c in collected[:-1]):
            return collected[-1], None
        # Streaming deltas: join without spaces to avoid breaking JSON tokens and floats
        return "".join(collected), None

    return None, "No completed answer text returned"


def query_stream_assist_with_retry(
    address: str,
    prompt_template: str,
    token: str,
    endpoint: str,
    project: str,
    session: requests.Session,
    max_retries: int = 3,
    timeout: int = 45,
) -> GeocodeResult:
    """
    Executes the streamAssist API call with exponential backoff for transient errors
    and validates input and output coordinates.
    """
    # 1. Input Validation
    cleaned_addr = str(address).strip()
    if not cleaned_addr or cleaned_addr.lower() in {"nan", "none", "null", "n/a", "undefined", ""}:
        return GeocodeResult(
            latitude=None,
            longitude=None,
            resolved_location=None,
            match_type=None,
            geocode_status="SKIPPED_EMPTY",
            error_message="Address input was empty, NaN, or whitespace",
        )

    query_text = prompt_template.replace("{address}", cleaned_addr)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project,
    }
    payload = {"query": {"text": query_text}}

    last_err = ""
    last_raw = ""
    backoff_seconds = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.encoding = "utf-8"
            last_raw = response.text

            # Transient rate limit (429) or server errors (5xx) -> retry with backoff
            if response.status_code in {429, 500, 502, 503, 504}:
                last_err = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning("Attempt %d/%d failed with %s. Backing off for %.2fs", attempt, max_retries, last_err, backoff_seconds)
                if attempt < max_retries:
                    time.sleep(backoff_seconds + random.uniform(0.1, 0.4))
                    backoff_seconds *= 2.0
                    continue
                else:
                    return GeocodeResult(
                        latitude=None,
                        longitude=None,
                        resolved_location=None,
                        match_type=None,
                        geocode_status="FAILED",
                        error_message=last_err,
                        raw_response=last_raw[:500],
                    )

            if response.status_code != 200:
                return GeocodeResult(
                    latitude=None,
                    longitude=None,
                    resolved_location=None,
                    match_type=None,
                    geocode_status="FAILED",
                    error_message=f"API request failed (HTTP {response.status_code})",
                    raw_response=last_raw[:500],
                )

            # Parse Discovery Engine stream response
            ans_text, parse_err = parse_stream_assist_response(last_raw)
            if parse_err or not ans_text:
                return GeocodeResult(
                    latitude=None,
                    longitude=None,
                    resolved_location=None,
                    match_type=None,
                    geocode_status="UNRESOLVED",
                    error_message=parse_err or "No completed answer returned",
                    raw_response=last_raw[:500],
                )

            # Parse coordinates from answer text
            lat, lon, res_loc, m_type = parse_geocoding_result(ans_text)
            if is_valid_coordinate(lat, lon):
                return GeocodeResult(
                    latitude=lat,
                    longitude=lon,
                    resolved_location=res_loc or cleaned_addr,
                    match_type=m_type or "geocoded",
                    geocode_status="SUCCESS",
                    raw_response=ans_text,
                )
            else:
                return GeocodeResult(
                    latitude=None,
                    longitude=None,
                    resolved_location=res_loc,
                    match_type=m_type or "unresolved_geometry",
                    geocode_status="UNRESOLVED",
                    error_message="Coordinates missing or outside valid range (-90..90, -180..180)",
                    raw_response=ans_text[:500],
                )

        except (requests.Timeout, requests.ConnectionError) as net_err:
            last_err = f"Network Error ({type(net_err).__name__}): {str(net_err)}"
            logger.warning("Attempt %d/%d network issue: %s", attempt, max_retries, last_err)
            if attempt < max_retries:
                time.sleep(backoff_seconds + random.uniform(0.1, 0.4))
                backoff_seconds *= 2.0
            else:
                return GeocodeResult(
                    latitude=None,
                    longitude=None,
                    resolved_location=None,
                    match_type=None,
                    geocode_status="FAILED",
                    error_message=last_err,
                )
        except Exception as exc:
            return GeocodeResult(
                latitude=None,
                longitude=None,
                resolved_location=None,
                match_type=None,
                geocode_status="FAILED",
                error_message=f"Unexpected processing exception: {str(exc)}",
            )

    return GeocodeResult(
        latitude=None,
        longitude=None,
        resolved_location=None,
        match_type=None,
        geocode_status="FAILED",
        error_message=last_err or "Max retries exceeded without successful response",
        raw_response=last_raw[:500],
    )


def generate_excel_workbook(result_df: pd.DataFrame, summary_data: Dict[str, Any]) -> bytes:
    """Generates an Excel workbook containing both data records and an executive summary sheet."""
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Geocoded_Addresses")

        summary_rows = [
            {"Metric": "Processing Timestamp (UTC)", "Value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")},
            {"Metric": "Total Records Processed", "Value": summary_data.get("total", 0)},
            {"Metric": "Successfully Geocoded", "Value": summary_data.get("successful", 0)},
            {"Metric": "Unresolved Records", "Value": summary_data.get("unresolved", 0)},
            {"Metric": "Failed Records (API/Network)", "Value": summary_data.get("failed", 0)},
            {"Metric": "Skipped (Empty / Blank)", "Value": summary_data.get("skipped", 0)},
            {"Metric": "Success Rate (%)", "Value": f"{summary_data.get('success_rate', 0.0):.2f}%"},
        ]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, index=False, sheet_name="Summary_Report")
    return excel_buffer.getvalue()


# ── Main UI: File Upload & Processing ────────────────────────────────────────
auto_token = get_google_access_token()
manual_token = st.text_input(
    "🔐 Bearer Access Token",
    value=auto_token or "",
    type="password",
    help="Leave default if ADC (Application Default Credentials) is active or paste a gcloud token.",
)
active_token = manual_token.strip() or auto_token

tab_batch, tab_single = st.tabs(["📁 Batch Geocoding (Excel Upload)", "📍 Single Address Geocoding"])

with tab_batch:
    uploaded_file = st.file_uploader("Upload Excel File with Addresses (.xlsx, .xls)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"File loaded successfully ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
            st.stop()

        col1, col2 = st.columns([2, 1])
        with col1:
            columns = list(df.columns)
            default_idx = 0
            for i, col_name in enumerate(columns):
                if any(term in str(col_name).lower() for term in ["address", "location", "street", "city", "postal", "postcode", "zip", "pcode"]):
                    default_idx = i
                    break
            selected_col = st.selectbox("Select Address Column", columns, index=default_idx)

        with col2:
            max_rows = st.number_input("Max rows to process (0 = all)", min_value=0, max_value=len(df), value=0)

        with st.expander("Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        if st.button("🚀 Start Geocoding", type="primary", use_container_width=True):
            if not active_token:
                st.error("No valid Google Cloud access token found. Please run `gcloud auth application-default login` or provide a Bearer Access Token above.")
                st.stop()

            rows_to_process = df if max_rows == 0 else df.head(max_rows)
            total_rows = len(rows_to_process)

            latitudes: List[Optional[float]] = []
            longitudes: List[Optional[float]] = []
            resolved_locations: List[Optional[str]] = []
            match_types: List[Optional[str]] = []
            statuses: List[str] = []
            error_messages: List[Optional[str]] = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Use persistent HTTP Session for socket reuse & performance
            with requests.Session() as http_session:
                for idx, (_, row) in enumerate(rows_to_process.iterrows()):
                    addr_val = row[selected_col]
                    addr_str = "" if pd.isna(addr_val) else str(addr_val).strip()
                    status_text.text(f"Processing row {idx + 1}/{total_rows}: {addr_str[:45]}...")

                    res = query_stream_assist_with_retry(
                        address=addr_str,
                        prompt_template=custom_prompt,
                        token=active_token,
                        endpoint=cfg_url,
                        project=cfg_project,
                        session=http_session,
                        max_retries=cfg_max_retries,
                        timeout=cfg_timeout,
                    )

                    latitudes.append(res.latitude)
                    longitudes.append(res.longitude)
                    resolved_locations.append(res.resolved_location)
                    match_types.append(res.match_type)
                    statuses.append(res.geocode_status)
                    error_messages.append(res.error_message)

                    progress_bar.progress((idx + 1) / total_rows)

            status_text.text("Geocoding process finished!")

            # Build output dataframe
            result_df = rows_to_process.copy()
            result_df["latitude"] = latitudes
            result_df["longitude"] = longitudes
            result_df["resolved_location"] = resolved_locations
            result_df["match_type"] = match_types
            result_df["geocode_status"] = statuses
            result_df["error_message"] = error_messages

            # ── Executive Summary Report Calculations ────────────────────────────
            success_count = sum(1 for s in statuses if s == "SUCCESS")
            unresolved_count = sum(1 for s in statuses if s == "UNRESOLVED")
            failed_count = sum(1 for s in statuses if s == "FAILED")
            skipped_count = sum(1 for s in statuses if s == "SKIPPED_EMPTY")
            success_rate = (success_count / total_rows * 100.0) if total_rows > 0 else 0.0

            summary_metrics = {
                "total": total_rows,
                "successful": success_count,
                "unresolved": unresolved_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "success_rate": success_rate,
            }

            st.markdown("<h3 style='margin-top: 1.5rem;'>📈 Executive Summary Report</h3>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Processed", total_rows)
            m2.metric("Successful (Coordinates Resolved)", success_count, delta=f"{success_rate:.1f}%")
            m3.metric("Unresolved", unresolved_count, help="Inputs that could not be mapped to valid coordinates")
            m4.metric("Failed / Empty", failed_count + skipped_count, delta=f"-{failed_count} errors" if failed_count else None, delta_color="inverse")

            # Map Visualization
            map_df = result_df.dropna(subset=["latitude", "longitude"])
            if not map_df.empty:
                st.subheader("🗺️ Geocoded Locations Map")
                st.map(map_df[["latitude", "longitude"]])

            st.subheader("📊 Output Preview")
            st.dataframe(result_df, use_container_width=True)

            # Multi-sheet Excel export
            excel_data = generate_excel_workbook(result_df, summary_metrics)

            st.download_button(
                label="📥 Download Geocoded Excel File (With Summary Sheet)",
                data=excel_data,
                file_name=f"geocoded_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
    else:
        st.info("Please upload an Excel file containing addresses in the panel above to begin.")

with tab_single:
    st.subheader("📍 Geocode a Single Address")
    single_address = st.text_input(
        "Enter Address or Postal Code",
        placeholder="e.g., 10 Downing Street, London SW1A 2AA or 90210, USA",
    )
    if st.button("📍 Geocode Address", type="primary", use_container_width=True):
        if not active_token:
            st.error("No valid Google Cloud access token found. Please run `gcloud auth application-default login` or provide a Bearer Access Token above.")
            st.stop()
        if not single_address.strip():
            st.warning("Please enter an address or postal code.")
        else:
            with st.spinner("Geocoding location..."):
                with requests.Session() as http_session:
                    res = query_stream_assist_with_retry(
                        address=single_address.strip(),
                        prompt_template=custom_prompt,
                        token=active_token,
                        endpoint=cfg_url,
                        project=cfg_project,
                        session=http_session,
                        max_retries=cfg_max_retries,
                        timeout=cfg_timeout,
                    )
            st.session_state["single_result"] = (single_address.strip(), res)

    if "single_result" in st.session_state:
        addr_text, res = st.session_state["single_result"]
        if res.geocode_status == "SUCCESS":
            st.success(f"**Resolved Location:** {res.resolved_location}")
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Latitude", f"{res.latitude:.6f}")
            sm2.metric("Longitude", f"{res.longitude:.6f}")
            sm3.metric("Match Type", res.match_type or "geocoded")

            map_df = pd.DataFrame([{"latitude": res.latitude, "longitude": res.longitude}])
            st.subheader("🗺️ Location Map")
            st.map(map_df)
        elif res.geocode_status == "UNRESOLVED":
            st.warning(f"Could not resolve coordinates: {res.error_message or 'Unresolved address'}")
            if res.resolved_location:
                st.caption(f"Identified location string: {res.resolved_location}")
        elif res.geocode_status == "SKIPPED_EMPTY":
            st.warning("The address input was empty or whitespace.")
        else:
            st.error(f"Geocoding failed: {res.error_message or 'Unknown error'}")

        # Build output dataframe and summary for single address download
        single_df = pd.DataFrame([{
            "input_address": addr_text,
            "latitude": res.latitude,
            "longitude": res.longitude,
            "resolved_location": res.resolved_location,
            "match_type": res.match_type,
            "geocode_status": res.geocode_status,
            "error_message": res.error_message,
        }])
        single_summary = {
            "total": 1,
            "successful": 1 if res.geocode_status == "SUCCESS" else 0,
            "unresolved": 1 if res.geocode_status == "UNRESOLVED" else 0,
            "failed": 1 if res.geocode_status == "FAILED" else 0,
            "skipped": 1 if res.geocode_status == "SKIPPED_EMPTY" else 0,
            "success_rate": 100.0 if res.geocode_status == "SUCCESS" else 0.0,
        }
        single_excel_data = generate_excel_workbook(single_df, single_summary)

        st.subheader("📊 Output Preview")
        st.dataframe(single_df, use_container_width=True)

        st.download_button(
            label="📥 Download Geocoded Excel File (With Summary Sheet)",
            data=single_excel_data,
            file_name=f"geocoded_single_address_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
