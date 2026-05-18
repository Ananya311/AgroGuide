from __future__ import annotations

import json
import pickle
from pathlib import Path
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from scipy.sparse import load_npz

from calculator import calculate_fertilizer_doses
from prompts import build_prompt


BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = VECTORSTORE_DIR / "vectorizer.pkl"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
SYSTEM_PROMPT = "You are an agricultural advisory system for Tamil Nadu crop nutrient management."
GROQ_MODEL = "llama-3.3-70b-versatile"
CROPS = ["Rice", "Maize", "Sorghum", "Pearl Millet"]
DISTRICTS = ["Thanjavur", "Nagapattinam", "Thiruvarur", "Ramanathapuram", "Trichy"]
SOIL_TYPES = ["Black cotton", "Red loamy", "Alluvial", "Sandy loam"]
SEASONS_BY_CROP = {
    "Rice": ["Kuruvai", "Samba"],
    "Maize": ["General"],
    "Sorghum": ["General"],
    "Pearl Millet": ["General"],
}
ROLES = ["Farmer", "Agronomist", "Manager"]
SEASON_HINTS = {
    "Kuruvai": "Kuruvai short-duration rice season early irrigation delta June to September",
    "Samba": "Samba long-duration rice season main monsoon crop August to January",
    "General": "general crop season in Tamil Nadu",
}
MANAGER_PRICES = {
    "Rice": 22.0,
    "Maize": 20.0,
    "Sorghum": 24.0,
    "Pearl Millet": 23.0,
}
MAX_TOKENS_BY_ROLE = {
    "Farmer": 1000,
    "Agronomist": 3000,
    "Manager": 2500,
}
CROP_DISPLAY_MAP = {
    "Rice": "Rice",
    "Maize": "Maize",
    "Sorghum": "Sorghum",
    "Pearl Millet": "Pearl Millet",
}


load_dotenv()


class IncompleteResponseError(RuntimeError):
    """Raised when the model response is missing required sections."""


def vectorstore_exists() -> bool:
    return INDEX_PATH.exists() and METADATA_PATH.exists() and VECTORIZER_PATH.exists()


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    if not vectorstore_exists():
        raise FileNotFoundError("Vector store files are missing. Please run `python ingest.py` first.")
    matrix = load_npz(INDEX_PATH)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return matrix, metadata, vectorizer


def retrieve_context(crop: str, district: str, season: str, soil_type: str) -> list[dict]:
    if not vectorstore_exists():
        raise FileNotFoundError("Vector store files are missing. Please run `python ingest.py` first.")
    season_hint = SEASON_HINTS.get(season, season)
    query = (
        f"{crop} fertilizer recommendation in Tamil Nadu for {district} district "
        f"during {season} season on {soil_type} soil. "
        f"{season_hint} {district} {soil_type} {crop} nutrient management."
    )
    matrix, records, vectorizer = get_vectorstore()
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    crop_term = crop.lower()
    district_term = district.lower()
    season_term = season.lower()
    soil_type_term = soil_type.lower()
    for index, record in enumerate(records):
        text = record["text"].lower()
        if crop_term in text:
            scores[index] += 1.5
        if district_term in text:
            scores[index] += 0.8
        if season != "General" and season_term in text:
            scores[index] += 1.2
        if soil_type_term in text:
            scores[index] += 0.6
    top_indices = scores.argsort()[-4:][::-1]

    return [
        {
            "page_content": records[index]["text"],
            "metadata": records[index]["metadata"],
        }
        for index in top_indices
    ]


def format_context(documents: list[dict], role: str) -> str:
    max_chars_by_role = {
        "Farmer": 900,
        "Agronomist": 650,
        "Manager": 800,
    }
    max_chars = max_chars_by_role.get(role, 800)
    formatted_chunks = []
    for index, doc in enumerate(documents, start=1):
        chunk_text = doc["page_content"][:max_chars].strip()
        formatted_chunks.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Source: {doc['metadata'].get('source', 'Unknown')}",
                    f"Page: {doc['metadata'].get('page', 'Unknown')}",
                    chunk_text,
                ]
            )
        )
    return "\n\n".join(formatted_chunks)


def format_warnings(warnings: list[str]) -> str:
    return "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No special warnings."


def build_source_summary(documents: list[dict]) -> str:
    return "; ".join(
        f"{doc['metadata'].get('source', 'Unknown')} page {doc['metadata'].get('page', 'Unknown')}"
        for doc in documents
    )


def normalize_crop_name(crop: str) -> str:
    return CROP_DISPLAY_MAP.get(crop, crop)


def response_has_required_sections(role: str, text: str) -> bool:
    normalized = text.lower()
    required_sections = {
        "Farmer": [
            "## fertilizers to buy",
            "## when to apply",
            "## symptoms to watch",
            "## warning",
        ],
        "Agronomist": [
            "## nutrient interpretation of the soil test",
            "## split application schedule",
            "## lcc-based nitrogen adjustment",
            "## micronutrient and soil reaction considerations",
            "## source-grounded justification",
        ],
        "Manager": [
            "## final procurement quantities",
            "## estimated fertilizer cost",
            "## expected yield uplift",
            "## approximate roi",
            "## bulk purchase and scheduling advice",
        ],
    }
    return all(section in normalized for section in required_sections[role])


def build_zero_dose_notes(calculation: dict) -> list[str]:
    notes = []
    soil_inputs = calculation["inputs_used"]
    total_requirement = calculation["per_hectare_requirements"]["total_requirement"]
    if calculation["urea_kg"] == 0:
        notes.append(
            f"Urea is 0.0 kg because soil nitrogen ({soil_inputs['soil_n']} kg/ha) already meets or exceeds the crop nitrogen requirement per hectare ({total_requirement['N']} kg/ha)."
        )
    if calculation["ssp_kg"] == 0:
        notes.append(
            f"SSP is 0.0 kg because soil phosphorus ({soil_inputs['soil_p']} kg/ha) already meets or exceeds the crop phosphorus requirement per hectare ({total_requirement['P']} kg/ha)."
        )
    if calculation["mop_kg"] == 0:
        notes.append(
            f"MOP is 0.0 kg because soil potassium ({soil_inputs['soil_k']} kg/ha) already meets or exceeds the crop potassium requirement per hectare ({total_requirement['K']} kg/ha)."
        )
    return notes


def generate_fallback_response(
    role: str,
    crop: str,
    calculation: dict,
    district: str,
    season: str,
    soil_type: str,
    field_area: float,
    pH: float,
    organic_carbon: float,
    documents: list[dict],
) -> str:
    source_summary = build_source_summary(documents)
    warnings = calculation["warnings"] or ["No special warnings."]
    urea = calculation["urea_kg"]
    ssp = calculation["ssp_kg"]
    mop = calculation["mop_kg"]
    soil_n = calculation["inputs_used"]["soil_n"]
    soil_p = calculation["inputs_used"]["soil_p"]
    soil_k = calculation["inputs_used"]["soil_k"]
    crop_name = normalize_crop_name(crop)
    season_text = (
        "Kuruvai is a shorter irrigated season, so procurement and topdressing should be tightly scheduled."
        if season == "Kuruvai"
        else "Samba is a longer main-season crop, so procurement and nutrient scheduling should cover a longer crop duration."
        if season == "Samba"
        else f"{crop_name} nutrient operations should follow the local production schedule for the selected season."
    )
    season_ops = (
        "- Complete basal fertilizer placement before or at transplanting and keep follow-up topdress material ready early.\n"
        "- Avoid delays because the Kuruvai crop window is short and timing losses are harder to recover."
        if season == "Kuruvai"
        else "- Stage fertilizer stocks for a longer crop duration and monitor later-season nutrient demand carefully.\n"
             "- Keep procurement buffers because Samba extends longer and operational delays can affect later crop stages."
        if season == "Samba"
        else "- Plan basal and follow-up fertilizer stock according to the local crop calendar.\n"
             "- Keep enough material ready for the next critical growth stage."
    )
    farmer_schedule = {
        "Rice": "Apply urea in split doses: one part at establishment, one part during active tillering, and one part near panicle initiation.",
        "Maize": "Apply urea in split doses: one part at sowing, one part at knee-high stage, and one part before tasseling.",
        "Sorghum": "Apply urea in split doses: one part at sowing and the rest at early vegetative growth.",
        "Pearl Millet": "Apply urea in split doses: one part at sowing and the rest during early crop growth.",
    }
    farmer_mop = {
        "Rice": "Apply MOP as basal or in two splits, depending on field practice and water management.",
        "Maize": "Apply MOP mostly as basal or early topdressing before rapid vegetative growth.",
        "Sorghum": "Apply MOP as basal or early growth topdressing based on soil need.",
        "Pearl Millet": "Apply MOP as basal or early growth topdressing based on soil need.",
    }
    agr_schedule = {
        "Rice": "- Apply urea in 3 splits at basal, active tillering, and panicle initiation stages; if urea is zero, no N topdressing is required from this calculation.\n- Apply MOP as basal or in two splits between basal and panicle initiation; if MOP is zero, no potassic fertilizer is required from this calculation.",
        "Maize": "- Apply urea in 3 splits at sowing, knee-high stage, and pre-tasseling; if urea is zero, no additional N topdressing is required from this calculation.\n- Apply MOP as basal or early vegetative-stage application; if MOP is zero, no potassic fertilizer is required from this calculation.",
        "Sorghum": "- Apply urea between basal and early vegetative growth in practical splits; if urea is zero, no additional N topdressing is required from this calculation.\n- Apply MOP as basal or early growth application; if MOP is zero, no potassic fertilizer is required from this calculation.",
        "Pearl Millet": "- Apply urea between basal and early vegetative growth in practical splits; if urea is zero, no additional N topdressing is required from this calculation.\n- Apply MOP as basal or early growth application; if MOP is zero, no potassic fertilizer is required from this calculation.",
    }

    if role == "Farmer":
        return f"""## Fertilizers to Buy
- Buy **{urea} kg urea**, **{ssp} kg SSP**, and **{mop} kg MOP** for your **{field_area} acre** {crop_name.lower()} field.
- These are the final quantities for the current soil test and field area.

## When to Apply
- Apply SSP as basal fertilizer at final land preparation or before transplanting.
- {farmer_schedule[crop_name]}
- {farmer_mop[crop_name]}

## Symptoms to Watch
- Low nitrogen: pale green leaves and weak growth.
- Low phosphorus: stunted plants and poor rooting.
- Low potassium: yellowing or scorching along leaf edges and weak stems.

## Warning
- Soil pH: **{pH}**
- Organic carbon: **{organic_carbon}%**
- Season note: {season_text}
- Notes: {"; ".join(warnings)}
- Source basis: {source_summary}
"""

    if role == "Agronomist":
        return f"""## Nutrient Interpretation of the Soil Test
- Available N is **{soil_n} kg/ha**, available P is **{soil_p} kg/ha**, and available K is **{soil_k} kg/ha**.
- Final computed doses for **{field_area} acre** are **{urea} kg urea**, **{ssp} kg SSP**, and **{mop} kg MOP**.
- The recommendation is restricted to **{crop_name}** in **{district}**, **{season}** season, on **{soil_type}** soil.

## Split Application Schedule
- Apply the full **SSP ({ssp} kg)** as basal at puddling or before transplanting.
{agr_schedule[crop_name]}
- Seasonal operations note: {season_text}

## LCC-Based Nitrogen Adjustment
- Use Leaf Colour Chart guidance during vegetative growth and adjust only operational timing, not the final computed quantity.
- If leaf colour remains adequate, avoid unnecessary supplemental N.

## Micronutrient and Soil Reaction Considerations
- Soil pH is **{pH}** and organic carbon is **{organic_carbon}%**.
- Review Zn, S, and Fe behavior under local {crop_name.lower()} conditions, especially under problem soils or nutrient-stress situations.
- Warnings: {"; ".join(warnings)}

## Source-Grounded Justification
- The explanation is grounded in retrieved Tamil Nadu {crop_name.lower()} references from: {source_summary}.
- Retrieved local context was used for timing and interpretation, while fertilizer quantities remained fixed from the calculator.
"""

    total_cost = round((urea * 6.5) + (ssp * 7.0) + (mop * 34.0), 2)
    sale_price = MANAGER_PRICES[crop_name]
    base_yield = calculation["inputs_used"]["target_yield"]
    # sale_price is assumed to be in Rs/kg, while base_yield is in t/ha.
    yield_gain_low = round(field_area * base_yield * 1000 * 0.05 * sale_price * 0.404686, 2)
    yield_gain_high = round(field_area * base_yield * 1000 * 0.12 * sale_price * 0.404686, 2)
    roi_low = round(((yield_gain_low - total_cost) / total_cost) * 100, 2) if total_cost else 0.0
    roi_high = round(((yield_gain_high - total_cost) / total_cost) * 100, 2) if total_cost else 0.0
    return f"""## Final Procurement Quantities
- Urea: **{urea} kg**
- SSP: **{ssp} kg**
- MOP: **{mop} kg**
- Planning basis: **{crop_name} | {district} | {season} | {soil_type}**

## Estimated Fertilizer Cost
- Approximate total fertilizer cost: **Rs {total_cost}**

## Expected Yield Uplift
- Practical balanced-fertilization uplift range: **5% to 12%**, subject to field management and local conditions.
- Seasonal note: {season_text}

## Approximate ROI
- Estimated return range based on an assumed **Rs {sale_price}/kg** farm-gate price: **{roi_low}% to {roi_high}%**

## Bulk Purchase and Scheduling Advice
- Procure fertilizers before transplanting to avoid seasonal price spikes.
{season_ops}
- Warnings: {"; ".join(warnings)}
- Source basis: {source_summary}
"""


def generate_explanation(prompt: str, max_output_tokens: int) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in the .env file.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_completion_tokens": max_output_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    retry_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    data = None

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code in retry_statuses:
                last_error = RuntimeError(f"Groq returned status {response.status_code}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                "Groq is temporarily unavailable. Please try again in a minute."
            ) from exc
    else:
        raise RuntimeError("Groq request failed.") from last_error

    if data is None:
        raise RuntimeError("Groq request failed.") from last_error

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no response choices.")
    finish_reason = choices[0].get("finish_reason", "")
    if finish_reason == "length":
        raise IncompleteResponseError("Groq response was truncated by the token limit.")
    if finish_reason in {"content_filter", "tool_calls"}:
        raise RuntimeError(f"Groq stopped generation with finish reason: {finish_reason}.")
    text = choices[0].get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError(
            f"Groq returned an empty response{f' (finish reason: {finish_reason})' if finish_reason else ''}."
        )
    return text


def generate_role_response(role: str, prompt: str) -> str:
    enhanced_prompt = prompt
    max_output_tokens = MAX_TOKENS_BY_ROLE.get(role, 1500)
    for attempt in range(3):
        response_text = generate_explanation(enhanced_prompt, max_output_tokens=max_output_tokens)
        if response_has_required_sections(role, response_text):
            return response_text
        enhanced_prompt = (
            prompt
            + "\n\nYour previous answer was incomplete."
            + " Return every required section fully using the exact Markdown headings requested."
            + " Do not stop after the first section."
        )
    raise IncompleteResponseError("The explanation model returned an incomplete answer.")


def render_retrieved_sources(documents: list[dict]) -> None:
    st.subheader("Retrieved Source Chunks")
    for index, doc in enumerate(documents, start=1):
        with st.expander(
            f"{index}. {doc['metadata'].get('source', 'Unknown')} | Page {doc['metadata'].get('page', 'Unknown')}",
            expanded=index == 1,
        ):
            st.write(doc["page_content"])


def main() -> None:
    st.set_page_config(page_title="AgroGuide", layout="wide")
    st.title("AgroGuide 🌾")
    st.subheader("Precision Fertilizer Guidance for Tamil Nadu Crops")
    st.caption("Smart, soil-based fertilizer advice tailored for farmers, agronomists, and managers.")
    st.markdown("**Data-Driven Crop Nutrition Advisor**")
    st.info("💡 Enter your soil details and get instant fertilizer recommendations.")

    with st.sidebar:
        st.header("🌱 Farm Context")
        crop = st.selectbox("Crop", CROPS)
        district = st.selectbox("District", DISTRICTS)
        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        season = st.selectbox("Season", SEASONS_BY_CROP[crop])
        role = st.selectbox("Role", ROLES)

    col1, col2 = st.columns(2)
    with col1:
        soil_n = st.number_input("Soil N (kg/ha)", min_value=0.0, value=0.0, step=1.0)
        soil_p = st.number_input("Soil P (kg/ha)", min_value=0.0, value=0.0, step=1.0)
        soil_k = st.number_input("Soil K (kg/ha)", min_value=0.0, value=0.0, step=1.0)
    with col2:
        pH = st.number_input("Soil pH", min_value=0.0, value=7.0, step=0.1, format="%.1f")
        organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, value=0.5, step=0.1, format="%.1f")
        field_area = st.number_input("Field Area (acres)", min_value=0.0, value=1.0, step=0.1, format="%.2f")

    if st.button("Get Recommendation", type="primary"):
        if not vectorstore_exists():
            st.error("Vector store not found. Please run `python ingest.py` first.")
            return

        if field_area <= 0:
            st.error("Field area must be greater than zero.")
            return

        try:
            calculation = calculate_fertilizer_doses(
                soil_n=soil_n,
                soil_p=soil_p,
                soil_k=soil_k,
                field_area_acres=field_area,
                crop=crop,
                season=season,
                pH=pH,
            )

            retrieved_docs = retrieve_context(crop=crop, district=district, season=season, soil_type=soil_type)
            context_text = format_context(retrieved_docs, role=role)

            prompt = build_prompt(
                role,
                crop=crop,
                district=district,
                season=season,
                soil_type=soil_type,
                soil_n=calculation["inputs_used"]["soil_n"],
                soil_p=calculation["inputs_used"]["soil_p"],
                soil_k=calculation["inputs_used"]["soil_k"],
                pH=calculation["inputs_used"]["pH"],
                organic_carbon=organic_carbon,
                field_area_acres=field_area,
                urea_kg=calculation["urea_kg"],
                ssp_kg=calculation["ssp_kg"],
                mop_kg=calculation["mop_kg"],
                warnings=format_warnings(calculation["warnings"]),
                context=context_text,
            )

            try:
                response_text = generate_role_response(role, prompt)
            except IncompleteResponseError:
                response_text = generate_fallback_response(
                    role=role,
                    crop=crop,
                    calculation=calculation,
                    district=district,
                    season=season,
                    soil_type=soil_type,
                    field_area=field_area,
                    pH=pH,
                    organic_carbon=organic_carbon,
                    documents=retrieved_docs,
                )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("🌿 Urea (kg)", calculation["urea_kg"])
            metric_col2.metric("🧪 SSP (kg)", calculation["ssp_kg"])
            metric_col3.metric("🌾 MOP (kg)", calculation["mop_kg"])

            zero_dose_notes = build_zero_dose_notes(calculation)
            for note in zero_dose_notes:
                st.info(note)

            st.caption(
                f"Target yield used for calculation: {calculation['inputs_used']['target_yield']} t/ha ({crop}, {season})."
            )

            if calculation["warnings"]:
                for warning in calculation["warnings"]:
                    st.warning(warning)

            st.subheader(f"{role} Recommendation")
            st.markdown(response_text)
            st.markdown("---")

            render_retrieved_sources(retrieved_docs)

        except Exception as exc:
            st.error(f"Could not generate recommendation: {exc}")


if __name__ == "__main__":
    main()
