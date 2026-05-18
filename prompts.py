from __future__ import annotations


FERTILIZER_GUARDRAIL = (
    "The fertilizer quantities provided are final calculated values. "
    "Do NOT change, round, or suggest different quantities. Only explain them."
)


FARMER_PROMPT = """
You are preparing a fertilizer explanation for a {crop} farmer in Tamil Nadu.
{guardrail}

Instructions:
- Respond only in English.
- Start directly with the recommendation. Do not greet the user.
- Do not add a title, heading, or repeated label like "Farmer Recommendation" or "Fertilizer Recommendation".
- Do not restate the location, area, season, or other input details in an opening sentence.
- Use simple plain language.
- Avoid technical terms and laboratory jargon.
- Focus on what to buy, how much to buy, and when to apply it.
- Mention visible crop symptoms to watch for.
- Stay strictly within {crop} cultivation in Tamil Nadu.
- Use only the retrieved context when citing local guidance.
- Keep the answer well-structured with short headings and bullet points.

Farm details:
- Crop: {crop}
- District: {district}
- Season: {season}
- Soil type: {soil_type}
- Soil N: {soil_n} kg/ha
- Soil P: {soil_p} kg/ha
- Soil K: {soil_k} kg/ha
- Soil pH: {pH}
- Organic carbon: {organic_carbon}
- Field area: {field_area_acres} acres

Final fertilizer quantities:
- Urea: {urea_kg} kg
- SSP: {ssp_kg} kg
- MOP: {mop_kg} kg

Warnings:
{warnings}

Retrieved context:
{context}

Write a practical answer with:
Use exactly these Markdown headings:
## Fertilizers to Buy
## When to Apply
## Symptoms to Watch
## Warning
Return only these four sections and nothing before them.
"""


AGRONOMIST_PROMPT = """
You are preparing a fertilizer explanation for a {crop} agronomist in Tamil Nadu.
{guardrail}

Instructions:
- Respond only in English.
- Start directly with the recommendation. Do not greet the user.
- Do not add a title, heading, or repeated label like "Agronomist Recommendation" or "Fertilizer Recommendation".
- Do not restate the location, area, season, or other input details in an opening sentence.
- Use technical agronomic language.
- Include precise split application schedules using crop-appropriate stages or timings.
- Reference LCC-based nitrogen management.
- Discuss micronutrient interactions and likely constraints.
- Stay strictly within {crop} cultivation in Tamil Nadu.
- Ground the explanation in the retrieved context.
- Keep the answer well-structured with short headings and bullet points.

Farm details:
- Crop: {crop}
- District: {district}
- Season: {season}
- Soil type: {soil_type}
- Soil N: {soil_n} kg/ha
- Soil P: {soil_p} kg/ha
- Soil K: {soil_k} kg/ha
- Soil pH: {pH}
- Organic carbon: {organic_carbon}
- Field area: {field_area_acres} acres

Final fertilizer quantities:
- Urea: {urea_kg} kg
- SSP: {ssp_kg} kg
- MOP: {mop_kg} kg

Warnings:
{warnings}

Retrieved context:
{context}

Write a structured advisory covering:
Use exactly these Markdown headings:
## Nutrient Interpretation of the Soil Test
## Split Application Schedule
## LCC-Based Nitrogen Adjustment
## Micronutrient and Soil Reaction Considerations
## Source-Grounded Justification
Return only these five sections and nothing before them.
"""


MANAGER_PROMPT = """
You are preparing a fertilizer explanation for a {crop} farm manager in Tamil Nadu.
{guardrail}

Instructions:
- Respond only in English.
- Start directly with the recommendation. Do not greet the user.
- Do not add a title, heading, or repeated label like "Manager Recommendation" or "Fertilizer Recommendation".
- Do not restate the location, area, season, or other input details in an opening sentence.
- Use clear managerial language.
- Calculate total input cost in rupees using these approximate market rates:
  Urea Rs 6.5/kg, SSP Rs 7.0/kg, MOP Rs 34.0/kg.
- Estimate yield uplift percentage as a practical range based on balanced fertilization and the retrieved context.
- Compute a simple return on investment using a crop-appropriate farm-gate price assumption.
- Suggest bulk procurement and operations planning advice.
- Stay strictly within {crop} cultivation in Tamil Nadu.
- Do not alter the fertilizer quantities.
- Keep the answer well-structured with short headings and bullet points.

Farm details:
- Crop: {crop}
- District: {district}
- Season: {season}
- Soil type: {soil_type}
- Soil N: {soil_n} kg/ha
- Soil P: {soil_p} kg/ha
- Soil K: {soil_k} kg/ha
- Soil pH: {pH}
- Organic carbon: {organic_carbon}
- Field area: {field_area_acres} acres

Final fertilizer quantities:
- Urea: {urea_kg} kg
- SSP: {ssp_kg} kg
- MOP: {mop_kg} kg

Warnings:
{warnings}

Retrieved context:
{context}

Write a concise management brief with:
Use exactly these Markdown headings:
## Final Procurement Quantities
## Estimated Fertilizer Cost
## Expected Yield Uplift
## Approximate ROI
## Bulk Purchase and Scheduling Advice
Return only these five sections and nothing before them.
"""


ROLE_PROMPTS = {
    "Farmer": FARMER_PROMPT,
    "Agronomist": AGRONOMIST_PROMPT,
    "Manager": MANAGER_PROMPT,
}


def _escape_prompt_kwargs(values: dict) -> dict:
    return {
        key: str(value).replace("{", "{{").replace("}", "}}")
        for key, value in values.items()
    }


def build_prompt(role: str, **kwargs: str) -> str:
    if role not in ROLE_PROMPTS:
        raise ValueError(f"Unsupported role: {role}")
    template = ROLE_PROMPTS[role]
    escaped_kwargs = _escape_prompt_kwargs(kwargs)
    return template.format(guardrail=FERTILIZER_GUARDRAIL, **escaped_kwargs)
