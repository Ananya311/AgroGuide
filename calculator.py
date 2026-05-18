from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONSTANTS_PATH = BASE_DIR / "data" / "constants.json"

ACRE_TO_HECTARE = 0.404686
DEFAULT_SOIL_FERTILITY = {"N": 180.0, "P": 22.0, "K": 240.0}
CROP_NAME_MAP = {
    "Rice": "rice",
    "Maize": "maize",
    "Sorghum": "sorghum",
    "Pearl Millet": "pearl_millet",
}

@lru_cache(maxsize=1)
def load_constants() -> dict:
    with CONSTANTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_soil_value(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _get_crop_config(constants: dict, crop: str) -> dict:
    crop_key = CROP_NAME_MAP.get(crop, crop.lower())
    crops = constants["crops"]
    if crop_key not in crops:
        raise ValueError(f"Unsupported crop: {crop}")
    return crops[crop_key]


def _resolve_target_yield(crop_config: dict, season: str | None, warnings: list[str]) -> float:
    seasonal_yields = crop_config.get("target_yield_by_season", {})
    if season and season in seasonal_yields:
        return float(seasonal_yields[season])
    if season and seasonal_yields and season not in seasonal_yields:
        warnings.append(
            f"Season '{season}' was not found for the selected crop, so the default target yield was used."
        )
    return float(crop_config["target_yield"])


def calculate_fertilizer_doses(
    soil_n: float | None,
    soil_p: float | None,
    soil_k: float | None,
    field_area_acres: float,
    crop: str = "rice",
    season: str | None = None,
    pH: float | None = None,
) -> dict:
    constants = load_constants()
    crop_config = _get_crop_config(constants, crop)
    warnings: list[str] = []
    target_yield = _resolve_target_yield(crop_config, season, warnings)

    soil_n = _normalize_soil_value(soil_n)
    soil_p = _normalize_soil_value(soil_p)
    soil_k = _normalize_soil_value(soil_k)
    if field_area_acres is not None and float(field_area_acres) < 0:
        raise ValueError("Field area cannot be negative.")
    field_area_acres = float(field_area_acres or 0.0)
    hectares = field_area_acres * ACRE_TO_HECTARE

    if field_area_acres == 0:
        warnings.append("Field area is zero, so all fertilizer quantities will be zero.")

    if soil_n == 0 and soil_p == 0 and soil_k == 0:
        soil_n = DEFAULT_SOIL_FERTILITY["N"]
        soil_p = DEFAULT_SOIL_FERTILITY["P"]
        soil_k = DEFAULT_SOIL_FERTILITY["K"]
        warnings.append(
            "All soil values were zero, so average soil fertility defaults were used: N 180, P 22, K 240 kg/ha."
        )

    total_requirement = {
        nutrient: target_yield * crop_config["nutrient_per_ton"][nutrient]
        for nutrient in ("N", "P", "K")
    }

    soil_supply = {"N": soil_n, "P": soil_p, "K": soil_k}
    net_requirement_per_ha = {}
    for nutrient in ("N", "P", "K"):
        raw_requirement = (
            total_requirement[nutrient] - soil_supply[nutrient]
        ) / constants["efficiency"][nutrient]
        net_requirement_per_ha[nutrient] = max(raw_requirement, 0.0)

    fertilizer_per_ha = {
        "urea_kg": net_requirement_per_ha["N"] / constants["fertilizers"]["urea"],
        "ssp_kg": net_requirement_per_ha["P"] / constants["fertilizers"]["ssp"],
        "mop_kg": net_requirement_per_ha["K"] / constants["fertilizers"]["mop"],
    }

    scaled_fertilizer = {
        name: round(max(value * hectares, 0.0), 2)
        for name, value in fertilizer_per_ha.items()
    }

    if pH is not None:
        pH = float(pH)
        if pH < 5.5:
            warnings.append(
                "Low soil pH detected. Apply suitable liming or soil amendment before fertilizer application."
            )
        elif pH > 8.5:
            warnings.append(
                "High soil pH detected. Apply suitable reclamation amendment before fertilizer application."
            )

    return {
        "urea_kg": scaled_fertilizer["urea_kg"],
        "ssp_kg": scaled_fertilizer["ssp_kg"],
        "mop_kg": scaled_fertilizer["mop_kg"],
        "warnings": warnings,
        "inputs_used": {
            "crop": crop,
            "soil_n": soil_n,
            "soil_p": soil_p,
            "soil_k": soil_k,
            "field_area_acres": field_area_acres,
            "field_area_hectares": round(hectares, 4),
            "season": season,
            "target_yield": target_yield,
            "pH": pH,
        },
        "per_hectare_requirements": {
            "total_requirement": total_requirement,
            "net_requirement": {
                nutrient: round(value, 2) for nutrient, value in net_requirement_per_ha.items()
            },
        },
    }
