import json
from pathlib import Path

import pandas as pd
import requests


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


def fetch_tcga_case_metadata(
    case_ids: list[str],
    chunk_size: int = 200,
) -> pd.DataFrame:
    """
    Retrieve TCGA case-level metadata from the GDC API.

    Uses POST so large case-ID filters are sent in the request body
    rather than in an excessively long URL.
    """
    case_ids = sorted(
        {
            str(case_id).strip()
            for case_id in case_ids
            if pd.notna(case_id)
        }
    )

    records = []

    fields = [
        "submitter_id",
        "case_id",
        "project.project_id",
        "primary_site",
        "disease_type",
        "diagnoses.primary_diagnosis",
        "diagnoses.morphology",
        "diagnoses.tumor_grade",
        "diagnoses.ajcc_pathologic_stage",
    ]

    for start in range(0, len(case_ids), chunk_size):
        current_ids = case_ids[start:start + chunk_size]

        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": current_ids,
            },
        }

        payload = {
            "filters": filters,
            "fields": ",".join(fields),
            "expand": "diagnoses,project",
            "format": "JSON",
            "size": len(current_ids),
        }

        response = requests.post(
            GDC_CASES_ENDPOINT,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        hits = response.json()["data"]["hits"]

        for hit in hits:
            project = hit.get("project") or {}
            diagnoses = hit.get("diagnoses") or [{}]

            for diagnosis_index, diagnosis in enumerate(diagnoses):
                records.append({
                    "case_id": hit.get("submitter_id"),
                    "gdc_case_uuid": hit.get("case_id"),
                    "project_id": project.get("project_id"),
                    "primary_site": hit.get("primary_site"),
                    "disease_type": hit.get("disease_type"),
                    "diagnosis_index": diagnosis_index,
                    "primary_diagnosis": diagnosis.get(
                        "primary_diagnosis"
                    ),
                    "morphology": diagnosis.get("morphology"),
                    "tumor_grade": diagnosis.get("tumor_grade"),
                    "pathologic_stage": diagnosis.get(
                        "ajcc_pathologic_stage"
                    ),
                })

        print(
            f"Fetched {min(start + chunk_size, len(case_ids))}"
            f"/{len(case_ids)} cases"
        )

    return pd.DataFrame(records)


def main():
    input_csv = Path("./tcga.csv")
    output_csv = Path("./tcga_slides_with_metadata.csv")

    slides = pd.read_csv(input_csv)

    required_columns = {"case_id", "slide_id"}
    missing = required_columns - set(slides.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    metadata = fetch_tcga_case_metadata(
        slides["case_id"].dropna().astype(str).tolist()
    )

    # If multiple diagnosis rows exist, initially retain the first.
    # Review duplicates rather than silently assuming all diagnoses
    # are equivalent.
    duplicate_cases = metadata[
        metadata.duplicated("case_id", keep=False)
    ]

    if not duplicate_cases.empty:
        duplicate_cases.to_csv(
            "tcga_cases_with_multiple_diagnoses.csv",
            index=False,
        )
        print(
            "Warning: some cases have multiple diagnosis records. "
            "They were written to "
            "'tcga_cases_with_multiple_diagnoses.csv'."
        )

    one_row_per_case = (
        metadata
        .sort_values(["case_id", "diagnosis_index"])
        .drop_duplicates("case_id", keep="first")
    )

    merged = slides.drop(
        columns=["label"],
        errors="ignore",
    ).merge(
        one_row_per_case,
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    missing_metadata = merged["project_id"].isna()

    if missing_metadata.any():
        missing_ids = sorted(
            merged.loc[missing_metadata, "case_id"].unique()
        )
        print(
            f"Metadata was not found for {len(missing_ids)} cases:"
        )
        print(missing_ids[:20])

    merged.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()