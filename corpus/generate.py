"""Synthetic payer-policy corpus generator.

Everything here is invented. The payers, plans, policy bulletins, riders and
clause text do not correspond to any real organisation or document. The formats
are modelled on public medical-policy bulletins so the retrieval problem is
realistic; the content is not.

The corpus deliberately contains the failure modes a coverage-determination
agent has to survive:

  T1  stale-version trap     a policy whose criteria changed on a known date
  T2  rider override         a plan rider that overrides its own base policy
  T3  lookalike-but-not      two policies with near-identical titles, different criteria
  T4  compositional criteria "A and B unless C" structures
  T5  contradiction          a provider FAQ that conflicts with the bulletin
  T6  not-in-corpus          procedures with no governing policy (refusal cases)

Run:  python -m corpus.generate
Out:  corpus/generated/corpus.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent / "generated"

# --------------------------------------------------------------------------
# payers and plans
# --------------------------------------------------------------------------

PAYERS = [
    {"payer_id": "MHP", "name": "Meridian Health Plan"},
    {"payer_id": "CSM", "name": "Cascadia Mutual"},
    {"payer_id": "NBA", "name": "Northgate Benefit Alliance"},
]

PLANS = [
    # T2 lives here: GOLD carries the step-therapy waiver rider, BASE does not.
    {"plan_id": "MHP-PPO-GOLD", "payer_id": "MHP", "name": "Meridian PPO Gold",
     "rider_ids": ["MHP-RID-STEP-WAIVE", "MHP-RID-DME-EXPAND"]},
    {"plan_id": "MHP-HMO-BASE", "payer_id": "MHP", "name": "Meridian HMO Base",
     "rider_ids": []},
    {"plan_id": "CSM-PPO-STD", "payer_id": "CSM", "name": "Cascadia PPO Standard",
     "rider_ids": ["CSM-RID-SLEEP-NARROW"]},
    {"plan_id": "CSM-EPO-VALUE", "payer_id": "CSM", "name": "Cascadia EPO Value",
     "rider_ids": []},
    {"plan_id": "NBA-PPO-PREM", "payer_id": "NBA", "name": "Northgate PPO Premier",
     "rider_ids": []},
]

# --------------------------------------------------------------------------
# code sets
# --------------------------------------------------------------------------

CODES = [
    ("95249", "CPT", "Ambulatory continuous glucose monitoring, patient-owned equipment, initial hookup and training"),
    ("95250", "CPT", "Ambulatory continuous glucose monitoring, physician-owned equipment, up to 72 hours"),
    ("95251", "CPT", "Ambulatory continuous glucose monitoring, interpretation and report"),
    ("A9276", "HCPCS", "Disposable sensor, continuous glucose monitoring system, one unit = one day supply"),
    ("E0607", "HCPCS", "Home blood glucose monitor"),
    ("95810", "CPT", "Polysomnography, sleep staging, attended by technologist"),
    ("95806", "CPT", "Sleep study, unattended, home-based, simultaneous recording"),
    ("E0601", "HCPCS", "Continuous positive airway pressure (CPAP) device"),
    ("22551", "CPT", "Arthrodesis, anterior interbody, cervical below C2"),
    ("63030", "CPT", "Laminotomy with decompression of nerve root, lumbar, one interspace"),
    ("97110", "CPT", "Therapeutic exercise to develop strength and endurance"),
    ("J1745", "HCPCS", "Injection, infliximab, 10 mg"),
    ("J0135", "HCPCS", "Injection, adalimumab, 20 mg"),
    ("70553", "CPT", "MRI brain with and without contrast"),
    ("77067", "CPT", "Screening mammography, bilateral"),
    ("0501T", "CPT", "Noninvasive estimated coronary fractional flow reserve from CT data"),
    ("E1399", "HCPCS", "Durable medical equipment, miscellaneous"),
    ("31241", "CPT", "Nasal endoscopy, surgical, with ligation of sphenopalatine artery"),
]

ICD = [
    ("E10.9", "ICD-10", "Type 1 diabetes mellitus without complications"),
    ("E11.9", "ICD-10", "Type 2 diabetes mellitus without complications"),
    ("E11.65", "ICD-10", "Type 2 diabetes mellitus with hyperglycemia"),
    ("G47.33", "ICD-10", "Obstructive sleep apnea (adult) (pediatric)"),
    ("M54.16", "ICD-10", "Radiculopathy, lumbar region"),
    ("M05.79", "ICD-10", "Rheumatoid arthritis with rheumatoid factor, multiple sites"),
    ("K50.90", "ICD-10", "Crohn's disease, unspecified, without complications"),
    ("R51.9", "ICD-10", "Headache, unspecified"),
]


def _clause(policy_id: str, version: str, n: int, section: str, text: str) -> dict[str, Any]:
    return {
        "clause_id": f"{policy_id}.{version}.C{n}",
        "section": section,
        "text": " ".join(text.split()),
    }


# --------------------------------------------------------------------------
# policy bulletins
#
# Each policy has one or more versions with disjoint effective ranges. A clause
# is the atomic retrievable and citable unit — the Verifier fetches by clause_id
# and checks the quoted text is verbatim and in effect on the as_of date.
# --------------------------------------------------------------------------

POLICIES: list[dict[str, Any]] = []


def policy(policy_id, payer_id, title, doc_type, codes, versions, notes=""):
    POLICIES.append({
        "policy_id": policy_id,
        "payer_id": payer_id,
        "title": title,
        "doc_type": doc_type,  # medical_policy_bulletin | provider_faq
        "codes": codes,
        "notes": notes,
        "versions": versions,
    })


# ---- T1: stale-version trap -------------------------------------------------
# Criteria changed 2024-07-01. A determination dated before that MUST cite v1.
policy(
    "MHP-MP-0142", "MHP", "Continuous Glucose Monitoring (Personal Use)",
    "medical_policy_bulletin",
    ["95249", "95251", "A9276", "E0607"],
    notes="T1 stale-version trap; T4 compositional criteria",
    versions=[
        {"version": "v1", "effective_start": "2023-01-01", "effective_end": "2024-06-30", "clauses": [
            _clause("MHP-MP-0142", "v1", 1, "Scope",
                    "This policy governs personal-use continuous glucose monitoring (CGM) systems "
                    "dispensed for home use, including sensors billed under A9276 and initial "
                    "training billed under 95249."),
            _clause("MHP-MP-0142", "v1", 2, "Coverage Criteria",
                    "Personal-use CGM is considered medically necessary when the member has a "
                    "documented diagnosis of Type 1 diabetes mellitus AND the treating clinician "
                    "documents that the member performs a minimum of four (4) self-monitored blood "
                    "glucose fingerstick tests per day, UNLESS the member has documented "
                    "hypoglycemia unawareness, in which case the fingerstick frequency requirement "
                    "does not apply."),
            _clause("MHP-MP-0142", "v1", 3, "Coverage Criteria",
                    "For members with Type 2 diabetes mellitus, personal-use CGM is considered "
                    "medically necessary only when the member is on an intensive insulin regimen "
                    "of three (3) or more injections per day."),
            _clause("MHP-MP-0142", "v1", 4, "Limitations",
                    "Sensor supply is limited to one (1) unit per day. Quantities in excess of "
                    "31 units per 31-day period require additional documentation."),
            _clause("MHP-MP-0142", "v1", 5, "Exclusions",
                    "CGM systems dispensed solely for gestational diabetes are excluded under this "
                    "version of the policy."),
        ]},
        {"version": "v2", "effective_start": "2024-07-01", "effective_end": None, "clauses": [
            _clause("MHP-MP-0142", "v2", 1, "Scope",
                    "This policy governs personal-use continuous glucose monitoring (CGM) systems "
                    "dispensed for home use, including sensors billed under A9276 and initial "
                    "training billed under 95249."),
            _clause("MHP-MP-0142", "v2", 2, "Coverage Criteria",
                    "Personal-use CGM is considered medically necessary when the member has a "
                    "documented diagnosis of Type 1 diabetes mellitus. The prior requirement for a "
                    "minimum of four self-monitored fingerstick tests per day has been removed "
                    "effective July 1, 2024."),
            _clause("MHP-MP-0142", "v2", 3, "Coverage Criteria",
                    "For members with Type 2 diabetes mellitus, personal-use CGM is considered "
                    "medically necessary when the member is treated with any insulin regimen, or "
                    "has a documented history of level 2 hypoglycemia."),
            _clause("MHP-MP-0142", "v2", 4, "Limitations",
                    "Sensor supply is limited to one (1) unit per day. Quantities in excess of "
                    "31 units per 31-day period require additional documentation."),
            _clause("MHP-MP-0142", "v2", 5, "Coverage Criteria",
                    "Personal-use CGM for gestational diabetes is considered medically necessary "
                    "when initiated after 20 weeks gestation with documented insulin therapy."),
        ]},
    ],
)

# ---- T3: lookalike-but-not --------------------------------------------------
policy(
    "MHP-MP-0143", "MHP", "Professional (Intermittent) Glucose Sensor Monitoring",
    "medical_policy_bulletin",
    ["95250", "95251"],
    notes="T3 lookalike of MHP-MP-0142 — different criteria, different codes",
    versions=[
        {"version": "v1", "effective_start": "2022-06-01", "effective_end": None, "clauses": [
            _clause("MHP-MP-0143", "v1", 1, "Scope",
                    "This policy governs professional, physician-owned intermittent glucose sensor "
                    "monitoring billed under 95250. It does not govern personal-use continuous "
                    "glucose monitoring, which is addressed in policy MHP-MP-0142."),
            _clause("MHP-MP-0143", "v1", 2, "Coverage Criteria",
                    "Professional intermittent glucose sensor monitoring is considered medically "
                    "necessary once per 12-month period for members with Type 1 or Type 2 diabetes "
                    "mellitus whose glycemic control is inadequate as evidenced by an A1c of 8.0 "
                    "percent or greater."),
            _clause("MHP-MP-0143", "v1", 3, "Limitations",
                    "More than one professional monitoring session per rolling 12-month period is "
                    "considered not medically necessary."),
        ]},
    ],
)

# ---- T2 base policy: step therapy, overridden by a rider --------------------
policy(
    "MHP-MP-0210", "MHP", "Biologic Therapy for Inflammatory Conditions",
    "medical_policy_bulletin",
    ["J1745", "J0135"],
    notes="T2 base policy overridden by MHP-RID-STEP-WAIVE on MHP-PPO-GOLD",
    versions=[
        {"version": "v3", "effective_start": "2024-01-01", "effective_end": None, "clauses": [
            _clause("MHP-MP-0210", "v3", 1, "Scope",
                    "This policy governs biologic disease-modifying therapy for rheumatoid "
                    "arthritis and inflammatory bowel disease, including infliximab (J1745) and "
                    "adalimumab (J0135)."),
            _clause("MHP-MP-0210", "v3", 2, "Coverage Criteria",
                    "Biologic therapy is considered medically necessary when the member has a "
                    "confirmed diagnosis of moderate to severe rheumatoid arthritis AND has "
                    "completed a trial of at least one conventional synthetic DMARD of no less "
                    "than three (3) months duration with inadequate response, UNLESS conventional "
                    "synthetic DMARD therapy is contraindicated."),
            _clause("MHP-MP-0210", "v3", 3, "Coverage Criteria",
                    "For Crohn's disease, biologic therapy is considered medically necessary when "
                    "the member has moderate to severe active disease and has failed or is "
                    "intolerant to corticosteroid therapy."),
            _clause("MHP-MP-0210", "v3", 4, "Step Therapy",
                    "Step therapy requirements described in this policy may be waived where an "
                    "applicable plan rider expressly provides for waiver. In the event of conflict "
                    "between this bulletin and a plan rider, the rider controls."),
        ]},
    ],
)

# ---- T5: contradiction — a provider FAQ that disagrees with the bulletin ----
policy(
    "MHP-FAQ-0210", "MHP", "Provider FAQ: Biologic Therapy Prior Authorization",
    "provider_faq",
    ["J1745", "J0135"],
    notes="T5 contradiction: asserts a 6-month DMARD trial vs the bulletin's 3 months",
    versions=[
        {"version": "v1", "effective_start": "2024-02-01", "effective_end": None, "clauses": [
            _clause("MHP-FAQ-0210", "v1", 1, "Frequently Asked Questions",
                    "Q: How long must a member try a conventional DMARD before biologic therapy is "
                    "approved? A: A trial of at least six (6) months is required before biologic "
                    "therapy will be authorized."),
            _clause("MHP-FAQ-0210", "v1", 2, "Frequently Asked Questions",
                    "Q: Does a plan rider change the step therapy requirement? A: Riders do not "
                    "affect step therapy requirements."),
        ]},
    ],
)

policy(
    "CSM-MP-0331", "CSM", "Attended Polysomnography and Home Sleep Apnea Testing",
    "medical_policy_bulletin",
    ["95810", "95806", "E0601"],
    notes="T4 compositional criteria; narrowed by CSM-RID-SLEEP-NARROW on CSM-PPO-STD",
    versions=[
        {"version": "v1", "effective_start": "2023-03-01", "effective_end": "2025-02-28", "clauses": [
            _clause("CSM-MP-0331", "v1", 1, "Scope",
                    "This policy governs attended in-laboratory polysomnography (95810) and "
                    "unattended home sleep apnea testing (95806) for suspected obstructive sleep "
                    "apnea in adults."),
            _clause("CSM-MP-0331", "v1", 2, "Coverage Criteria",
                    "Home sleep apnea testing is considered medically necessary when the member "
                    "has a high pre-test probability of moderate to severe obstructive sleep apnea "
                    "AND has no significant comorbid condition, UNLESS the member has documented "
                    "congestive heart failure, chronic obstructive pulmonary disease, or "
                    "neuromuscular disease, in which case attended polysomnography is required."),
            _clause("CSM-MP-0331", "v1", 3, "Coverage Criteria",
                    "Attended in-laboratory polysomnography is considered medically necessary when "
                    "home sleep apnea testing is technically inadequate or negative and clinical "
                    "suspicion of obstructive sleep apnea remains."),
            _clause("CSM-MP-0331", "v1", 4, "Limitations",
                    "Repeat testing within a 12-month period requires documentation of a change in "
                    "clinical status or a weight change of at least ten (10) percent."),
        ]},
        {"version": "v2", "effective_start": "2025-03-01", "effective_end": None, "clauses": [
            _clause("CSM-MP-0331", "v2", 1, "Scope",
                    "This policy governs attended in-laboratory polysomnography (95810) and "
                    "unattended home sleep apnea testing (95806) for suspected obstructive sleep "
                    "apnea in adults and in members aged 13 years and older."),
            _clause("CSM-MP-0331", "v2", 2, "Coverage Criteria",
                    "Home sleep apnea testing is considered medically necessary as the initial "
                    "diagnostic study for members with suspected obstructive sleep apnea, "
                    "regardless of comorbid status, UNLESS the member requires attended titration."),
            _clause("CSM-MP-0331", "v2", 3, "Limitations",
                    "Repeat testing within a 12-month period requires documentation of a change in "
                    "clinical status or a weight change of at least ten (10) percent."),
        ]},
    ],
)

policy(
    "CSM-MP-0405", "CSM", "Lumbar Decompression and Spinal Fusion",
    "medical_policy_bulletin",
    ["63030", "22551", "97110"],
    versions=[
        {"version": "v2", "effective_start": "2024-04-01", "effective_end": None, "clauses": [
            _clause("CSM-MP-0405", "v2", 1, "Scope",
                    "This policy governs lumbar laminotomy with nerve root decompression (63030) "
                    "and cervical anterior interbody arthrodesis (22551)."),
            _clause("CSM-MP-0405", "v2", 2, "Coverage Criteria",
                    "Lumbar decompression is considered medically necessary when the member has "
                    "radiculopathy corroborated by advanced imaging AND has completed at least six "
                    "(6) weeks of conservative therapy including supervised physical therapy, "
                    "UNLESS the member presents with progressive neurologic deficit or cauda equina "
                    "syndrome, in which case conservative therapy is not required."),
            _clause("CSM-MP-0405", "v2", 3, "Exclusions",
                    "Spinal fusion performed solely for axial low back pain without instability or "
                    "deformity is considered not medically necessary."),
        ]},
    ],
)

policy(
    "NBA-MP-0512", "NBA", "Advanced Imaging: Brain MRI",
    "medical_policy_bulletin",
    ["70553"],
    versions=[
        {"version": "v1", "effective_start": "2023-09-01", "effective_end": None, "clauses": [
            _clause("NBA-MP-0512", "v1", 1, "Scope",
                    "This policy governs magnetic resonance imaging of the brain with and without "
                    "contrast (70553)."),
            _clause("NBA-MP-0512", "v1", 2, "Coverage Criteria",
                    "Brain MRI with and without contrast is considered medically necessary for "
                    "new-onset headache accompanied by at least one red-flag feature, including "
                    "papilledema, focal neurologic deficit, headache awakening the member from "
                    "sleep, or onset after age 50."),
            _clause("NBA-MP-0512", "v1", 3, "Exclusions",
                    "Brain MRI for uncomplicated primary headache without red-flag features is "
                    "considered not medically necessary."),
        ]},
    ],
)

policy(
    "NBA-MP-0640", "NBA", "Preventive Screening Mammography",
    "medical_policy_bulletin",
    ["77067"],
    versions=[
        {"version": "v1", "effective_start": "2022-01-01", "effective_end": None, "clauses": [
            _clause("NBA-MP-0640", "v1", 1, "Coverage Criteria",
                    "Screening mammography (77067) is covered as a preventive service once per "
                    "calendar year for members aged 40 years and older, with no member cost share."),
            _clause("NBA-MP-0640", "v1", 2, "Limitations",
                    "Additional diagnostic imaging arising from an abnormal screening result is "
                    "adjudicated under the diagnostic imaging benefit and may be subject to cost "
                    "share."),
        ]},
    ],
)

policy(
    "CSM-MP-0702", "CSM", "Investigational and Experimental Services",
    "medical_policy_bulletin",
    ["0501T", "E1399"],
    notes="T6 support: names a category as investigational without governing specific procedures",
    versions=[
        {"version": "v4", "effective_start": "2024-01-01", "effective_end": None, "clauses": [
            _clause("CSM-MP-0702", "v4", 1, "Coverage Criteria",
                    "Services designated as investigational or experimental are excluded from "
                    "coverage. Noninvasive estimated coronary fractional flow reserve derived from "
                    "computed tomography data (0501T) is designated investigational."),
            _clause("CSM-MP-0702", "v4", 2, "Limitations",
                    "Miscellaneous durable medical equipment billed under E1399 requires "
                    "individual consideration and is not covered under a standing policy."),
        ]},
    ],
)

# --------------------------------------------------------------------------
# riders — these override base policy for specific plans
# --------------------------------------------------------------------------

RIDERS = [
    {
        "rider_id": "MHP-RID-STEP-WAIVE",
        "payer_id": "MHP",
        "plan_id": "MHP-PPO-GOLD",
        "title": "Gold Plan Step Therapy Waiver Rider",
        "overrides_policy_ids": ["MHP-MP-0210"],
        "effective_start": "2024-01-01",
        "effective_end": None,
        "clauses": [
            {"clause_id": "MHP-RID-STEP-WAIVE.C1", "section": "Rider",
             "text": "For members enrolled in Meridian PPO Gold, the conventional synthetic DMARD "
                     "trial requirement described in policy MHP-MP-0210 is waived. Biologic therapy "
                     "may be authorized as first-line treatment upon confirmed diagnosis."},
            {"clause_id": "MHP-RID-STEP-WAIVE.C2", "section": "Rider",
             "text": "This rider does not waive the requirement for a confirmed diagnosis of "
                     "moderate to severe disease."},
        ],
    },
    {
        "rider_id": "MHP-RID-DME-EXPAND",
        "payer_id": "MHP",
        "plan_id": "MHP-PPO-GOLD",
        "title": "Gold Plan Expanded DME Rider",
        "overrides_policy_ids": ["MHP-MP-0142"],
        "effective_start": "2023-01-01",
        "effective_end": None,
        "clauses": [
            {"clause_id": "MHP-RID-DME-EXPAND.C1", "section": "Rider",
             "text": "For members enrolled in Meridian PPO Gold, the sensor quantity limitation in "
                     "policy MHP-MP-0142 is increased to allow up to 40 units per 31-day period "
                     "without additional documentation."},
        ],
    },
    {
        "rider_id": "CSM-RID-SLEEP-NARROW",
        "payer_id": "CSM",
        "plan_id": "CSM-PPO-STD",
        "title": "Standard PPO Sleep Study Site-of-Service Rider",
        "overrides_policy_ids": ["CSM-MP-0331"],
        "effective_start": "2024-01-01",
        "effective_end": None,
        "clauses": [
            {"clause_id": "CSM-RID-SLEEP-NARROW.C1", "section": "Rider",
             "text": "For members enrolled in Cascadia PPO Standard, attended in-laboratory "
                     "polysomnography requires prior authorization in all cases, including where "
                     "policy CSM-MP-0331 would otherwise permit it without prior authorization."},
        ],
    },
]


def build() -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []

    for p in POLICIES:
        for v in p["versions"]:
            for c in v["clauses"]:
                clauses.append({
                    **c,
                    "policy_id": p["policy_id"],
                    "policy_title": p["title"],
                    "doc_type": p["doc_type"],
                    "payer_id": p["payer_id"],
                    "plan_id": None,
                    "version": v["version"],
                    "effective_start": v["effective_start"],
                    "effective_end": v["effective_end"],
                    "codes": p["codes"],
                    "source": "policy",
                })

    for r in RIDERS:
        for c in r["clauses"]:
            clauses.append({
                **c,
                "policy_id": r["rider_id"],
                "policy_title": r["title"],
                "doc_type": "plan_rider",
                "payer_id": r["payer_id"],
                "plan_id": r["plan_id"],
                "version": "v1",
                "effective_start": r["effective_start"],
                "effective_end": r["effective_end"],
                "codes": [],
                "source": "rider",
                "overrides_policy_ids": r["overrides_policy_ids"],
            })

    return {
        "payers": PAYERS,
        "plans": PLANS,
        "policies": POLICIES,
        "riders": RIDERS,
        "clauses": clauses,
        "codes": [{"code": c, "system": s, "descriptor": d} for c, s, d in CODES + ICD],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = build()
    out = OUT_DIR / "corpus.json"
    out.write_text(json.dumps(corpus, indent=2))
    print(
        f"wrote {out}\n"
        f"  payers   {len(corpus['payers'])}\n"
        f"  plans    {len(corpus['plans'])}\n"
        f"  policies {len(corpus['policies'])} "
        f"({sum(len(p['versions']) for p in corpus['policies'])} versions)\n"
        f"  riders   {len(corpus['riders'])}\n"
        f"  clauses  {len(corpus['clauses'])}\n"
        f"  codes    {len(corpus['codes'])}"
    )


if __name__ == "__main__":
    main()
