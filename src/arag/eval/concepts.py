"""Controlled concept vocabulary — the cross-version join key.

## Why this module exists

Spike S5 measured the two Star Comprehensive wordings and found they use **incompatible
identifier schemes for identical content**:

| benefit | star-comprehensive-2021 | star-comprehensive-2025 |
|---|---|---|
| Bariatric surgery | ``Section 6`` | item ``15`` under ``Section II`` |
| Delivery and new born | ``Section 2`` | item ``14`` under ``Section II`` |
| Organ donor | ``Section 3`` | item ``9`` |
| Air ambulance | ``Section 1`` | item ``8`` |

There is no arithmetic relationship. So ``clause_id`` — which is correct and sufficient as
*within-document* provenance — **cannot express "the same clause in a different version"**,
which is the entire point of the supersession stratum.

What does survive across versions is the benefit's name. All 16 anchor phrases tested in
S5 appear in both documents. So the join key is a **controlled vocabulary slug** assigned
by a human at labelling time, not a string extracted from the PDF.

## Why a controlled vocabulary and not free-text matching

Fuzzy-matching heading text across versions would introduce a second, silent failure mode:
a near-miss would look like "this concept does not exist in the old version", which is
indistinguishable from a genuine supersession finding. A closed vocabulary validated on
write means a typo is a load error, not a wrong metric.

## Division of labour between the keys

* ``clause_id`` — within-document provenance. What ``SpanMatcher`` scores against.
* ``concept_id`` — cross-document identity. What supersession and cross-insurer
  comparison items join on.
* ``must_not_cite`` — for the supersession trap, **document-level is sufficient**
  (``"star-comprehensive-2021"``). It only has to name the wrong *document*; it never
  needs a matching clause, which is fortunate given the above.
"""

from __future__ import annotations

import re
from enum import StrEnum

CLAUSE_ID_PATTERN = re.compile(
    r"""^(?:
        excl\.\d{2}                   # IRDAI standardised exclusion code: excl.02
        |
        def\.[a-z][a-z0-9_]{1,38}     # a defined term: def.hospital
        |
        (?:[IVX]{1,4}|\d{1,2})        # section: roman (II) or numeric (6)
        (?:\.\d{1,3})*                # numeric sub-levels: .14, .2.1
        (?:\.[a-z]{1,3})?             # alpha sub-clause: .b, .iii
        (?:\.table)?                  # a table inside that clause
    )$""",
    re.VERBOSE,
)

# "Code Excl 02", "Excl 2", "excl.02" -> "excl.02"
_EXCL_CODE = re.compile(r"^(?:code[\s.]*)?excl[\s.]*(\d{1,2})$", re.IGNORECASE)

# "Hospital", "Hospital:", "def.hospital", "definition.hospital" -> "def.hospital"
#
# The Definitions section carries NO clause numbering - measured: page_clauses for p4 of
# star-comprehensive-2025 is empty. Without this form a definitional span could only be
# matched by page, and page numbers are not comparable across versions (the two Star
# wordings differ 2.7x in pagination). A defined term IS stable across versions and across
# insurers, so `def.<term>` does for definitions what `excl.NN` does for exclusions.
_DEF_TERM = re.compile(
    r"^(?:def(?:inition)?[\s.]+)?([A-Za-z][A-Za-z0-9 /&'-]{1,38})$", re.IGNORECASE
)


def validate_clause_id(value: str) -> str:
    """Canonical clause id, or raise.

    Accepted: ``II.15``, ``6``, ``4.2.b``, ``3.1.table``, ``II.14.A`` (case-folded), and
    ``excl.02`` (also written ``Code Excl 02`` / ``Excl 2``).
    Rejected: ``page 34``, ``II-15``, ``""``. (``Section 6`` is accepted and normalised.)

    The format is validated on write because the alternative is a label the
    ``SpanMatcher`` silently fails to match — an item that is present in the file, absent
    from every metric, and produces no error anywhere.

    **The ``excl.NN`` form is the strongest identifier in this corpus.** IRDAI mandates
    standardised exclusion wording with a code, so ``Code Excl 02`` (specified-disease
    waiting period) appears verbatim in *both* Star versions — 38 codes in the 2021
    wording, 35 in the 2025 one. Unlike a bare numeric clause id it is unambiguous, and
    unlike an insurer's own section numbering it is stable across versions AND across
    insurers. Prefer it wherever the clause is a standardised exclusion.
    """
    cleaned = value.strip().rstrip(".")

    # IRDAI exclusion codes are checked first: they are a closed, regulator-defined
    # namespace, so they need none of the section-prefix stripping below and normalising
    # to a zero-padded form makes "Excl 2" and "Code Excl 02" the same identifier.
    excl = _EXCL_CODE.match(cleaned)
    if excl:
        return f"excl.{int(excl.group(1)):02d}"

    # Defined terms. Checked before the dotted-numeric grammar because a term is words,
    # not digits, and would otherwise fall through to the generic rejection.
    if cleaned.lower().startswith(("def.", "def ", "definition.", "definition ")):
        term = _DEF_TERM.match(cleaned)
        if term:
            return f"def.{slug_term(term.group(1))}"
    # "Section II.15" / "Sec 6" -> "II.15" / "6"
    cleaned = re.sub(r"^(?:section|sec|clause|cl)[\s.]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(" ", "")
    # Alpha sub-clauses are lower-cased so II.14.A and II.14.a are one clause.
    parts = cleaned.split(".")
    parts = [p.lower() if p.isalpha() and not _is_roman(p) else p for p in parts]
    cleaned = ".".join(parts)

    if not CLAUSE_ID_PATTERN.match(cleaned):
        raise ValueError(
            f"clause_id {value!r} is not in canonical form. Expected a dotted identifier "
            "such as 'II.15', '6', '4.2.b', '3.1.table'. Write the clause the answer "
            "actually lives in, not a page number and not a section title."
        )
    return cleaned


def slug_term(term: str) -> str:
    """Normalise a defined term into a stable slug.

    "Hospital" -> "hospital", "Grace Period" -> "grace_period",
    "Disclosure to information norm" -> "disclosure_to_information_norm".
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", term.strip().lower()).strip("_")
    return cleaned


def _is_roman(token: str) -> bool:
    return bool(re.fullmatch(r"[IVX]{1,4}", token, re.IGNORECASE))


class Concept(StrEnum):
    """Cross-version, cross-insurer benefit identity.

    Extend this list when labelling requires it — adding a member is a deliberate,
    reviewable act. Values are namespaced so a report can group by prefix.
    """

    # --- benefits ---
    BENEFIT_BARIATRIC_SURGERY = "benefit.bariatric_surgery"
    BENEFIT_DELIVERY_NEWBORN = "benefit.delivery_and_new_born"
    BENEFIT_ORGAN_DONOR = "benefit.organ_donor"
    BENEFIT_AIR_AMBULANCE = "benefit.air_ambulance"
    BENEFIT_AYUSH = "benefit.ayush_treatment"
    BENEFIT_HEALTH_CHECKUP = "benefit.health_checkup"
    BENEFIT_HOSPITAL_CASH = "benefit.hospital_cash"
    BENEFIT_DOMICILIARY = "benefit.domiciliary_hospitalisation"
    BENEFIT_DAY_CARE = "benefit.day_care_procedures"
    BENEFIT_CUMULATIVE_BONUS = "benefit.cumulative_bonus"
    BENEFIT_MATERNITY = "benefit.maternity"
    BENEFIT_VACCINATION = "benefit.vaccination"
    BENEFIT_PRE_POST_HOSP = "benefit.pre_and_post_hospitalisation"

    # Added 2026-10-06 while labelling h-21 (clause_tension). The vocabulary had no member
    # for out-patient dental cover, so a span on the Star dental benefit clause could only
    # be labelled with a concept that was wrong about it. See CONCEPT_ANCHORS for the
    # measurement that chose its anchors.
    BENEFIT_DENTAL_OPD = "benefit.dental_opd"

    # --- limits and deductions (where the tables live) ---
    LIMIT_ROOM_RENT = "limit.room_rent"
    LIMIT_ICU = "limit.icu_charges"
    LIMIT_PROPORTIONATE_DEDUCTION = "limit.proportionate_deduction"
    LIMIT_CO_PAYMENT = "limit.co_payment"
    LIMIT_SUB_LIMIT_CATARACT = "limit.sub_limit_cataract"
    LIMIT_SUM_INSURED = "limit.sum_insured"
    LIMIT_DEDUCTIBLE = "limit.deductible"

    # --- waiting periods ---
    WAIT_INITIAL = "waiting_period.initial"
    WAIT_PRE_EXISTING = "waiting_period.pre_existing_disease"
    WAIT_SPECIFIED_DISEASE = "waiting_period.specified_disease"
    WAIT_MATERNITY = "waiting_period.maternity"

    # --- definitions ---
    DEF_HOSPITAL = "definition.hospital"
    DEF_HOSPITALISATION = "definition.hospitalisation"
    DEF_PRE_EXISTING_DISEASE = "definition.pre_existing_disease"
    DEF_GRACE_PERIOD = "definition.grace_period"

    # --- procedure / governance ---
    PROC_CLAIM_NOTIFICATION = "procedure.claim_notification"
    PROC_PORTABILITY = "procedure.portability"
    PROC_FREE_LOOK = "procedure.free_look_period"
    PROC_MORATORIUM = "procedure.moratorium"
    EXCLUSION_PERMANENT = "exclusion.permanent"

    # Added while labelling the `unanswerable` stratum. Both name a subject the corpus
    # does NOT settle, which is unusual for this vocabulary and is the point: an
    # unanswerable item still needs a concept_id, because `must_not_cite` requires one,
    # and the decoy list is what makes an abstention test a test rather than a formality.
    #
    # exclusion.geographic_scope - whether treatment outside India is covered at all. The
    #   Star wordings define zones A-E, but those are Indian co-payment zones; a retriever
    #   asked about the USA lands on them and they answer a different question.
    # pricing.premium - the rupee amount payable. Policy *wordings* do not contain premium
    #   tables (those live in the prospectus and the schedule), so no document in this
    #   corpus can answer it, while several define premium-adjacent terms.
    EXCLUSION_GEOGRAPHIC_SCOPE = "exclusion.geographic_scope"
    PRICING_PREMIUM = "pricing.premium"

    @property
    def namespace(self) -> str:
        return self.value.split(".", 1)[0]


CONCEPT_VALUES = frozenset(c.value for c in Concept)


def validate_concept_id(value: str) -> str:
    if value not in CONCEPT_VALUES:
        near = sorted(c for c in CONCEPT_VALUES if value.split(".")[-1][:6] in c)
        hint = f" Did you mean one of {near}?" if near else ""
        raise ValueError(
            f"concept_id {value!r} is not in the controlled vocabulary. Add a member to "
            f"arag.eval.concepts.Concept if the benefit is genuinely new.{hint}"
        )
    return value


# --- anchor phrases: the cross-version join MECHANISM ---------------------------------
#
# A concept_id is assigned by a human, but it is not arbitrary: each one is grounded in
# phrases that actually occur in the documents. These anchors are what make the join
# *checkable* rather than asserted — `arag-ingest clause-index` records which pages of
# which documents contain each anchor, and validation uses that to confirm a labelled
# span is plausible for the concept it claims.
#
# Anchors are matched case-insensitively against NFKC-normalised page text, which is why
# normalisation had to land first: "beneﬁt" would not match "benefit".
#
# Phrases must match how the DOCUMENTS word things, not how we do. Found the hard way:
# WAIT_INITIAL originally listed "30 days waiting", but star-comprehensive-2025 p32 says
# "30-day waiting period". The concept check then ERRORED on a correct label, because a
# missing anchor is indistinguishable from a wrong page. When a concept check rejects a
# span you believe is right, extend the anchors here first.
CONCEPT_ANCHORS: dict[Concept, tuple[str, ...]] = {
    Concept.BENEFIT_BARIATRIC_SURGERY: ("bariatric surgery",),
    Concept.BENEFIT_DELIVERY_NEWBORN: ("delivery and new born", "delivery & new born"),
    Concept.BENEFIT_ORGAN_DONOR: ("organ donor",),
    Concept.BENEFIT_AIR_AMBULANCE: ("air ambulance",),
    Concept.BENEFIT_AYUSH: ("ayush",),
    Concept.BENEFIT_HEALTH_CHECKUP: ("health check", "health check-up", "healthcheck"),
    Concept.BENEFIT_HOSPITAL_CASH: ("hospital cash",),
    Concept.BENEFIT_DOMICILIARY: ("domiciliary",),
    Concept.BENEFIT_DAY_CARE: ("day care", "day-care"),
    Concept.BENEFIT_CUMULATIVE_BONUS: ("cumulative bonus",),
    Concept.BENEFIT_MATERNITY: ("maternity",),
    Concept.BENEFIT_VACCINATION: (
        "vaccination",
        "vaccine",
        "immunisation",
        "immunization",
    ),
    # Measured across all six documents BEFORE these anchors were kept, because an anchor
    # set that matches everything confirms nothing. The union of the three reaches exactly
    # the dental footprint of each document - the DEFINITION, the BENEFIT clause and the
    # EXCLUSION - and nothing else:
    #
    #   star-comprehensive-2025   [4 def, 16 benefit cl.17, 35 excl.32]
    #   star-comprehensive-2021   [3 def, 5  benefit Sec 3, 11 excl.32]
    #   nivabupa-reassure2        [2 def, 18 exclusion]
    #   nivabupa-rise             [2 def, 16 exclusion]
    #   irdai-annexure-2024       []        irdai-master-circular-2024  []
    #
    # For contrast, a bare "dental" anchor reaches 11 pages of star-2025 and 8 of star-2021.
    # THAT would be a catch-all; this is a footprint.
    #
    # The namespace says `benefit.` while "dental treatment" also reaches the exclusion and
    # the definition. That is deliberate and already precedented - BENEFIT_MATERNITY anchors
    # on "maternity", which reaches WAIT_MATERNITY's clause too. The concept check asks "is
    # this page about this subject", not "is this clause of this kind".
    Concept.BENEFIT_DENTAL_OPD: ("out-patient dental", "dental treatment", "licensed dentist"),
    Concept.BENEFIT_PRE_POST_HOSP: (
        "pre-hospitalisation",
        "post-hospitalisation",
        "pre hospitalization",
    ),
    Concept.LIMIT_ROOM_RENT: (
        "room rent",
        "room/icu",
        "room category",
        "single private a/c room",
        "room type",
    ),
    Concept.LIMIT_ICU: ("icu charges", "intensive care"),
    Concept.LIMIT_PROPORTIONATE_DEDUCTION: ("proportionate deduction",),
    Concept.LIMIT_CO_PAYMENT: ("co-payment", "copayment", "co payment"),
    Concept.LIMIT_SUB_LIMIT_CATARACT: ("cataract",),
    Concept.LIMIT_SUM_INSURED: ("sum insured",),
    Concept.LIMIT_DEDUCTIBLE: ("deductible",),
    Concept.WAIT_INITIAL: (
        "initial waiting period",
        "30-day waiting",
        "30 day waiting",
        "30 days waiting",
        "first policy commencement date",
        "excl 03",
    ),
    Concept.WAIT_PRE_EXISTING: (
        "pre-existing disease",
        "pre existing disease",
        "pre- existing disease",
        "excl 01",
        # The conditional-override clause: an optional rider that reduces the PED waiting
        # period from 36 to 12 months. Anchored so a label on that clause resolves
        # without moving the span, which is the correct fix when a concept genuinely
        # appears in wording the vocabulary had not captured.
        "buy back",
        "buy-back",
        "buy back of pre-existing",
    ),
    Concept.WAIT_SPECIFIED_DISEASE: (
        "specified disease",
        "specific disease",
        "specified illness",
        "specific waiting period",
        "excl 02",
    ),
    Concept.WAIT_MATERNITY: ("maternity",),
    Concept.DEF_HOSPITAL: ("hospital means", "definition of hospital"),
    Concept.DEF_HOSPITALISATION: ("hospitalisation means", "hospitalization means"),
    Concept.DEF_PRE_EXISTING_DISEASE: ("pre-existing disease means", "pre existing disease means"),
    Concept.DEF_GRACE_PERIOD: ("grace period",),
    Concept.PROC_CLAIM_NOTIFICATION: ("notification of claim", "intimation", "notify"),
    Concept.PROC_PORTABILITY: ("portability",),
    Concept.PROC_FREE_LOOK: ("free look", "free-look"),
    Concept.PROC_MORATORIUM: ("moratorium", "moratorium period", "excl 04"),
    # "code excl" added 2026-10-06. The section HEADER ("Permanent Exclusions") sits on
    # star-comprehensive-2025 p30-31, and the numbered list runs on to p32-35 without
    # repeating the word - p35 says "exclusion" singular, which the plural anchor misses.
    # So three labels on genuine permanent exclusions (excl.06 p33, excl.08 p33, excl.32
    # p35) were rejected by the concept cross-check as "almost certainly on the wrong page"
    # when the page was right and the anchor was too narrow.
    #
    # Measured before widening: 6, 9, 7, 9, 1, 4 pages across the six documents.
    # Measured after:          10, 9, 7, 9, 1, 4 - exactly +[32, 33, 34, 35] in star-2025
    # and NO change anywhere else. A broadened anchor set that matches everything confirms
    # nothing, so the widening was measured before it was kept.
    Concept.EXCLUSION_PERMANENT: ("permanent exclusion", "exclusions", "code excl"),
}


def anchors_for(concept: Concept) -> tuple[str, ...]:
    """Phrases that ground a concept in the actual documents.

    Falls back to the slug's own words so a newly-added Concept without curated anchors
    still resolves to something, rather than silently having no anchor at all.
    """
    curated = CONCEPT_ANCHORS.get(concept)
    if curated:
        return curated
    return (concept.value.split(".", 1)[1].replace("_", " "),)
