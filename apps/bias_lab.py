import time
import asyncio
import html
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# QUT001 Zylometry Lab — v33 FINAL
#
# Same teaching mechanics as v6, with a more game-like UI:
#   Mission 1. Build a founding team -> subjective labels
#   Mission 2. Train an AI from those labels
#   Mission 3. Scale the company until old applicant pool ends
#   Mission 3b. Guild scandal occurs inside scaling -> historical bias
#   Mission 5. Label post-Guild applicants and update the AI
#   Mission 7. Interstate hiring -> sampling bias
#   Mission 8. Add interstate examples and retrain
#   Mission 9. Final company report
#
# Visual changes:
#   * persistent scoreboard
#   * clickable candidate cards (5 per page)
#   * abstract avatars only (no age/gender/ethnicity cues)
#   * in-page training/deployment animations
#   * applicant ticker and company-growth dashboard
#   * event banners and concept unlock cards
#   * larger typography / spacing / game-like panels
# ============================================================

FEATURES = ["Experience", "Qualification", "Work ethic", "Teamwork", "Guild accreditation"]
TRAINING_ANIMATION_SECONDS = 25.0
RETRAIN_ANIMATION_SECONDS = 2.5
DEPLOY_ANIMATION_SECONDS = 1.8
CARDS_PER_PAGE = 4
GROWTH_BATCH_SIZE = 50
STARTING_APPLICANTS = 200
STARTING_COMPANY_VALUE = 0.00  # company has not launched before the founding team is hired
FOUNDING_LAUNCH_THRESHOLD = 75.0
POST_NEWS_DIAGNOSIS_ROUNDS = 4
REPAIR_BATCH_SIZE = 50
REDEPLOY_ANIMATION_SECONDS = 4.0
INTERSTATE_ANIMATION_SECONDS = 4.5
INTERSTATE_POOL_SIZE = 100
INTERSTATE_SUPPORT_LAMBDA = 0.45

# A natural mix of common names students might encounter in a multicultural
# Australian workplace.  First names only, and deliberately not tied to any
# particular candidate profile or Guild status.
NAMES = [
    "Alex", "Priya", "Bailey", "Minh", "Casey",
    "Aisha", "Devon", "Luca", "Mei", "Finley",
    "Arjun", "Harper", "Hana", "Jules", "Kai",
    "Leila", "Morgan", "Ravi", "Sofia", "Zion",
]

CURRENT_NAMES = [
    "Avery", "Noor", "Blake", "Mateo", "Cameron",
    "Yuki", "Drew", "Amira", "Elliot", "Nadia",
    "Jamie", "Omar", "Kendall", "Mika", "Nico",
    "Sana", "Robin", "Sam", "Toby", "Val",
]

MARKET_NAMES = [
    "Amara", "Chen", "Diego", "Elena", "Farah", "Hugo", "Imani", "Jin", "Kavya", "Leo",
    "Mariam", "Nikhil", "Owen", "Rina", "Sora", "Tariq", "Uma", "Vikram", "Yara", "Zane",
    "Anika", "Ben", "Cleo", "Dara", "Emi", "Faisal", "Grace", "Hassan", "Inez", "Jonah",
    "Keiko", "Luis", "Mina", "Nora", "Pavel", "Rohan", "Sara", "Theo", "Wei", "Zara",
    "Adil", "Bianca", "Dae", "Eva", "Hamza", "Lina", "Milan", "Nia", "Rafael", "Tala",
]

INTERSTATE_NAMES = [
    "Aarav", "Bella", "Chloe", "Dinesh", "Elif", "Gabriel", "Hyejin", "Isaac", "Jia", "Kiran",
    "Lara", "Malik", "Naomi", "Oscar", "Pari", "Quentin", "Rosa", "Sanjay", "Tessa", "Yusuf",
]

# Founding applicants. Guild accreditation is added separately below.
LEVEL1_VALUES = np.array([
    [5, 1, 2, 2], [2, 2, 2, 2], [4, 1, 4, 3], [1, 2, 4, 5],
    [5, 1, 2, 2], [3, 2, 4, 4], [4, 1, 2, 5], [2, 1, 2, 2],
    [3, 2, 2, 2], [5, 2, 2, 2], [1, 1, 2, 2], [4, 2, 3, 4],
    [2, 2, 2, 2], [3, 1, 2, 2], [5, 1, 4, 3], [1, 2, 3, 5],
    [4, 1, 4, 4], [2, 1, 2, 2], [3, 2, 4, 5], [5, 2, 2, 4],
], dtype=int)

# Visible historical credential. In the old Zylometry market, accreditation was
# strongly correlated with the candidates who looked attractive to hiring managers.
# It is an AI input, but contributes nothing to the true workplace score.
LEVEL1_GUILD = np.array([
    0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
    0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
], dtype=int)

# ============================================================
# Synthetic applicant generation
# ============================================================

def middle_rating(rng, size):
    return rng.choice(
        np.array([1, 2, 3, 4, 5]), size=size,
        p=np.array([0.03, 0.20, 0.54, 0.20, 0.03]),
    )


def profile_is_reasonable(row):
    row = np.asarray(row)
    if np.sum(row == 5) > 1:
        return False
    if np.sum(row >= 4) > 2:
        return False
    if row.sum() > 14:
        return False
    return True


def generate_traditional(rng, n):
    rows = []
    while len(rows) < n:
        row = np.array([
            rng.choice([2, 3, 4, 5], p=[0.12, 0.48, 0.32, 0.08]),
            rng.choice([1, 2], p=[0.70, 0.30]),
            *middle_rating(rng, 2),
        ])
        if profile_is_reasonable(row):
            rows.append(row)
    return np.asarray(rows, dtype=int)


def generate_graduates(rng, n):
    rows = []
    while len(rows) < n:
        row = np.array([
            rng.choice([1, 2], p=[0.82, 0.18]),
            rng.choice([4, 5], p=[0.86, 0.14]),
            rng.choice([2, 3, 4], p=[0.20, 0.60, 0.20]),
            rng.choice([2, 3, 4], p=[0.20, 0.60, 0.20]),
        ])
        if profile_is_reasonable(row):
            rows.append(row)
    return np.asarray(rows, dtype=int)


def generate_interstate_candidates(seed, n):
    """Highly qualified applicants with limited on-the-job experience.

    Most have 1–2 stars of Experience, with only a very small chance of 3.
    Qualification is always 4–5 stars. Work ethic and Teamwork overlap the
    local population so this group is different mainly in the *shape* of its
    preparation, not simply better overall.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append([
            int(rng.choice([1, 2, 3], p=[0.74, 0.25, 0.01])),
            int(rng.choice([4, 5], p=[0.72, 0.28])),
            int(rng.choice([2, 3, 4, 5], p=[0.15, 0.45, 0.35, 0.05])),
            int(rng.choice([2, 3, 4, 5], p=[0.08, 0.32, 0.42, 0.18])),
        ])
    return np.asarray(rows, dtype=int)


def mixed_population(seed, n, graduate_fraction=0.5):
    rng = np.random.default_rng(seed)
    n_grad = int(round(n * graduate_fraction))
    n_trad = n - n_grad
    X = np.vstack([generate_traditional(rng, n_trad), generate_graduates(rng, n_grad)])
    pathway = np.array(["Traditional"] * n_trad + ["New graduate"] * n_grad)
    order = rng.permutation(len(X))
    return X[order], pathway[order]


def augment_features(X, guild_flags):
    X = np.asarray(X, dtype=int)
    guild_flags = np.asarray(guild_flags, dtype=int).reshape(-1, 1)
    return np.hstack([X, guild_flags])


def founding_quality_target(state, floor=3.55):
    s = state or initial_state()
    selected = np.asarray(s.get("initial_hires") or s.get("initial_selected", [False] * 20), dtype=bool)
    if selected.sum() == 5:
        return max(float(agreed_score(LEVEL1_VALUES[selected]).mean()), floor)
    return floor


def generate_quality_matched_applicants(rng, n, target_score, band=0.42):
    """Generate applicants whose genuine profile quality is close to a target.

    This is used so later hiring rounds are not full of obviously weak people:
    applicants are roughly as strong on the four real workplace attributes as
    the founding hires.  Guild membership is added separately.
    """
    rows = []
    attempts = 0
    while len(rows) < n and attempts < 200000:
        attempts += 1
        row = generate_traditional(rng, 1)[0]
        if abs(float(agreed_score(row.reshape(1, -1))[0]) - float(target_score)) <= band:
            rows.append(row)
    if len(rows) < n:
        # Very defensive fallback for unusual low/high founding selections.
        while len(rows) < n:
            rows.append(generate_traditional(rng, 1)[0])
    return np.asarray(rows, dtype=int)


def old_market_batch(seed, n, target_score):
    """Pre-scandal market: good applicants, most still carrying Guild badges."""
    rng = np.random.default_rng(seed)
    X = generate_quality_matched_applicants(rng, n, target_score)
    guild = (rng.random(n) < 0.72).astype(int)
    return X, guild


def post_guild_batch(seed, n, target_score):
    """Post-scandal market: comparably good applicants, nobody buys Guild status."""
    rng = np.random.default_rng(seed)
    X = generate_quality_matched_applicants(rng, n, max(target_score, 3.65), band=0.38)
    guild = np.zeros(n, dtype=int)
    return X, guild


def founder_like_market_batch(seed, n, state, guild_probability=0.86):
    """Create a market whose genuine profiles closely match the founding team.

    The four workplace attributes are generated independently of Guild status, and
    the *batch averages* are kept very close to the founders. This prevents normal
    scaling from quietly introducing a lower-Teamwork or otherwise lower-quality
    workforce before the historical-bias event.
    """
    s = state or initial_state()
    selected = np.asarray(s.get("initial_hires") or s.get("initial_selected", [False] * 20), dtype=bool)
    founders = LEVEL1_VALUES[selected] if selected.sum() == 5 else LEVEL1_VALUES[np.argsort(agreed_score(LEVEL1_VALUES))[-5:]]
    founder_means = founders.mean(axis=0)
    rng = np.random.default_rng(seed)

    # Start from repeated founder profiles, which gives an exactly matched mean
    # whenever n is a multiple of five (our hiring rounds use 50).
    reps = int(np.ceil(n / len(founders)))
    X = np.tile(founders, (reps, 1))[:n].copy()
    rng.shuffle(X, axis=0)

    # Add paired +/-1 perturbations. Each pair preserves the column mean while
    # making the applicant pool look varied rather than cloned.
    for j in range(4):
        order = rng.permutation(n)
        for a, b in zip(order[0::2], order[1::2]):
            if rng.random() < 0.30:
                if X[a, j] < 5 and X[b, j] > 1:
                    X[a, j] += 1
                    X[b, j] -= 1
                elif X[a, j] > 1 and X[b, j] < 5:
                    X[a, j] -= 1
                    X[b, j] += 1

    # Defensive correction for non-multiples of five / clipped perturbations.
    # We only accept very small deviations from the founding team's genuine profile.
    for _ in range(80):
        diff = founder_means - X.mean(axis=0)
        if np.max(np.abs(diff)) <= 0.20:
            break
        j = int(np.argmax(np.abs(diff)))
        if diff[j] > 0:
            candidates = np.flatnonzero(X[:, j] < 5)
            if len(candidates):
                X[rng.choice(candidates), j] += 1
        else:
            candidates = np.flatnonzero(X[:, j] > 1)
            if len(candidates):
                X[rng.choice(candidates), j] -= 1

    guild = (rng.random(n) < float(guild_probability)).astype(int)
    return X.astype(int), guild


def find_stable_pre_news_batch(model, threshold, state, batch_n, round_idx):
    """Build a normal-growth batch whose *high-scoring hires* remain founder-like.

    A large reservoir is generated from the founding team's genuine profile. The
    fraction that clears the student's learned hiring threshold determines how many
    hires appear in the 50-person round. We then choose those eligible applicants so
    their four genuine attribute averages stay close to the founding team. This keeps
    normal scaling from introducing a systematic Culture/Efficiency drift while still
    letting the AI's score decide who is eligible to be hired.
    """
    s = state or initial_state()
    selected = np.asarray(s.get("initial_hires") or [False] * 20, dtype=bool)
    founders = LEVEL1_VALUES[selected]
    founder_means = founders.mean(axis=0)
    rng = np.random.default_rng(61000 + round_idx)

    reservoir_n = 600
    X_all, guild_all = founder_like_market_batch(
        61000 + round_idx * 17, reservoir_n, s, guild_probability=0.86
    )
    probs_all = model.predict_proba(augment_features(X_all, guild_all))[:, 1]
    eligible = np.flatnonzero(probs_all >= threshold)
    ineligible = np.flatnonzero(probs_all < threshold)

    if len(eligible) == 0:
        # Extremely unlikely in the pre-news market, but keep the function safe.
        chosen = rng.choice(np.arange(reservoir_n), size=batch_n, replace=False)
        X, guild, probs = X_all[chosen], guild_all[chosen], probs_all[chosen]
        return X, guild, probs

    natural_rate = len(eligible) / reservoir_n
    target_hires = int(np.clip(round(batch_n * natural_rate), 3, min(22, batch_n)))
    target_hires = min(target_hires, len(eligible))
    target_non = batch_n - target_hires
    if len(ineligible) < target_non:
        target_non = len(ineligible)
        target_hires = batch_n - target_non

    # Greedily choose eligible applicants so the cumulative hire profile remains
    # close to the founding team on each genuine attribute.
    remaining = eligible.tolist()
    chosen_hires = []
    running = np.zeros(4, dtype=float)
    for k in range(target_hires):
        best_pos = None
        best_obj = None
        # Randomise tie order so successive rounds do not look cloned.
        candidates = rng.permutation(remaining)
        for idx in candidates:
            new_mean = (running + X_all[idx]) / (k + 1)
            drift = np.abs(new_mean - founder_means)
            obj = float(drift.max() + 0.25 * drift.mean())
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_pos = int(idx)
        chosen_hires.append(best_pos)
        running += X_all[best_pos]
        remaining.remove(best_pos)

    chosen_non = []
    if target_non > 0:
        chosen_non = rng.choice(ineligible, size=target_non, replace=False).tolist()

    chosen = np.asarray(chosen_hires + chosen_non, dtype=int)
    rng.shuffle(chosen)
    X, guild, probs = X_all[chosen], guild_all[chosen], probs_all[chosen]
    return X, guild, probs

def founding_hire_threshold(model, state):
    """Hiring cutoff anchored to the student's own founding Hire labels.

    The cutoff sits just below the lowest-scoring example the student explicitly
    labelled Hire. This keeps deployment selective, but avoids the confusing case
    where the AI immediately refuses to hire someone from its own positive training
    examples simply because an upper-quantile threshold was too harsh.
    """
    s = state or initial_state()
    y = np.asarray(s.get("initial_hires") or [False] * len(LEVEL1_VALUES), dtype=bool)
    probs = model.predict_proba(augment_features(LEVEL1_VALUES, LEVEL1_GUILD))[:, 1]
    positive = probs[y]
    if len(positive) == 0:
        return 0.50
    # Use a middle/high part of the student's positive-score range. This is
    # substantially less strict than the old 90th-percentile cutoff, so normal
    # scaling produces a healthy number of hires while the later loss of the
    # historically predictive Guild signal can still cause a dramatic collapse.
    return float(max(0.40, np.quantile(positive, 0.65)))


# ============================================================
# Hidden workplace mechanics — same recalibration as v6
# ============================================================

def workplace_diagnostics(X):
    """Return the exact hidden scoring components for temporary diagnosis UI."""
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return None

    exp, qual, work, teamwork = [X[:, i] for i in range(4)]
    preparation = np.maximum(exp, qual)

    prep_mean = float(preparation.mean())
    work_mean = float(work.mean())
    team_mean = float(teamwork.mean())
    exp_mean = float(exp.mean())
    qual_mean = float(qual.mean())
    balance = max(0.0, 1.0 - abs(exp_mean - qual_mean) / 4.0)

    def sat(value, low, high):
        return float(np.clip((value - low) / (high - low), 0, 1))

    eff_parts = {
        "Base": 18.0,
        "Preparation": 28.0 * sat(prep_mean, 2.5, 3.8),
        "Work ethic": 32.0 * sat(work_mean, 2.5, 3.7),
        "Teamwork": 18.0 * sat(team_mean, 2.5, 3.7),
    }
    culture_parts = {
        "Base": 20.0,
        "Teamwork": 50.0 * sat(team_mean, 2.4, 3.8),
        "Work ethic": 20.0 * sat(work_mean, 2.4, 3.8),
    }

    culture_raw = sum(culture_parts.values())
    culture = float(np.clip(culture_raw, 0, 90))
    culture_penalty = max(0.0, 55.0 - culture_raw) * 0.80
    efficiency_raw = sum(eff_parts.values()) - culture_penalty
    efficiency = float(np.clip(efficiency_raw, 0, 90))

    return {
        "efficiency": efficiency,
        "culture": culture,
        "eff_parts": eff_parts,
        "culture_parts": culture_parts,
        "culture_penalty": culture_penalty,
        "prep_mean": prep_mean,
        "work_mean": work_mean,
        "team_mean": team_mean,
        "exp_mean": exp_mean,
        "qual_mean": qual_mean,
        "balance": balance,
    }


def workplace_metrics(X):
    """Hidden game mechanics for workplace outcomes."""
    d = workplace_diagnostics(X)
    if d is None:
        return 0.0, 0.0
    return d["efficiency"], d["culture"]

# ============================================================
# ML helpers
# ============================================================

def train_student_model(X, y):
    """Small categorical classifier used for the student-labelled hiring AI.

    The teaching goal is to make correlations in a tiny labelled dataset visible.
    All four ratings are discrete 1-5 values; Guild accreditation is binary.
    """
    X = np.asarray(X, dtype=int)
    min_categories = [6] * min(4, X.shape[1])
    if X.shape[1] == 5:
        min_categories.append(2)
    model = CategoricalNB(alpha=1.0, min_categories=min_categories)
    model.fit(X, y)
    return model


def train_sensitive_model(X, y):
    k = min(7, max(1, len(X) - 1))
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X, y)
    return model


def shortlist(model, X, n_hires):
    scores = model.predict_proba(X)[:, 1]
    n_hires = int(np.clip(n_hires, 1, len(X)))
    keep = np.zeros(len(X), dtype=bool)
    keep[np.argsort(scores)[-n_hires:]] = True
    return keep


def confidence_hire_mask(model, X, threshold=0.50, max_hires=None):
    """Hire only applicants the model rates above a confidence threshold.

    Unlike shortlist(), this does not force a fixed number of hires.  This is
    used after the profession changes so students can see an old model simply
    fail to recommend many profiles that lack a historically learned shortcut feature.
    """
    scores = model.predict_proba(X)[:, 1]
    eligible = np.flatnonzero(scores >= threshold)
    if max_hires is not None and len(eligible) > max_hires:
        eligible = eligible[np.argsort(scores[eligible])[-int(max_hires):]]
    keep = np.zeros(len(X), dtype=bool)
    keep[eligible] = True
    return keep, scores


def deployment_value_gain(efficiency, culture, n_hires):
    """Fictional company-value growth used by the game.

    Each additional employee adds meaningful value, with stronger Efficiency and
    Culture increasing the value created per hire. This deliberately scales much
    faster than earlier versions so company growth feels consequential.
    """
    if n_hires <= 0:
        return 0.0
    performance = float(np.clip((float(efficiency) + float(culture)) / 200.0, 0, 1))
    value_per_hire_m = 0.055 + 0.060 * performance
    return float(n_hires) * value_per_hire_m


def estimated_candidate_value_percent(row):
    """Fictional game estimate used only to make rejected standouts tangible."""
    score = float(agreed_score(np.asarray(row).reshape(1, -1))[0])
    norm = np.clip((score - 2.50) / (4.00 - 2.50), 0, 1)
    return 2.0 + 6.0 * norm


def agreed_score(X):
    X = np.asarray(X)
    exp, qual, work, teamwork = [X[:, i] for i in range(4)]
    preparation = np.maximum(exp, qual)
    return 0.25 * preparation + 0.30 * work + 0.45 * teamwork


def agreed_label(X):
    return (agreed_score(X) >= 3.55).astype(int)


# Mission 5 uses a fixed set of strong non-Guild applicants for manual labelling.
# It is initialised here because post_guild_batch uses agreed_score().
CURRENT_BATCH, CURRENT_GUILD = post_guild_batch(seed=4242, n=20, target_score=3.80)
CURRENT_PATHWAYS = np.array([""] * 20)

INTERSTATE_POOL = generate_interstate_candidates(seed=61001, n=INTERSTATE_POOL_SIZE)
INTERSTATE_LABEL_BATCH = generate_interstate_candidates(seed=61002, n=20)
INTERSTATE_TEST_POOL = generate_interstate_candidates(seed=61003, n=INTERSTATE_POOL_SIZE)

# ============================================================
# State
# ============================================================

def initial_state():
    return {
        "initial_selected": [False] * 20,
        "initial_page": 0,
        "initial_hires": None,
        "ai_trained": False,
        "current_selected": [False] * 20,
        "current_page": 0,
        "current_hires": None,
        "sampling_mix": 50,
        "growth_round": 0,
        "applicants_remaining": STARTING_APPLICANTS,
        "display_applicants": 20,
        "company_value": STARTING_COMPANY_VALUE,
        "employees": 0,
        "growth_workforce": None,
        "last_efficiency": None,
        "last_culture": None,
        "label_bias_unlocked": False,
        "historical_bias_unlocked": False,
        "sampling_bias_unlocked": False,
        "applicants_screened": 0,
        "historical_deployed": False,
        "updated_deployed": False,
        "guild_scandal": False,
        "last_pre_scandal_hired": None,
        "pre_scandal_total_hired": 0,
        "post_scandal_round_done": False,
        "diagnosis_rejected": [],
        "diagnosis_names": [],
        "diagnosis_scores": [],
        "guild_crisis_round": 0,
        "guild_crisis_hired": 0,
        "guild_crisis_screened": 0,
        "scaling_start_value": None,
        "scaling_start_efficiency": None,
        "scaling_start_culture": None,
        "scaling_start_employees": None,
        "fix_strategy": None,
        "repair_trained": False,
        "repair_active": False,
        "repair_hired_total": 0,
        "repair_screened_total": 0,
        "repair_start_employees": None,
        "repair_start_value": None,
        "last_round_hires": [],
        "last_round_applicants": [],
        "last_round_hired_count": 0,
        "interstate_stage": "waiting",
        "interstate_selected": [False] * 20,
        "interstate_selection_order": [],
        "interstate_hired": 0,
        "interstate_screened": 0,
        "interstate_round": 0,
        "interstate_rejected": [],
        "interstate_start_value": None,
        "interstate_fix_trained": False,
        "interstate_fix_hired": 0,
        "interstate_fix_rate": 0.0,
        "interstate_initial_hire_indices": [],
        "interstate_fix_screened": 0,
        "interstate_fix_start_value": None,
        "interstate_fix_round": 0,
        "final_local_hires": 0,
        "final_interstate_hires": 0,
        "final_redeploy_gain": 0.0,
    }

# ============================================================
# Visual / HTML helpers
# ============================================================

def fmt_metric(v):
    return "—" if v is None else f"{float(v):.0f}"


def score_tone(v):
    if v is None:
        return "neutral"
    if v >= 80:
        return "good"
    if v >= 65:
        return "fair"
    if v >= 50:
        return "mid"
    if v >= 25:
        return "low"
    return "bad"


def scoreboard_html(state):
    s = state or initial_state()
    value = float(s.get("company_value", STARTING_COMPANY_VALUE))
    eff = s.get("last_efficiency")
    cul = s.get("last_culture")
    applicants = int(s.get("display_applicants", s.get("applicants_remaining", STARTING_APPLICANTS)))
    employees = int(s.get("employees", 0))
    eff_pct = 0 if eff is None else int(np.clip(eff, 0, 100))
    cul_pct = 0 if cul is None else int(np.clip(cul, 0, 100))
    app_pct = int(np.clip(100 * applicants / max(1, STARTING_APPLICANTS), 0, 100))

    return f"""
    <div class="scoreboard-shell">
      <div class="score-tile value-tile">
        <div class="score-icon">{icon_svg('value')}</div>
        <div class="score-copy"><div class="score-label">COMPANY VALUE</div><div class="score-number">${value:.2f}M</div>
        <div class="score-sub">{employees} employees</div></div>
      </div>
      <div class="score-tile {score_tone(eff)} efficiency-tile">
        <div class="score-icon">{icon_svg('efficiency')}</div><div class="score-copy"><div class="score-label">EFFICIENCY</div>
        <div class="score-number">{fmt_metric(eff)}<span class="score-denom"> /100</span></div>
        <div class="mini-track"><div class="mini-fill" style="width:{eff_pct}%"></div></div></div>
      </div>
      <div class="score-tile {score_tone(cul)} culture-tile">
        <div class="score-icon">{icon_svg('culture')}</div><div class="score-copy"><div class="score-label">CULTURE</div>
        <div class="score-number">{fmt_metric(cul)}<span class="score-denom"> /100</span></div>
        <div class="mini-track"><div class="mini-fill" style="width:{cul_pct}%"></div></div></div>
      </div>
      <div class="score-tile applicant-tile">
        <div class="score-icon">{icon_svg('applicants')}</div><div class="score-copy"><div class="score-label">APPLICANTS</div>
        <div class="score-number">{applicants}</div>
        <div class="mini-track"><div class="mini-fill" style="width:{app_pct}%"></div></div></div>
      </div>
    </div>
    """


def inline_art(kind="team"):
    # Decorative, abstract SVG only; no portraits or demographic cues.
    if kind == "team":
        return """<svg viewBox='0 0 240 110' class='banner-art' aria-hidden='true'>
        <circle cx='55' cy='45' r='20' fill='#6d5dfc'/><circle cx='105' cy='55' r='25' fill='#21b8a6'/>
        <circle cx='160' cy='43' r='18' fill='#ffad42'/><rect x='35' y='70' width='48' height='22' rx='11' fill='#6d5dfc'/>
        <rect x='78' y='82' width='58' height='20' rx='10' fill='#21b8a6'/><rect x='142' y='67' width='42' height='23' rx='11' fill='#ffad42'/>
        <path d='M188 84 L216 36 L229 84 Z' fill='#d8d4ff'/><circle cx='214' cy='59' r='6' fill='#6d5dfc'/></svg>"""
    if kind == "ai":
        return """<svg viewBox='0 0 240 110' class='banner-art' aria-hidden='true'>
        <rect x='78' y='23' width='84' height='66' rx='20' fill='#243b73'/><circle cx='105' cy='55' r='7' fill='#62d9ff'/>
        <circle cx='136' cy='55' r='7' fill='#62d9ff'/><path d='M103 73 Q120 85 138 72' stroke='#62d9ff' stroke-width='6' fill='none'/>
        <path d='M120 23 V10' stroke='#6d5dfc' stroke-width='6'/><circle cx='120' cy='9' r='7' fill='#ffad42'/>
        <path d='M50 79 H77 M163 79 H191' stroke='#21b8a6' stroke-width='8' stroke-linecap='round'/></svg>"""
    if kind == "growth":
        return """<svg viewBox='0 0 240 110' class='banner-art' aria-hidden='true'>
        <rect x='32' y='62' width='32' height='35' rx='4' fill='#a7d9ff'/><rect x='73' y='47' width='32' height='50' rx='4' fill='#79c5ff'/>
        <rect x='114' y='31' width='32' height='66' rx='4' fill='#499df3'/><rect x='155' y='17' width='32' height='80' rx='4' fill='#2b76d0'/>
        <path d='M42 51 C85 38 115 40 171 9' stroke='#21b8a6' stroke-width='8' fill='none' stroke-linecap='round'/>
        <path d='M171 9 L157 10 M171 9 L166 23' stroke='#21b8a6' stroke-width='8' stroke-linecap='round'/></svg>"""
    if kind == "degree":
        return """<svg viewBox='0 0 240 110' class='banner-art' aria-hidden='true'>
        <path d='M42 43 L120 12 L198 43 L120 74 Z' fill='#6d5dfc'/><path d='M70 55 V80 Q120 103 170 80 V55' fill='#d9d4ff'/>
        <path d='M198 43 V79' stroke='#ffad42' stroke-width='6'/><circle cx='198' cy='85' r='8' fill='#ffad42'/></svg>"""
    if kind == "guild":
        return """<svg viewBox='0 0 240 110' class='banner-art' aria-hidden='true'>
        <path d='M120 10 L135 40 L169 45 L144 69 L150 102 L120 86 L90 102 L96 69 L71 45 L105 40 Z' fill='#ffd36a'/>
        <path d='M121 27 L111 52 L126 61 L116 83' stroke='#9f2335' stroke-width='7' fill='none' stroke-linecap='round' stroke-linejoin='round'/>
        <circle cx='187' cy='31' r='18' fill='#9f2335'/><path d='M179 23 L195 39 M195 23 L179 39' stroke='white' stroke-width='6' stroke-linecap='round'/></svg>"""
    return ""



def icon_svg(name, cls="", color=None):
    icons = {
        "hire": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><circle cx='18' cy='16' r='7' fill='none' stroke='currentColor' stroke-width='3'/><path d='M6 38c1-7 6-11 12-11s11 4 12 11' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/><path d='M35 16v12M29 22h12' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/></svg>",
        "train": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><rect x='10' y='12' width='28' height='22' rx='6' fill='none' stroke='currentColor' stroke-width='3'/><circle cx='20' cy='23' r='2.5' fill='currentColor'/><circle cx='28' cy='23' r='2.5' fill='currentColor'/><path d='M18 30c2 2 10 2 12 0M24 12V7' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/><circle cx='24' cy='5' r='2' fill='currentColor'/></svg>",
        "grow": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><path d='M10 33l9-9 7 6 12-14' fill='none' stroke='currentColor' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/><path d='M30 16h8v8' fill='none' stroke='currentColor' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/><path d='M8 39h32' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/></svg>",
        "goal": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><circle cx='24' cy='24' r='14' fill='none' stroke='currentColor' stroke-width='3'/><circle cx='24' cy='24' r='8' fill='none' stroke='currentColor' stroke-width='3'/><circle cx='24' cy='24' r='2.8' fill='currentColor'/><path d='M34 14l6-6' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/><path d='M34 8h6v6' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>",
        "value": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><path d='M10 32l9-9 7 5 12-14' fill='none' stroke='currentColor' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/><path d='M30 14h8v8' fill='none' stroke='currentColor' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/></svg>",
        "efficiency": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><circle cx='24' cy='24' r='8' fill='none' stroke='currentColor' stroke-width='3'/><path d='M24 8v4M24 36v4M8 24h4M36 24h4M13 13l3 3M32 32l3 3M35 13l-3 3M16 32l-3 3' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/></svg>",
        "culture": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><path d='M24 39S9 31 9 20c0-6 4-10 9-10 3 0 5 1 6 4 2-3 4-4 7-4 5 0 9 4 9 10 0 11-16 19-16 19Z' fill='none' stroke='currentColor' stroke-width='3' stroke-linejoin='round'/><path d='M17 23c2 4 4 6 7 6s6-2 8-6' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/></svg>",
        "applicants": "<svg viewBox='0 0 48 48' class='svg-icon {cls}' aria-hidden='true'><rect x='9' y='7' width='30' height='34' rx='5' fill='none' stroke='currentColor' stroke-width='3'/><circle cx='20' cy='20' r='5' fill='none' stroke='currentColor' stroke-width='3'/><path d='M13 32c1-5 4-7 7-7s6 2 7 7M30 17h5M30 23h5M30 29h5' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round'/></svg>",
    }
    svg = icons[name].format(cls=cls)
    if color:
        svg = svg.replace("currentColor", color)
    return svg


def briefing_score_preview():
    items = [
        ("value", "COMPANY VALUE", "Build this as quickly as possible.", "Company value rises when your workforce performs well."),
        ("efficiency", "EFFICIENCY", "How effectively your team performs.", "Your hiring choices can push this up or down."),
        ("culture", "CULTURE", "How well your team works together.", "A strong culture also helps build company value."),
        ("applicants", "APPLICANTS", "How much talent remains in the market.", "As you deploy the AI, this pool counts down."),
    ]
    cards = []
    for name, title, short, detail in items:
        cards.append(f"""<div class='score-preview-tile'><div class='score-preview-icon'>{icon_svg(name, color='#ffffff')}</div><div><div class='score-preview-label'>{title}</div><div class='score-preview-short'>{short}</div><div class='score-preview-detail'>{detail}</div></div></div>""")
    return "<div class='score-preview-grid'>" + "".join(cards) + "</div>"

def mission_banner(kicker, title, subtitle, kind="team", theme="blue"):
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    return f"""
    <section class="mission-banner {theme}">
      <div class="mission-banner-copy"><div class="mission-kicker">{html.escape(kicker)}</div>
      <h1>{html.escape(title)}</h1>{subtitle_html}</div>{inline_art(kind)}
    </section>
    """


def rating_html(value):
    value = int(value)
    parts = []
    for i in range(5):
        cls = "dots-on" if i < value else "dots-off"
        dot = "●" if i < value else "○"
        parts.append(f"<span class='{cls}'>{dot}</span>")
    return f"<span class='rating-dots'>{''.join(parts)}</span>"


def make_card_data(names, values, selected, page, pathways=None, guild_flags=None):
    start = page * CARDS_PER_PAGE
    end = min(start + CARDS_PER_PAGE, len(names))
    cards = []
    for idx in range(start, end):
        name = names[idx]
        row = values[idx]
        is_sel = bool(selected[idx])
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        path = "" if pathways is None else str(pathways[idx])
        guild = False if guild_flags is None else bool(guild_flags[idx])
        cards.append({
            "index": idx,
            "name": name,
            "initials": initials,
            "avatar_class": f"avatar-{idx % 8}",
            "pathway": path,
            "pathway_display": "none" if not path else "inline-flex",
            "guild_text": "★ ZYLOMETRY GUILD ACCREDITED" if guild else "",
            "guild_display": "inline-flex" if guild else "none",
            "card_class": "candidate-card selected" if is_sel else "candidate-card",
            "button_class": "hire-pill selected" if is_sel else "hire-pill",
            "button_text": "✓ SELECTED" if is_sel else "+ HIRE",
            "exp": rating_html(row[0]),
            "qual": rating_html(row[1]),
            "work": rating_html(row[2]),
            "team": rating_html(row[3]),
        })
    return cards


def make_all_card_data(names, values, selected, pathways=None, guild_flags=None):
    """Return every applicant card at once for the founding-team mission."""
    cards = []
    for idx in range(len(names)):
        name = names[idx]
        row = values[idx]
        is_sel = bool(selected[idx])
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        path = "" if pathways is None else str(pathways[idx])
        guild = False if guild_flags is None else bool(guild_flags[idx])
        cards.append({
            "index": idx,
            "name": name,
            "initials": initials,
            "avatar_class": f"avatar-{idx % 8}",
            "pathway": path,
            "pathway_display": "none" if not path else "inline-flex",
            "guild_text": "★ ZYLOMETRY GUILD ACCREDITED" if guild else "",
            "guild_display": "inline-flex" if guild else "none",
            "card_class": "candidate-card selected" if is_sel else "candidate-card",
            "button_class": "hire-pill selected" if is_sel else "hire-pill",
            "button_text": "✓ SELECTED" if is_sel else "+ HIRE",
            "exp": rating_html(row[0]),
            "qual": rating_html(row[1]),
            "work": rating_html(row[2]),
            "team": rating_html(row[3]),
        })
    return cards


CANDIDATE_TEMPLATE = r"""
<div class="candidate-grid">
{{#each value}}
  <div class="{{card_class}}" data-index="{{index}}" role="button" tabindex="0" aria-label="Toggle {{name}}">
    <div class="candidate-head">
      <div class="abstract-avatar {{avatar_class}}">{{initials}}</div>
      <div class="candidate-name-block"><div class="candidate-name">{{name}}</div>
      <span class="guild-badge" style="display:{{guild_display}}">{{guild_text}}</span>
      <span class="pathway-badge" style="display:{{pathway_display}}">{{pathway}}</span></div>
    </div>
    <div class="trait-row"><span>🧰 Experience</span><span class="rating">{{{exp}}}</span></div>
    <div class="trait-row"><span>🎓 Qualification</span><span class="rating">{{{qual}}}</span></div>
    <div class="trait-row"><span>💪 Work ethic</span><span class="rating">{{{work}}}</span></div>
    <div class="trait-row"><span>🤝 Teamwork</span><span class="rating">{{{team}}}</span></div>
    <div class="{{button_class}}">{{button_text}}</div>
  </div>
{{/each}}
</div>
"""

FOUNDING_CANDIDATE_TEMPLATE = CANDIDATE_TEMPLATE.replace(
    '<div class="candidate-grid">',
    '<div class="candidate-grid founding-grid">',
    1,
)


CANDIDATE_JS = r"""
const activate = (card) => trigger('click', {index: Number(card.dataset.index)});
element.querySelectorAll('.candidate-card').forEach(card => {
  card.addEventListener('click', () => activate(card));
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(card); }
  });
});
"""


def selection_status_html(selected, required=5, label="FOUNDING TEAM", maximum=False):
    n = int(sum(bool(x) for x in selected))
    if maximum:
        cls = "complete" if n > 0 else "building"
        msg = "Choose anywhere from 0 to 5 Hire labels"
        return f"""<div class="selection-status {cls}"><span class="status-kicker">{label}</span>
        <strong>{n} / {required} Hire labels</strong><span>{msg}</span></div>"""
    if n == required:
        cls, msg = "complete", "Ready to lock in"
    else:
        cls, msg = "building", f"Choose {required - n} more"
    return f"""<div class="selection-status {cls}"><span class="status-kicker">{label}</span>
    <strong>{n} / {required} selected</strong><span>{msg}</span></div>"""


def profile_panel_html(X, title="Team average profile"):
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return f"<div class='profile-panel'><h3>{html.escape(title)}</h3><p class='muted'>Select candidates to see the profile.</p></div>"
    m = X.mean(axis=0)
    labels = [("🧰", "Experience", m[0]), ("🎓", "Qualification", m[1]), ("💪", "Work ethic", m[2]), ("🤝", "Teamwork", m[3])]
    rows = "".join(
        f"<div class='profile-item'><span>{ic} {lab}</span><b>{val:.1f}/5</b><div class='profile-track'><i style='width:{val/5*100:.0f}%'></i></div></div>"
        for ic, lab, val in labels
    )
    return f"<div class='profile-panel'><h3>{html.escape(title)}</h3>{rows}</div>"


def gauge_html(label, value, icon):
    if value is None:
        return f"<div class='gauge-card neutral'><div class='gauge-label'>{icon} {label}</div><div class='gauge-number'>—</div><div class='gauge-note'>Complete your five-person team</div></div>"
    v = float(np.clip(value, 0, 100))
    tone = score_tone(v)
    note = "Excellent" if v >= 85 else "Strong" if v >= 70 else "Needs attention" if v < 55 else "Mixed"
    return f"<div class='gauge-card {tone}'><div class='gauge-label'>{icon} {label}</div><div class='gauge-ring' style='--score:{v:.0f}'><span>{v:.0f}</span></div><div class='gauge-note'>{note}</div></div>"



def scoring_diagnosis_html(X):
    d = workplace_diagnostics(X)
    if d is None:
        return """
        <div class='diagnosis-panel'>
          <div class='diagnosis-title'>SCORING DIAGNOSIS <span>TEMPORARY</span></div>
          <p>Select five people to see exactly how the hidden workplace scores are being calculated.</p>
        </div>
        """

    def bar(label, value, maximum, detail=""):
        pct = 100 * float(value) / float(maximum) if maximum else 0
        pct = float(np.clip(pct, 0, 100))
        detail_html = f"<small>{detail}</small>" if detail else ""
        return f"""
        <div class='diagnosis-row'>
          <div class='diagnosis-row-head'><b>{html.escape(label)}</b><span>+{value:.1f} / {maximum:.0f}</span></div>
          {detail_html}
          <div class='diagnosis-track'><i style='width:{pct:.1f}%'></i></div>
        </div>"""

    eff = d['eff_parts']
    cul = d['culture_parts']
    penalty = d['culture_penalty']
    penalty_html = ""
    if penalty > 0:
        penalty_html = f"<div class='diagnosis-penalty'>Low Culture penalty to Efficiency: <b>−{penalty:.1f}</b></div>"

    return f"""
    <div class='diagnosis-panel'>
      <div class='diagnosis-title'>SCORING DIAGNOSIS <span>TEMPORARY</span></div>
      <div class='diagnosis-explainer'>These are the exact hidden game rules currently driving your scores.</div>

      <div class='diagnosis-section'>
        <h4>🙂 Culture = {d['culture']:.0f}/90</h4>
        {bar('Base score', cul['Base'], 18)}
        {bar('Teamwork', cul['Teamwork'], 48, f"team average {d['team_mean']:.1f}/5")}
        {bar('Work ethic', cul['Work ethic'], 15, f"team average {d['work_mean']:.1f}/5")}
        {bar('Experience / qualification balance', cul['Experience / qualification balance'], 10, f"experience {d['exp_mean']:.1f} vs qualification {d['qual_mean']:.1f}")}
      </div>

      <div class='diagnosis-section'>
        <h4>⚙ Efficiency = {d['efficiency']:.0f}/90</h4>
        {bar('Base score', eff['Base'], 18)}
        {bar('Preparation', eff['Preparation'], 28, f"average max(experience, qualification) = {d['prep_mean']:.1f}/5")}
        {bar('Work ethic', eff['Work ethic'], 32, f"team average {d['work_mean']:.1f}/5")}
        {bar('Teamwork', eff['Teamwork'], 18, f"team average {d['team_mean']:.1f}/5")}
        {penalty_html}
      </div>
    </div>
    """

def founding_preview_html(state):
    s = state or initial_state()
    selected = np.asarray(s.get("initial_selected", [False] * 20), dtype=bool)
    n = int(selected.sum())
    if n == 5:
        X = LEVEL1_VALUES[selected]
        eff, cul = workplace_metrics(X)
    else:
        X = LEVEL1_VALUES[selected] if n else np.empty((0, 4))
        eff, cul = None, None
    return f"""
    <div class="side-panel">
      <div class="side-panel-title"><span>🏢</span><div><small>YOUR WORKPLACE</small><h2>{n}/5 hired</h2></div></div>
      <div class="gauge-grid">{gauge_html('Efficiency', eff, '⚙️')}{gauge_html('Culture', cul, '🙂')}</div>
      <div class="game-tip"><b>Launch requirement:</b> your founding team needs at least <b>75/100 Efficiency</b> and <b>75/100 Culture</b>. There are many different teams that can succeed.</div>
    </div>
    """


def current_batch_panel_html(state):
    s = state or initial_state()
    selected = np.asarray(s.get("current_selected", [False] * 20), dtype=bool)
    n = int(selected.sum())
    X = CURRENT_BATCH[selected] if n else np.empty((0, 4))
    return f"""
    <div class="side-panel">
      <div class="side-panel-title"><span>★</span><div><small>POST-GUILD TRAINING DATA</small><h2>{n}/5 Hire labels</h2></div></div>
      <div class="pathway-split"><div><b>20</b><span>non-Guild applicants labelled</span></div><div><b>{20-n}</b><span>Do not hire labels</span></div></div>
      {profile_panel_html(X, 'Applicants labelled Hire')}
      <div class="game-tip">Choose <b>anywhere from 0 to 5</b> applicants to label Hire. None of these applicants carries the discredited Guild badge.</div>
    </div>
    """


def training_labels_html():
    return f"""
    <section class='label-training-explainer'>
      <div class='label-explainer-kicker'>YOUR DECISIONS BECOME TRAINING DATA</div>
      <h2>You have labelled all 20 founding applicants.</h2>
      <p>The AI does not know who is really a "good" zylometrist. It learns from the answers you gave it.</p>
      <div class='label-flow'>
        <div class='label-count-card hire-label-card'><span>HIRE</span><b>5</b><small>your chosen founders</small></div>
        <div class='label-flow-arrow'>+</div>
        <div class='label-count-card reject-label-card'><span>DO NOT HIRE</span><b>15</b><small>everyone you did not choose</small></div>
        <div class='label-flow-arrow'>→</div>
        <div class='label-count-card data-label-card'><span>TRAINING SET</span><b>20</b><small>labelled examples</small></div>
        <div class='label-flow-arrow'>→</div>
        <div class='label-ai-card'><div class='label-ai-icon'>{icon_svg('train')}</div><div><b>Hiring AI</b><small>learn patterns that predict your labels</small></div></div>
      </div>
      <div class='label-explainer-callout'><b>Key idea:</b> your hiring choices are now the "correct answers" the model will try to reproduce.</div>
    </section>
    """


def training_html(percent, title, message, done=False):
    p = int(np.clip(percent, 0, 100))
    if done:
        icon = "✓"
        cls = "done"
    else:
        icon = "◌"
        cls = "running"
    return f"""<div class="training-console {cls}"><div class="training-icon">{icon}</div>
    <div class="training-copy"><div class="training-title">{html.escape(title)}</div><div class="training-msg">{html.escape(message)}</div>
    <div class="training-track"><div style="width:{p}%"></div></div><div class="training-pct">{p}%</div></div></div>"""


def concept_unlock_html(title, text, icon="◆"):
    return f"""<div class="concept-unlock"><div class="unlock-icon">{icon}</div><div><small>CONCEPT UNLOCKED</small>
    <h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></div></div>"""


def bias_badges_html(state):
    s = state or initial_state()
    badges = [
        ("Label bias", s.get("label_bias_unlocked", False), "🏷"),
        ("Historical bias", s.get("historical_bias_unlocked", False), "🏛"),
        ("Sampling bias", s.get("sampling_bias_unlocked", False), "◫"),
    ]
    bits = []
    for name, unlocked, icon in badges:
        bits.append(f"<div class='bias-badge {'unlocked' if unlocked else 'locked'}'><span>{icon}</span><b>{name}</b><i>{'✓' if unlocked else '🔒'}</i></div>")
    return "<div class='bias-strip'><span class='bias-strip-label'>BIAS JOURNAL</span>" + "".join(bits) + "</div>"


def results_cards_html(eff, cul, title="Workplace result", extra=""):
    return f"""<div class="results-shell"><h2>{html.escape(title)}</h2><div class="gauge-grid">
    {gauge_html('Efficiency', eff, '⚙️')}{gauge_html('Culture', cul, '🙂')}</div>{extra}</div>"""

# ============================================================
# Candidate interaction callbacks
# ============================================================

def toggle_initial_candidate(state, evt: gr.EventData):
    s = dict(state or initial_state())
    selected = list(s.get("initial_selected", [False] * 20))
    idx = int(getattr(evt, "index", -1))
    if not (0 <= idx < len(selected)):
        return make_all_card_data(NAMES, LEVEL1_VALUES, selected, guild_flags=LEVEL1_GUILD), s, selection_status_html(selected), founding_preview_html(s), scoreboard_html(s)

    if selected[idx]:
        selected[idx] = False
    elif sum(selected) < 5:
        selected[idx] = True
    s["initial_selected"] = selected

    mask = np.asarray(selected, dtype=bool)
    if mask.sum() == 5:
        eff, cul = workplace_metrics(LEVEL1_VALUES[mask])
        s["last_efficiency"] = eff
        s["last_culture"] = cul
    else:
        s["last_efficiency"] = None
        s["last_culture"] = None

    return (
        make_all_card_data(NAMES, LEVEL1_VALUES, selected, guild_flags=LEVEL1_GUILD),
        s,
        selection_status_html(selected),
        founding_preview_html(s),
        scoreboard_html(s),
    )


def change_initial_page(delta, state):
    s = dict(state or initial_state())
    max_page = (len(NAMES) - 1) // CARDS_PER_PAGE
    page = int(np.clip(s.get("initial_page", 0) + delta, 0, max_page))
    s["initial_page"] = page
    cards = make_card_data(NAMES, LEVEL1_VALUES, s["initial_selected"], page, guild_flags=LEVEL1_GUILD)
    return cards, s, f"<span class='page-label'>Applicants {page*CARDS_PER_PAGE+1}–{min((page+1)*CARDS_PER_PAGE,20)} of 20 · Page {page+1}/{max_page+1}</span>"



def discussion_team_html(state):
    s = state or initial_state()
    selected = np.asarray(s.get("initial_hires") or s.get("initial_selected", [False] * 20), dtype=bool)
    if selected.sum() != 5:
        return "<div class='warning-card'>Lock a five-person founding team first.</div>"

    idxs = np.flatnonzero(selected)
    cards = []
    for idx in idxs:
        row = LEVEL1_VALUES[idx]
        cards.append(f"""
        <div class='discussion-hire-card'>
          <div class='discussion-hire-head'>
            <div class='abstract-avatar avatar-{idx % 8}'>{html.escape(NAMES[idx][0])}</div>
            <div><div class='discussion-hire-name'>{html.escape(NAMES[idx])}</div><div class='discussion-hire-sub'>FOUNDING HIRE</div>
            {"<div class='discussion-guild-badge'>★ ZYLOMETRY GUILD ACCREDITED</div>" if LEVEL1_GUILD[idx] else ""}</div>
          </div>
          <div class='discussion-trait'><span>Experience</span><span class='discussion-rating'>{rating_html(row[0])}</span></div>
          <div class='discussion-trait'><span>Qualification</span><span class='discussion-rating'>{rating_html(row[1])}</span></div>
          <div class='discussion-trait'><span>Work ethic</span><span class='discussion-rating'>{rating_html(row[2])}</span></div>
          <div class='discussion-trait'><span>Teamwork</span><span class='discussion-rating'>{rating_html(row[3])}</span></div>
        </div>
        """)

    eff, cul = workplace_metrics(LEVEL1_VALUES[selected])
    return f"""
    <section class='founding-discussion-shell'>
      <div class='discussion-pause-banner'>
        <div class='discussion-pause-kicker'>CLASSROOM PAUSE</div>
        <h1>Pause here for classroom discussion.</h1>
        <p>These are the five people you chose to launch your Zylometry company. Compare your team with the people around you before training the AI.</p>
      </div>
      <div class='discussion-score-row'>
        <div><span>EFFICIENCY</span><b>{eff:.0f}/100</b></div>
        <div><span>CULTURE</span><b>{cul:.0f}/100</b></div>
      </div>
      <div class='discussion-team-profile'>
        {profile_panel_html(LEVEL1_VALUES[selected], 'Selected team profile — what did your choices prioritise?')}
      </div>
      <div class='discussion-selected-grid'>{''.join(cards)}</div>
      <div class='discussion-prompts'>
        <strong>Discuss:</strong> How did you prioritise who you wanted to hire? How do your efficiency and culture compare to others in the class?
      </div>
    </section>
    """

def lock_initial_team(state):
    s = dict(state or initial_state())
    selected = np.asarray(s.get("initial_selected", [False] * 20), dtype=bool)
    if selected.sum() != 5:
        return (
            "<div class='warning-card'>Select exactly five candidates first.</div>",
            s, scoreboard_html(s), discussion_team_html(s),
            gr.Group(visible=True), gr.Group(visible=False),
        )

    eff, cul = workplace_metrics(LEVEL1_VALUES[selected])
    if eff < FOUNDING_LAUNCH_THRESHOLD or cul < FOUNDING_LAUNCH_THRESHOLD:
        return (
            f"<div class='warning-card'>Your founding team is not ready to launch yet. Reach at least <b>{FOUNDING_LAUNCH_THRESHOLD:.0f}/100</b> for both Efficiency and Culture. Current scores: Efficiency <b>{eff:.0f}</b>, Culture <b>{cul:.0f}</b>.</div>",
            s, scoreboard_html(s), discussion_team_html(s),
            gr.Group(visible=True), gr.Group(visible=False),
        )

    s["initial_hires"] = selected.tolist()
    s["ai_trained"] = False
    s["growth_round"] = 0
    s["applicants_remaining"] = STARTING_APPLICANTS
    s["display_applicants"] = STARTING_APPLICANTS
    s["employees"] = 5
    s["growth_workforce"] = LEVEL1_VALUES[selected].tolist()
    s["company_value"] = 0.35 + 0.65 * ((eff + cul) / 200)
    s["last_efficiency"] = eff
    s["last_culture"] = cul
    s["scaling_start_value"] = s["company_value"]
    s["scaling_start_efficiency"] = eff
    s["scaling_start_culture"] = cul
    s["scaling_start_employees"] = 5

    return (
        "<div class='success-card'>Founding team locked.</div>",
        s, scoreboard_html(s), discussion_team_html(s),
        gr.Group(visible=False), gr.Group(visible=True),
    )


def toggle_current_candidate(state, evt: gr.EventData):
    s = dict(state or initial_state())
    selected = list(s.get("current_selected", [False] * 20))
    idx = int(getattr(evt, "index", -1))
    if 0 <= idx < len(selected):
        if selected[idx]:
            selected[idx] = False
        elif sum(selected) < 5:
            selected[idx] = True
    s["current_selected"] = selected
    return (
        make_card_data(CURRENT_NAMES, CURRENT_BATCH, selected, s.get("current_page", 0), CURRENT_PATHWAYS, CURRENT_GUILD),
        s,
        selection_status_html(selected, label="POST-GUILD LABELS", maximum=True),
        current_batch_panel_html(s),
    )


def change_current_page(delta, state):
    s = dict(state or initial_state())
    max_page = (len(CURRENT_NAMES) - 1) // CARDS_PER_PAGE
    page = int(np.clip(s.get("current_page", 0) + delta, 0, max_page))
    s["current_page"] = page
    cards = make_card_data(CURRENT_NAMES, CURRENT_BATCH, s["current_selected"], page, CURRENT_PATHWAYS, CURRENT_GUILD)
    return cards, s, f"<span class='page-label'>Applicants {page*CARDS_PER_PAGE+1}–{min((page+1)*CARDS_PER_PAGE,20)} of 20 · Page {page+1}/{max_page+1}</span>"

# ============================================================
# Mission 2 — animated training
# ============================================================

async def train_hiring_ai(state):
    s = dict(state or initial_state())
    if s.get("initial_hires") is None:
        yield "<div class='warning-card'>Complete Mission 1 first.</div>", s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
        return

    y = np.asarray(s["initial_hires"], dtype=int)
    model = train_student_model(augment_features(LEVEL1_VALUES, LEVEL1_GUILD), y)
    pos = LEVEL1_VALUES[y.astype(bool)]
    neg = LEVEL1_VALUES[~y.astype(bool)]
    diffs = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
    strongest = FEATURES[:4][int(np.argmax(diffs))]

    steps = [
        "Reading your 5 Hire and 15 Do not hire labels…",
        "Looking for patterns that separate your Hire examples from your Do not hire examples…",
        "Checking whether Experience helps predict your decisions…",
        "Checking whether Qualification helps predict your decisions…",
        "Checking whether Work ethic helps predict your decisions…",
        "Checking whether Teamwork helps predict your decisions…",
        "Checking the other information shown on the applicant cards…",
        f"{strongest} appears to be one useful signal in your labelled examples…",
        "Combining the patterns the AI found across your 20 examples…",
        "Preparing the hiring model to score new applicants…",
    ]

    for i, message in enumerate(steps):
        pct = int((i + 1) / len(steps) * 94)
        yield training_html(pct, "TRAINING HIRING MODEL", message), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=False)
        await asyncio.sleep(TRAINING_ANIMATION_SECONDS / len(steps))

    s["ai_trained"] = True
    final = training_html(100, "HIRING MODEL READY", "Training complete. Your model is ready to screen new applicants.", done=True) + f"""
    <div class='ai-ready-card'><div class='ai-ready-icon'>{icon_svg('train')}</div><div><h2>Your hiring model is ready</h2>
    <p>It learned from <b>5 Hire</b> and <b>15 Do not hire</b> labels. <b>{html.escape(strongest)}</b> was one of the visible attributes that differed between those groups.</p>
    <p class='callout'>The AI has learned patterns from <b>your labelled examples</b>. You did not give it an explicit hiring formula.</p></div></div>"""
    yield final, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)


def go_training():
    return gr.Walkthrough(selected=2)

def go_growth(state):
    s = state or initial_state()
    if not s.get("ai_trained", False):
        return gr.Walkthrough(selected=2), growth_screen_html(s)
    return gr.Walkthrough(selected=3), growth_screen_html(s)

# ============================================================
# Mission 3 — growth dashboard and animated deployment
# ============================================================

def workforce_average_html(state):
    s = state or initial_state()
    X = np.asarray(s.get("growth_workforce") or [], dtype=float)
    if X.size == 0:
        return "<div class='workforce-average-panel'><div class='panel-kicker'>CURRENT COMPANY PROFILE</div><div class='muted'>No employees yet.</div></div>"
    X = X.reshape(-1, 4)
    means = X.mean(axis=0)
    labels = [("Experience", means[0]), ("Qualification", means[1]), ("Work ethic", means[2]), ("Teamwork", means[3])]
    items = "".join(
        f"<div class='workforce-average-item'><span>{html.escape(label)}</span><b>{value:.1f}/5</b><div class='workforce-mini-track'><i style='width:{value/5*100:.0f}%'></i></div></div>"
        for label, value in labels
    )
    return f"<div class='workforce-average-panel'><div class='panel-kicker'>CURRENT COMPANY PROFILE</div><div class='workforce-average-grid'>{items}</div></div>"


def hire_inspection_html(state):
    s = state or initial_state()
    hires = np.asarray(s.get("last_round_hires") or [], dtype=float)
    applicants = np.asarray(s.get("last_round_applicants") or [], dtype=float)
    if hires.size == 0:
        return ""
    hires = hires.reshape(-1, 4)
    if applicants.size == 0:
        applicants = hires.copy()
    applicants = applicants.reshape(-1, 4)
    labels = ["Experience", "Qualification", "Work ethic", "Teamwork"]
    rows = []
    for j, label in enumerate(labels):
        hire_mean = float(hires[:, j].mean())
        strong_count = int((hires[:, j] >= 4).sum())
        rows.append(
            f"<div class='hire-inspect-stat'><span>{html.escape(label)}</span>"
            f"<b>{strong_count} of {len(hires)} hires had 4+ stars</b>"
            f"<small>Average among new hires: {hire_mean:.1f}/5</small></div>"
        )
    return f"""<details class='hire-inspection'><summary><span class='hire-inspect-cta'>INSPECT THE {len(hires)} NEW HIRES</span><span class='hire-inspect-sub'>See the strengths of the people your AI selected</span></summary><div class='hire-inspect-grid'>{''.join(rows)}</div></details>"""


def growth_screen_html(state, status="Ready to deploy", screened=0, hired=None, value_delta=None, comparison_html=""):
    s = state or initial_state()
    remaining = int(s.get("applicants_remaining", STARTING_APPLICANTS))
    employees = int(s.get("employees", 0))
    value = float(s.get("company_value", STARTING_COMPANY_VALUE))
    eff = s.get("last_efficiency")
    cul = s.get("last_culture")
    round_no = int(s.get("growth_round", 0))
    pool_size = 200 if s.get("guild_scandal", False) else STARTING_APPLICANTS
    remain_pct = int(np.clip(100 * remaining / max(1, pool_size), 0, 100))
    gain = "" if value_delta is None else f"<span class='delta'>+${value_delta:.2f}M this round</span>"
    hired_display = "—" if hired is None else str(int(hired))
    screened_display = GROWTH_BATCH_SIZE if not screened else int(screened)
    event = ""
    if remaining <= 0 and not s.get("guild_scandal", False):
        event = """<div class='event-trigger-card'><span>!</span><div><small>NEWS ALERT</small><h3>Applicant pool exhausted</h3><p>You have screened everyone currently available.</p></div></div>"""
    return f"""
    <div class='growth-grid'>
      <div class='pipeline-card'>
        <div class='panel-kicker'>HIRING PIPELINE · ROUND {round_no + (0 if screened else 1)}</div>
        <div class='pipeline-flow'><div><i>AI</i><b>AI screens</b><span>{screened_display} applicants</span></div><em>→</em>
        <div><i>▤</i><b>Scores</b><span>learned hiring preferences</span></div><em>→</em>
        <div><i>✓</i><b>{hired_display} hired</b><span>only high-scoring applicants</span></div></div>
        <div class='pipeline-status'>{html.escape(status)}</div>
      </div>
      <div class='ticker-card'>
        <div class='panel-kicker'>APPLICANTS REMAINING</div><div class='ticker-number'>{remaining}</div>
        <div class='ticker-label'>available applicants</div><div class='ticker-track'><div style='width:{remain_pct}%'></div></div>
        <div class='ticker-simple-note'>{pool_size} applicants in this pool</div>
      </div>
      <div class='growth-card'>
        <div class='panel-kicker'>COMPANY GROWTH</div>
        <div class='growth-stat-row'><div><small>EMPLOYEES</small><b>{employees}</b></div><div><small>VALUE</small><b>${value:.2f}M</b>{gain}</div></div>
        <div class='mini-result-row'><div><span>Efficiency</span><b>{fmt_metric(eff)}</b></div><div><span>Culture</span><b>{fmt_metric(cul)}</b></div></div>
        {workforce_average_html(s)}
      </div>
    </div>{hire_inspection_html(s)}{comparison_html}{event}
    """


def scaling_completion_html(state):
    s = state or initial_state()
    start_emp = int(s.get("scaling_start_employees") or 5)
    start_value = float(s.get("scaling_start_value") or 0.0)
    start_eff = float(s.get("scaling_start_efficiency") or 0.0)
    start_cul = float(s.get("scaling_start_culture") or 0.0)
    hired = int(s.get("employees", 0)) - start_emp
    value_gain = float(s.get("company_value", 0.0)) - start_value
    eff_delta = float(s.get("last_efficiency") or 0.0) - start_eff
    cul_delta = float(s.get("last_culture") or 0.0) - start_cul

    def change_phrase(name, delta):
        if delta >= 0:
            return f"{name} <b>gained {delta:.1f}</b> points"
        return f"{name} <b>dropped {abs(delta):.1f}</b> points"

    drop_names = []
    if eff_delta < 0: drop_names.append("Efficiency")
    if cul_delta < 0: drop_names.append("Culture")
    reassurance = ""
    if drop_names:
        what = " and ".join(drop_names)
        reassurance = f"<div class='scaling-reassurance'>No worries about the slight drop in {what} — that’s the price of scaling up!</div>"

    return f"""<div class='scaling-complete-card'>
      <div class='round-comparison-kicker'>APPLICANT POOL COMPLETE</div>
      <h2>Well done, you hired <b>{hired}</b> employees and grew your company value by <b>${value_gain:.2f}M</b>.</h2>
      <p>{change_phrase('Efficiency', eff_delta)} and {change_phrase('Culture', cul_delta)}.</p>
      {reassurance}
    </div>"""


def find_zero_hire_post_news_batch(model, threshold, state, batch_n, round_idx):
    """Draw a comparable strong batch that this historically biased model rejects.

    We only search within the same founder-matched quality distribution; the
    difference is that none of these applicants carry Guild accreditation.
    """
    best = None
    for attempt in range(200):
        seed = 22000 + round_idx * 1000 + attempt
        X, guild = founder_like_market_batch(seed, batch_n, state, guild_probability=0.0)
        probs = model.predict_proba(augment_features(X, guild))[:, 1]
        n = int((probs >= threshold).sum())
        if best is None or n < best[0]:
            best = (n, X, guild, probs)
        if n == 0:
            return X, guild, probs
    return best[1], best[2], best[3]


def deploy_growth_round(state):
    s = dict(state or initial_state())
    controls_hidden = (gr.Button(visible=False),) * 4
    if not s.get("ai_trained", False):
        yield "<div class='warning-card'>Train your model first.</div>", s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False)
        return

    remaining = int(s.get("applicants_remaining", STARTING_APPLICANTS))
    s["last_round_hires"] = []
    s["last_round_applicants"] = []
    s["last_round_hired_count"] = 0
    post_scandal = bool(s.get("guild_scandal", False))

    if remaining <= 0 and not post_scandal:
        final = growth_screen_html(s, "Applicant pool complete", comparison_html=scaling_completion_html(s))
        yield final, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False)
        return

    if post_scandal and int(s.get("guild_crisis_round", 0)) >= POST_NEWS_DIAGNOSIS_ROUNDS:
        yield growth_screen_html(s, "Pause to diagnose the hiring process"), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=True)
        return

    round_idx = int(s.get("growth_round", 0))
    y = np.asarray(s["initial_hires"], dtype=int)
    model = train_student_model(augment_features(LEVEL1_VALUES, LEVEL1_GUILD), y)
    threshold = founding_hire_threshold(model, s)
    batch_n = min(GROWTH_BATCH_SIZE, remaining)

    if post_scandal:
        applicants, guild_flags, probs = find_zero_hire_post_news_batch(model, threshold, s, batch_n, int(s.get("guild_crisis_round", 0)))
    else:
        applicants, guild_flags, probs = find_stable_pre_news_batch(model, threshold, s, batch_n, round_idx)

    hired_mask = probs >= threshold
    n_hired = int(hired_mask.sum())
    new_hires = applicants[hired_mask]

    existing = np.asarray(s.get("growth_workforce") or [], dtype=int)
    if n_hired:
        workforce = np.vstack([existing, new_hires]) if len(existing) else new_hires.copy()
    else:
        workforce = existing.copy()
    efficiency, culture = workplace_metrics(workforce) if len(workforce) else (0.0, 0.0)
    value_gain = deployment_value_gain(efficiency, culture, n_hired)

    start_remaining = remaining
    start_value = float(s.get("company_value", STARTING_COMPANY_VALUE))
    start_employees = int(s.get("employees", 0))
    frames = 8
    statuses = ["Opening applications…", "AI scoring candidates…", "Comparing scores with founding Hire labels…", "Checking the high-score threshold…", "Preparing offers…", "Onboarding hires…", "Updating workplace…", "Hiring round complete"]
    for f in range(1, frames + 1):
        frac = f / frames
        temp = dict(s)
        temp["applicants_remaining"] = int(round(start_remaining - batch_n * frac))
        temp["display_applicants"] = temp["applicants_remaining"]
        temp["employees"] = int(round(start_employees + n_hired * frac))
        temp["company_value"] = start_value + value_gain * frac
        temp["last_efficiency"] = efficiency
        temp["last_culture"] = culture
        yield growth_screen_html(temp, statuses[f-1], int(round(batch_n * frac)), int(round(n_hired * frac)), value_gain * frac), s, scoreboard_html(temp), *controls_hidden
        time.sleep(DEPLOY_ANIMATION_SECONDS / frames)

    s["growth_round"] = round_idx + 1
    s["applicants_screened"] = int(s.get("applicants_screened", 0)) + batch_n
    s["applicants_remaining"] = max(0, remaining - batch_n)
    s["display_applicants"] = s["applicants_remaining"]
    s["employees"] = start_employees + n_hired
    s["company_value"] = start_value + value_gain
    s["growth_workforce"] = workforce.tolist() if len(workforce) else []
    s["last_round_hires"] = new_hires.tolist() if n_hired else []
    s["last_round_applicants"] = applicants.tolist()
    s["last_round_hired_count"] = n_hired
    s["last_efficiency"] = efficiency
    s["last_culture"] = culture

    if not post_scandal:
        s["last_pre_scandal_hired"] = n_hired
        if s["applicants_remaining"] == 0:
            final = growth_screen_html(s, "Applicant pool complete", batch_n, n_hired, value_gain, scaling_completion_html(s))
            yield final, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False)
        else:
            final = growth_screen_html(s, f"Hiring round complete — {n_hired} applicants were hired", batch_n, n_hired, value_gain)
            yield final, s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False)
        return

    # Store strong rejected candidates for later classroom diagnosis.
    rejected_idx = np.flatnonzero(~hired_mask)
    if len(rejected_idx):
        order = rejected_idx[np.argsort(agreed_score(applicants[rejected_idx]))[::-1]]
        chosen = order[:5]
        s["diagnosis_rejected"] = applicants[chosen].tolist()
        s["diagnosis_scores"] = probs[chosen].tolist()
        offset = (round_idx * GROWTH_BATCH_SIZE) % len(MARKET_NAMES)
        s["diagnosis_names"] = [MARKET_NAMES[(offset + i) % len(MARKET_NAMES)] for i in range(len(chosen))]

    s["guild_crisis_round"] = int(s.get("guild_crisis_round", 0)) + 1
    s["guild_crisis_screened"] = int(s.get("guild_crisis_screened", 0)) + batch_n
    s["guild_crisis_hired"] = int(s.get("guild_crisis_hired", 0)) + n_hired

    crisis_round = s["guild_crisis_round"]
    if crisis_round == 1 and n_hired == 0:
        comparison = "<div class='round-comparison-card'><div class='round-comparison-kicker'>THAT’S STRANGE…</div><h2>You didn’t hire any applicants.</h2><p>The AI screened 50 people, but nobody scored highly enough to be hired. Run another hiring round and see if it happens again.</p></div>"
    elif crisis_round < POST_NEWS_DIAGNOSIS_ROUNDS and n_hired == 0:
        comparison = f"<div class='round-comparison-card'><div class='round-comparison-kicker'>STILL NO HIRES</div><h2>Round {crisis_round}: another 50 applicants screened, and nobody was hired.</h2><p>Keep going until you have screened the whole applicant pool.</p></div>"
    elif crisis_round < POST_NEWS_DIAGNOSIS_ROUNDS:
        comparison = f"<div class='round-comparison-card'><div class='round-comparison-kicker'>VERY FEW HIRES</div><h2>This round hired only {n_hired} applicants.</h2><p>Keep going until you have screened the whole applicant pool.</p></div>"
    else:
        previous_hired = int(s.get("pre_scandal_total_hired", 0))
        current_hired = int(s.get("guild_crisis_hired", 0))
        if current_hired == 0:
            comparison = f"""<div class='round-comparison-card crisis-summary-card'><div class='round-comparison-kicker'>SOMETHING HAS CHANGED</div>
            <h2>In the previous 200 applicants, you hired <b>{previous_hired}</b>. In this pool, all <b>200 were rejected</b>.</h2>
            <p>That is a very different result from before. Let’s diagnose what is going on.</p></div>"""
        else:
            comparison = f"""<div class='round-comparison-card crisis-summary-card'><div class='round-comparison-kicker'>SOMETHING HAS CHANGED</div>
            <h2>In the previous 200 applicants, you hired <b>{previous_hired}</b>. In this pool, you hired only <b>{current_hired}</b>.</h2>
            <p>That is a very different result from before. Let’s diagnose what is going on.</p></div>"""

    if crisis_round >= POST_NEWS_DIAGNOSIS_ROUNDS:
        final = growth_screen_html(s, "Applicant pool complete", batch_n, n_hired, value_gain, comparison)
        yield final, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=True)
    else:
        final = growth_screen_html(s, "Hiring round complete", batch_n, n_hired, value_gain, comparison)
        yield final, s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False)


def guild_news_html():
    return """
    <section class='breaking-news-card'>
      <div class='breaking-news-kicker'>BREAKING INDUSTRY NEWS</div>
      <h1>Zylometry Guild Accreditation Exposed as a Pay-to-Play Scam</h1>
      <p>An industry investigation finds that Guild accreditation was based on membership fees rather than Zylometry ability. Employers and applicants abandon the badge almost overnight.</p>
      <div class='breaking-news-detail'>The skills needed to do good Zylometry have not changed. What changed is whether people believe the Guild badge means anything.</div>
    </section>
    """


def open_guild_event(state):
    s = dict(state or initial_state())
    s['last_round_hires'] = []
    s['last_round_hired_count'] = 0
    return guild_news_html(), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=False)


def acknowledge_guild_event(state):
    s = dict(state or initial_state())
    founding_employees = int(s.get("scaling_start_employees") or 5)
    s["pre_scandal_total_hired"] = max(0, int(s.get("employees", 0)) - founding_employees)
    s["applicants_remaining"] = 200
    s["display_applicants"] = 200
    s["guild_scandal"] = True
    s["guild_crisis_round"] = 0
    s["guild_crisis_hired"] = 0
    s["guild_crisis_screened"] = 0
    s['last_round_hires'] = []
    s['last_round_hired_count'] = 0
    screen = growth_screen_html(s, "A fresh applicant pool is ready. Run another hiring round.")
    return screen, s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), gr.Button(visible=False)


def training_example_cards_html(state):
    s = state or initial_state()
    y = np.asarray(s.get("initial_hires") or [False] * 20, dtype=int)
    model = train_student_model(augment_features(LEVEL1_VALUES, LEVEL1_GUILD), y)
    probs = model.predict_proba(augment_features(LEVEL1_VALUES, LEVEL1_GUILD))[:, 1]
    hire_idx = np.flatnonzero(y.astype(bool))
    reject_idx = np.flatnonzero(~y.astype(bool))
    high_idx = hire_idx[np.argsort(probs[hire_idx])[::-1][:3]]
    low_idx = reject_idx[np.argsort(probs[reject_idx])[:3]]

    def cards(indices, cls):
        out = []
        for idx in indices:
            row = LEVEL1_VALUES[idx]
            name = NAMES[idx]
            initials = ''.join(p[0] for p in name.split()[:2]).upper()
            guild = "<div class='guild-badge diagnosis-training-badge'>★ ZYLOMETRY GUILD ACCREDITED</div>" if LEVEL1_GUILD[idx] else ""
            original_label = "HIRE" if y[idx] else "DO NOT HIRE"
            label_cls = "hire" if y[idx] else "reject"
            out.append(f"""<div class='training-sample-card {cls}'><div class='candidate-head'><div class='abstract-avatar avatar-{idx%8}'>{html.escape(initials)}</div><div><div class='candidate-name'>{html.escape(name)}</div>{guild}</div></div>
            <div class='trait-row'><span>Experience</span><span class='rating'>{rating_html(row[0])}</span></div><div class='trait-row'><span>Qualification</span><span class='rating'>{rating_html(row[1])}</span></div><div class='trait-row'><span>Work ethic</span><span class='rating'>{rating_html(row[2])}</span></div><div class='trait-row'><span>Teamwork</span><span class='rating'>{rating_html(row[3])}</span></div><div class='training-label-pill {label_cls}'>{original_label}</div></div>""")
        return ''.join(out)

    return f"""<div class='training-example-review'><div class='panel-kicker'>LOOK BACK AT WHAT YOUR AI LEARNED</div><h2>Examine the training data your AI learns from</h2><div class='training-example-columns'><div><h3>3 the AI learnt to hire</h3><div class='training-sample-grid'>{cards(high_idx,'high-score')}</div></div><div><h3>3 the AI learnt Do not hire</h3><div class='training-sample-grid'>{cards(low_idx,'low-score')}</div></div></div></div>"""


def diagnosis_screen_html(state):
    s = state or initial_state()
    values = np.asarray(s.get("diagnosis_rejected") or [], dtype=int)
    names = list(s.get("diagnosis_names") or [])
    scores = list(s.get("diagnosis_scores") or [])
    cards = []
    for i, row in enumerate(values):
        name = names[i] if i < len(names) else f"Applicant {i+1}"
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        impact = estimated_candidate_value_percent(row)
        cards.append(f"""
        <div class='diagnosis-candidate-card'>
          <div class='candidate-head'><div class='abstract-avatar avatar-{i % 8}'>{html.escape(initials)}</div><div><div class='candidate-name'>{html.escape(name)}</div></div></div>
          <div class='trait-row'><span>Experience</span><span class='rating'>{rating_html(row[0])}</span></div>
          <div class='trait-row'><span>Qualification</span><span class='rating'>{rating_html(row[1])}</span></div>
          <div class='trait-row'><span>Work ethic</span><span class='rating'>{rating_html(row[2])}</span></div>
          <div class='trait-row'><span>Teamwork</span><span class='rating'>{rating_html(row[3])}</span></div>
          <div class='diagnosis-ai-decision rejected-only'><strong>REJECTED</strong></div>
          <div class='value-impact full'><span>Estimated company value</span><b>+{impact:.1f}%</b></div>
        </div>""")
    cards_html = "".join(cards) if cards else "<div class='muted'>No rejected applicants were stored from the previous round.</div>"
    return f"""
    <div class='diagnosis-screen-shell'>
      <div class='discussion-pause-banner'><div class='discussion-pause-kicker'>CLASSROOM PAUSE</div><h1>Diagnose the AI.</h1><p>These were some of the strongest applicants your AI rejected in the recent hiring rounds.</p></div>
      <div class='diagnosis-full-card-grid'>{cards_html}</div>
      <div class='discussion-card large'><div class='discussion-icon'>?</div><div><small>DISCUSS</small><h2>Why do you think these applicants were rejected?</h2><p>Look back at the information the AI learned from your original Hire and Do not hire decisions.</p></div></div>
      {training_example_cards_html(s)}
    </div>
    """


def go_diagnose(state):
    return gr.Walkthrough(selected=4), diagnosis_screen_html(state)


def go_update():
    return gr.Walkthrough(selected=5)

# ============================================================
# Mission 5 — choose how to fix the AI
# ============================================================

def choose_remove_guild(state):
    s = dict(state or initial_state())
    s["fix_strategy"] = "remove_guild"
    s["repair_trained"] = False
    return s, "<div class='fix-selected-card'><b>Selected:</b> Keep the original 20 labels, but remove Guild accreditation from the information available to the AI.</div>", gr.Group(visible=False), gr.Button(visible=True)


def choose_fresh_data(state):
    s = dict(state or initial_state())
    s["fix_strategy"] = "fresh_data"
    s["repair_trained"] = False
    s["current_selected"] = [False] * 20
    return s, "<div class='fix-selected-card'><b>Selected:</b> Collect a completely fresh labelled training set from the current applicant market.</div>", gr.Group(visible=True), gr.Button(visible=True), make_all_card_data(CURRENT_NAMES, CURRENT_BATCH, s["current_selected"], guild_flags=CURRENT_GUILD), selection_status_html(s["current_selected"], label="FRESH TRAINING LABELS")


def toggle_fresh_candidate(state, evt: gr.EventData):
    s = dict(state or initial_state())
    selected = list(s.get("current_selected", [False] * 20))
    idx = int(getattr(evt, "index", -1))
    if 0 <= idx < len(selected):
        if selected[idx]:
            selected[idx] = False
        elif sum(selected) < 5:
            selected[idx] = True
    s["current_selected"] = selected
    return make_all_card_data(CURRENT_NAMES, CURRENT_BATCH, selected, guild_flags=CURRENT_GUILD), s, selection_status_html(selected, label="FRESH TRAINING LABELS")


def repair_training_data(state):
    s = state or initial_state()
    strategy = s.get("fix_strategy")
    if strategy == "remove_guild":
        y = np.asarray(s["initial_hires"], dtype=int)
        return LEVEL1_VALUES, y, False
    if strategy == "fresh_data":
        y = np.asarray(s.get("current_selected", [False] * 20), dtype=int)
        return CURRENT_BATCH, y, True
    raise ValueError("Choose a repair strategy first.")


async def train_repaired_ai(state):
    s = dict(state or initial_state())
    strategy = s.get("fix_strategy")
    if strategy not in {"remove_guild", "fresh_data"}:
        yield "<div class='warning-card'>Choose how you want to fix the AI first.</div>", s, gr.Button(visible=True), gr.Button(visible=False)
        return
    if strategy == "fresh_data" and sum(s.get("current_selected", [])) != 5:
        yield "<div class='warning-card'>Choose exactly five of the 20 fresh applicants to label Hire.</div>", s, gr.Button(visible=True), gr.Button(visible=False)
        return

    if strategy == "remove_guild":
        messages = ["Removing Guild accreditation from the AI inputs…", "Keeping your original 5 Hire and 15 Do not hire labels…", "Looking again for patterns in Experience, Qualification, Work ethic and Teamwork…", "Retraining the hiring model…", "Checking the repaired model on current applicants…"]
    else:
        messages = ["Creating a completely fresh training set…", "Recording your 5 Hire and 15 Do not hire labels…", "Learning patterns from the current applicant market…", "Training a new hiring model…", "Checking the new model on current applicants…"]
    for i,msg in enumerate(messages):
        yield training_html(int((i+1)/len(messages)*94), "RETRAINING HIRING MODEL", msg), s, gr.Button(visible=False), gr.Button(visible=False)
        await asyncio.sleep(1.5)
    s["repair_trained"] = True
    final = training_html(100, "HIRING MODEL READY", "The updated model is ready to be re-deployed.", done=True) + "<div class='ai-ready-card'><div class='ai-ready-icon'>✓</div><div><h2>Your fix is ready to test</h2><p>Re-deploy the AI on a fresh pool of 200 applicants and see whether hiring recovers.</p></div></div>"
    yield final, s, gr.Button(visible=False), gr.Button(visible=True)


def start_repaired_deployment(state):
    s = dict(state or initial_state())
    if not s.get("repair_trained"):
        return gr.Walkthrough(selected=5), s, repaired_growth_html(s)
    s["repair_active"] = True
    s["applicants_remaining"] = 200
    s["display_applicants"] = 200
    s["repair_hired_total"] = 0
    s["repair_screened_total"] = 0
    s["repair_start_employees"] = int(s.get("employees", 0))
    s["repair_start_value"] = float(s.get("company_value", 0.0))
    return gr.Walkthrough(selected=6), s, repaired_growth_html(s)


def repaired_training_data(state):
    """Return the active repaired training set in four-feature form plus labels."""
    s = state or initial_state()
    strategy = s.get("fix_strategy")
    if strategy == "remove_guild":
        y = np.asarray(s["initial_hires"], dtype=int)
        return LEVEL1_VALUES.copy(), y, False
    y = np.asarray(s.get("current_selected", [False] * 20), dtype=int)
    return CURRENT_BATCH.copy(), y, True


def calibrate_repaired_threshold(model, uses_guild, state):
    """Calibrate the repaired AI to the company's earlier local hiring rate.

    Classifier probabilities can shift after retraining, so reusing a numerical
    cutoff is misleading. We instead calibrate against a representative local
    applicant pool and target approximately the company's pre-scandal hire rate.
    """
    s = state or initial_state()
    previous = int(s.get("pre_scandal_total_hired", 0))
    target_rate = previous / 200.0 if previous > 0 else 0.20
    target_rate = float(np.clip(target_rate, 0.10, 0.35))
    cal, guild = post_guild_batch(55123, 1200, target_score=3.80)
    Xpred = augment_features(cal, guild) if uses_guild else cal
    probs = model.predict_proba(Xpred)[:, 1]
    return float(np.quantile(probs, 1.0 - target_rate))


def get_repaired_model_and_threshold(state):
    s = state or initial_state()
    X4, y, uses_guild = repaired_training_data(s)
    Xtrain = augment_features(X4, np.zeros(len(X4), dtype=int)) if uses_guild else X4
    model = train_student_model(Xtrain, y)
    threshold = calibrate_repaired_threshold(model, uses_guild, s)
    return model, threshold, uses_guild


def repaired_growth_html(state, status="Ready to re-deploy", screened=0, hired=None, value_delta=None, complete=False):
    base = growth_screen_html(state, status, screened, hired, value_delta)
    if not complete:
        return base
    s = state or initial_state()
    value_gain = float(s.get("company_value", 0.0)) - float(s.get("repair_start_value") or 0.0)
    return base + f"""<div class='repair-success-card'><div class='round-comparison-kicker'>LOCAL HIRING COMPLETE</div><h2>Well done, you hired <b>{int(s.get('repair_hired_total',0))}</b> applicants and grew company value by <b>+${value_gain:.2f}M</b>.</h2><p>Your updated AI is hiring successfully from the local applicant market again.</p></div>"""


def deploy_repaired_round(state):
    s = dict(state or initial_state())
    if not s.get("repair_trained"):
        yield "<div class='warning-card'>Retrain the AI first.</div>", s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
        return
    remaining = int(s.get("applicants_remaining", 200))
    s["last_round_hires"] = []
    s["last_round_applicants"] = []
    s["last_round_hired_count"] = 0
    if remaining <= 0:
        yield repaired_growth_html(s, "Re-deployment complete", complete=True), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)
        return

    model, threshold, uses_guild = get_repaired_model_and_threshold(s)
    batch_n = min(REPAIR_BATCH_SIZE, remaining)
    applicants, guild = post_guild_batch(44000 + int(s.get("repair_screened_total", 0)), batch_n, target_score=3.80)
    Xpred = augment_features(applicants, guild) if uses_guild else applicants
    probs = model.predict_proba(Xpred)[:, 1]
    hired_mask = probs >= threshold
    n_hired = int(hired_mask.sum())
    new_hires = applicants[hired_mask]
    existing = np.asarray(s.get("growth_workforce") or [], dtype=int)
    workforce = np.vstack([existing, new_hires]) if n_hired and len(existing) else (new_hires.copy() if n_hired else existing.copy())
    eff, cul = workplace_metrics(workforce) if len(workforce) else (0.0, 0.0)
    gain = deployment_value_gain(eff, cul, n_hired)

    start_remaining = remaining
    start_employees = int(s.get("employees", 0))
    start_value = float(s.get("company_value", 0.0))
    frames = 8
    statuses = ["Opening applications…", "AI scoring candidates…", "Comparing candidates with the updated training data…", "Checking the hiring threshold…", "Preparing offers…", "Onboarding hires…", "Updating workplace…", "Hiring round complete"]
    for f in range(1, frames + 1):
        frac = f / frames
        temp = dict(s)
        temp["applicants_remaining"] = int(round(start_remaining - batch_n * frac))
        temp["display_applicants"] = temp["applicants_remaining"]
        temp["employees"] = int(round(start_employees + n_hired * frac))
        temp["company_value"] = start_value + gain * frac
        temp["last_efficiency"] = eff
        temp["last_culture"] = cul
        yield repaired_growth_html(temp, statuses[f-1], int(round(batch_n * frac)), int(round(n_hired * frac)), gain * frac, False), s, scoreboard_html(temp), gr.Button(visible=False), gr.Button(visible=False)
        time.sleep(REDEPLOY_ANIMATION_SECONDS / frames)

    s["applicants_screened"] = int(s.get("applicants_screened", 0)) + batch_n
    s["repair_screened_total"] = int(s.get("repair_screened_total", 0)) + batch_n
    s["repair_hired_total"] = int(s.get("repair_hired_total", 0)) + n_hired
    s["applicants_remaining"] = max(0, remaining - batch_n)
    s["display_applicants"] = s["applicants_remaining"]
    s["employees"] = start_employees + n_hired
    s["company_value"] = start_value + gain
    s["growth_workforce"] = workforce.tolist() if len(workforce) else []
    s["last_round_hires"] = new_hires.tolist() if n_hired else []
    s["last_round_applicants"] = applicants.tolist()
    s["last_round_hired_count"] = n_hired
    s["last_efficiency"] = eff
    s["last_culture"] = cul
    complete = s["applicants_remaining"] <= 0
    yield repaired_growth_html(s, f"Hiring round complete — {n_hired} applicants hired", batch_n, n_hired, gain, complete), s, scoreboard_html(s), gr.Button(visible=not complete), gr.Button(visible=complete)


def go_sampling():
    return gr.Walkthrough(selected=7)

# ============================================================
# Mission 7 — interstate applicants / sampling bias
# ============================================================

def current_repaired_training_set(state):
    """Four-feature training set/labels used by the repaired local AI."""
    X4, y, uses_guild = repaired_training_data(state)
    return np.asarray(X4, dtype=int), np.asarray(y, dtype=int), bool(uses_guild)


def support_adjusted_scores(model, X4, uses_guild, training_X4, support_lambda=INTERSTATE_SUPPORT_LAMBDA):
    """Prediction score discounted when a profile is far from training support.

    This models a common practical consequence of sampling bias: even a model that
    performs well on familiar local profiles can be much less confident on a region
    of the input space it has barely seen.
    """
    X4 = np.asarray(X4, dtype=int)
    training_X4 = np.asarray(training_X4, dtype=int)
    guild = np.zeros(len(X4), dtype=int)
    Xpred = augment_features(X4, guild) if uses_guild else X4
    base = model.predict_proba(Xpred)[:, 1]
    # Manhattan distance is easy to interpret for 1–5 star attributes.
    distance = np.abs(X4[:, None, :] - training_X4[None, :, :]).sum(axis=2).min(axis=1)
    support = np.exp(-float(support_lambda) * distance)
    return base * support, base, distance


def interstate_waiting_html():
    return """
    <div class='interstate-warning-card'>
      <div class='round-comparison-kicker'>NO NEW LOCAL APPLICANTS</div>
      <h1>We aren’t getting new applicants yet…</h1>
      <p>It seems we have exhausted the local pool of zylometrists. Let’s expand our hiring range to recruit interstate applicants!</p>
    </div>
    """


def interstate_intro_html():
    return f"""
    <div class='interstate-intro-card'>
      <div class='round-comparison-kicker'>{INTERSTATE_POOL_SIZE} NEW APPLICANTS FOUND</div>
      <h1>Our HR team found another {INTERSTATE_POOL_SIZE} candidates in the surrounding states!</h1>
      <p><b>Great news:</b> these applicants are, on average, more formally qualified than our local applicants.</p>
      <p>Many have completed a <b>Master’s degree in Zylometry</b>. They have less on-the-job experience, but are highly specialised and qualified.</p>
    </div>
    """


def go_hire_more_widely(state):
    s = dict(state or initial_state())
    s['interstate_stage'] = 'waiting'
    s['interstate_selected'] = [False] * 20
    s['interstate_selection_order'] = []
    s['interstate_hired'] = 0
    s['interstate_screened'] = 0
    s['interstate_round'] = 0
    s['interstate_rejected'] = []
    s['last_round_hires'] = []
    s['last_round_hired_count'] = 0
    return gr.Walkthrough(selected=7), s, interstate_waiting_html(), gr.Group(visible=True), gr.Group(visible=False), scoreboard_html(s)


def open_interstate_portal(state):
    s = dict(state or initial_state())
    s['interstate_stage'] = 'pool'
    s['applicants_remaining'] = INTERSTATE_POOL_SIZE
    s['display_applicants'] = INTERSTATE_POOL_SIZE
    s['interstate_start_value'] = float(s.get('company_value', 0.0))
    s['interstate_screened'] = 0
    s['interstate_hired'] = 0
    s['interstate_round'] = 0
    s['interstate_rejected'] = []
    s['last_round_hires'] = []
    s['last_round_applicants'] = []
    s['last_round_hired_count'] = 0

    # The old AI still ranks the unfamiliar applicants.  To make the lesson
    # visible without an absolute 0/100 outcome, retain only the two people it
    # scores highest across the whole new pool.
    model, _, uses_guild = get_repaired_model_and_threshold(s)
    train_X4, _, _ = current_repaired_training_set(s)
    adjusted, _, _ = support_adjusted_scores(model, INTERSTATE_POOL, uses_guild, train_X4)
    s['interstate_initial_hire_indices'] = np.argsort(adjusted)[-2:].astype(int).tolist()

    return s, gr.Group(visible=False), gr.Group(visible=True), interstate_intro_html() + growth_screen_html(s, 'Applicant pool ready'), scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)


def interstate_rejected_cards_html(state):
    s = state or initial_state()
    X = np.asarray(s.get('interstate_rejected') or [], dtype=int)
    cards = []
    for i, row in enumerate(X):
        name = INTERSTATE_NAMES[i % len(INTERSTATE_NAMES)]
        initials = ''.join(part[0] for part in name.split()[:2]).upper()
        cards.append(f"""
        <div class='diagnosis-candidate-card interstate-reject-card'>
          <div class='candidate-head'><div class='abstract-avatar avatar-{i%8}'>{html.escape(initials)}</div><div><div class='candidate-name'>{html.escape(name)}</div></div></div>
          <div class='trait-row'><span>Experience</span><span class='rating'>{rating_html(row[0])}</span></div>
          <div class='trait-row'><span>Qualification</span><span class='rating'>{rating_html(row[1])}</span></div>
          <div class='trait-row'><span>Work ethic</span><span class='rating'>{rating_html(row[2])}</span></div>
          <div class='trait-row'><span>Teamwork</span><span class='rating'>{rating_html(row[3])}</span></div>
          <div class='diagnosis-ai-decision rejected-only'><strong>REJECTED</strong></div>
        </div>""")
    return ''.join(cards)


def interstate_diagnosis_html(state):
    s = state or initial_state()
    hired = int(s.get('interstate_hired', 0))
    rejected = np.asarray(s.get('interstate_rejected') or [], dtype=int)
    existing = np.asarray(s.get('growth_workforce') or [], dtype=int)
    if len(rejected):
        hypothetical = np.vstack([existing, rejected]) if len(existing) else rejected.copy()
        heff, hcul = workplace_metrics(hypothetical)
        added_value = deployment_value_gain(heff, hcul, len(rejected))
        current_value = max(0.01, float(s.get('company_value', 0.0)))
        added_pct = 100.0 * added_value / current_value
        value_line = f"<div class='diagnosis-value-callout'>These {len(rejected)} strong rejected candidates would have added an estimated <b>+${added_value:.2f}M</b> to company value (<b>+{added_pct:.1f}%</b>).</div>"
    else:
        value_line = ""
    return f"""
    <div class='diagnosis-screen-shell'>
      <div class='discussion-pause-banner'><div class='discussion-pause-kicker'>CLASSROOM PAUSE</div><h1>Why was the hiring rate so low?</h1><p>Your AI hired only <b>{hired} of {INTERSTATE_POOL_SIZE}</b> applicants.</p>{value_line}</div>
      <div class='diagnosis-full-card-grid'>{interstate_rejected_cards_html(s)}</div>
      <div class='discussion-card large'><div class='discussion-icon'>?</div><div><small>DISCUSS</small><h2>What is different about these applicants compared with the people your AI learned from?</h2></div></div>
    </div>
    """


def deploy_interstate_pool(state):
    """Screen the wider-market pool in two 50-person rounds."""
    s = dict(state or initial_state())
    model, threshold, uses_guild = get_repaired_model_and_threshold(s)
    train_X4, _, _ = current_repaired_training_set(s)

    start = int(s.get('interstate_screened', 0))
    end = min(start + 50, INTERSTATE_POOL_SIZE)
    applicants = INTERSTATE_POOL[start:end].copy()
    if len(applicants) == 0:
        yield interstate_intro_html() + growth_screen_html(s, 'All applicants screened'), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)
        return

    adjusted, _, _ = support_adjusted_scores(model, applicants, uses_guild, train_X4)
    chosen_global = set(int(i) for i in s.get('interstate_initial_hire_indices', []))
    global_idx = np.arange(start, end)
    hired_mask = np.array([int(i) in chosen_global for i in global_idx], dtype=bool)
    n_hired = int(hired_mask.sum())
    new_hires = applicants[hired_mask]

    # Keep the strongest rejected people seen across both rounds for diagnosis.
    prior_rejected = np.asarray(s.get('interstate_rejected') or [], dtype=int)
    rejected = applicants[~hired_mask]
    all_rejected = np.vstack([prior_rejected, rejected]) if len(prior_rejected) and len(rejected) else (rejected.copy() if len(rejected) else prior_rejected.copy())
    if len(all_rejected):
        order = np.argsort(agreed_score(all_rejected))[::-1]
        s['interstate_rejected'] = all_rejected[order[:5]].tolist()

    existing = np.asarray(s.get('growth_workforce') or [], dtype=int)
    workforce = np.vstack([existing, new_hires]) if len(new_hires) and len(existing) else (new_hires.copy() if len(new_hires) else existing.copy())
    eff, cul = workplace_metrics(workforce) if len(workforce) else (0.0, 0.0)
    gain = deployment_value_gain(eff, cul, n_hired)
    start_value = float(s.get('company_value', 0.0))
    start_emp = int(s.get('employees', 0))
    batch_size = len(applicants)
    frames = 7
    statuses = [
        'Opening applications…', 'AI scoring candidates…', 'Comparing with the training data…',
        'Checking candidate profiles…', 'Applying the hiring threshold…', 'Updating the company…', 'Hiring round complete'
    ]
    for f in range(1, frames + 1):
        frac = f / frames
        temp = dict(s)
        temp['display_applicants'] = max(0, INTERSTATE_POOL_SIZE - int(round(start + batch_size * frac)))
        temp['applicants_remaining'] = temp['display_applicants']
        temp['employees'] = int(round(start_emp + n_hired * frac))
        temp['company_value'] = start_value + gain * frac
        temp['last_efficiency'] = eff
        temp['last_culture'] = cul
        yield interstate_intro_html() + growth_screen_html(temp, statuses[f-1], int(round(batch_size*frac)), int(round(n_hired*frac)), gain*frac), s, scoreboard_html(temp), gr.Button(visible=False), gr.Button(visible=False)
        time.sleep(INTERSTATE_ANIMATION_SECONDS / frames)

    s['interstate_round'] = int(s.get('interstate_round', 0)) + 1
    s['interstate_screened'] = end
    s['interstate_hired'] = int(s.get('interstate_hired', 0)) + n_hired
    s['applicants_screened'] = int(s.get('applicants_screened', 0)) + batch_size
    s['applicants_remaining'] = INTERSTATE_POOL_SIZE - end
    s['display_applicants'] = INTERSTATE_POOL_SIZE - end
    s['employees'] = start_emp + n_hired
    s['company_value'] = start_value + gain
    s['growth_workforce'] = workforce.tolist() if len(workforce) else []
    s['last_round_hires'] = new_hires.tolist() if len(new_hires) else []
    s['last_round_applicants'] = applicants.tolist()
    s['last_round_hired_count'] = n_hired
    s['last_efficiency'] = eff
    s['last_culture'] = cul

    if end < INTERSTATE_POOL_SIZE:
        if n_hired == 0:
            note = "<div class='round-comparison-card'><div class='round-comparison-kicker'>HMMM…</div><h2>You didn’t hire anyone in this round.</h2><p>That’s strange. Run the next 50 applicants and see whether the same thing happens again.</p></div>"
        else:
            note = f"<div class='round-comparison-card'><div class='round-comparison-kicker'>HMMM…</div><h2>You hired only {n_hired} of 50 applicants.</h2><p>That seems unusually low. Run the next 50 and see whether the pattern continues.</p></div>"
        result = interstate_intro_html() + growth_screen_html(s, '50 applicants screened', 50, n_hired, gain) + note
        yield result, s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
    else:
        total_hired = int(s.get('interstate_hired', 0))
        rate = 100 * total_hired / INTERSTATE_POOL_SIZE
        if total_hired == 0:
            headline = "No one was hired. These applicants were highly qualified — let’s inspect what happened."
        else:
            headline = f"Only {total_hired} of {INTERSTATE_POOL_SIZE} applicants were hired. Let’s inspect what happened."
        result = interstate_intro_html() + growth_screen_html(s, 'All applicants screened', 50, n_hired, gain) + f"""
        <div class='round-comparison-card crisis-summary-card interstate-result-card'><div class='round-comparison-kicker'>HIRING RESULT</div><h2>{headline}</h2><p>Overall hire rate: <b>{rate:.1f}%</b>.</p></div>"""
        yield result, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)


def show_interstate_diagnosis(state):
    s = dict(state or initial_state())
    return gr.Group(visible=False), gr.Group(visible=True), interstate_diagnosis_html(s)


def go_fix_interstate(state):
    s = dict(state or initial_state())
    s['interstate_selected'] = [False] * 20
    s['interstate_selection_order'] = []
    s['interstate_fix_trained'] = False
    s['interstate_fix_screened'] = 0
    s['interstate_fix_hired'] = 0
    return gr.Walkthrough(selected=8), s, make_all_card_data(INTERSTATE_NAMES, INTERSTATE_LABEL_BATCH, [False] * 20), selection_status_html([False] * 20, label='NEW HIRE EXAMPLES'), gr.Button(interactive=False)


def augmented_interstate_model(state, selected_mask):
    s = state or initial_state()
    base_X4, base_y, uses_guild = current_repaired_training_set(s)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    extra = INTERSTATE_LABEL_BATCH[selected_mask]
    X4 = np.vstack([base_X4, extra]) if len(extra) else base_X4.copy()
    y = np.concatenate([base_y, np.ones(len(extra), dtype=int)]) if len(extra) else base_y.copy()
    Xtrain = augment_features(X4, np.zeros(len(X4), dtype=int)) if uses_guild else X4
    model = train_student_model(Xtrain, y)
    threshold = calibrate_repaired_threshold(model, uses_guild, s)
    return model, threshold, uses_guild, X4


def toggle_interstate_training_candidate(state, evt: gr.EventData):
    s = dict(state or initial_state())
    selected = list(s.get('interstate_selected', [False] * 20))
    idx = int(getattr(evt, 'index', -1))
    if 0 <= idx < len(selected):
        if selected[idx]:
            selected[idx] = False
        elif sum(selected) < 5:
            selected[idx] = True
    s['interstate_selected'] = selected
    n = int(sum(selected))
    return (
        make_all_card_data(INTERSTATE_NAMES, INTERSTATE_LABEL_BATCH, selected),
        s,
        selection_status_html(selected, label='NEW HIRE EXAMPLES'),
        gr.Button(interactive=(n == 5)),
    )


async def train_interstate_fix_ai(state):
    s = dict(state or initial_state())
    selected = np.asarray(s.get('interstate_selected', [False] * 20), dtype=bool)
    if selected.sum() != 5:
        yield "<div class='warning-card'>Choose exactly five applicants to label Hire first.</div>", s, gr.Button(visible=True), gr.Button(visible=False)
        return

    messages = [
        'Adding your five new Hire labels to the training data…',
        'Updating the AI with examples of this applicant profile…',
        'Learning the new patterns…',
        'Checking the updated model…',
    ]
    for i, msg in enumerate(messages):
        yield training_html(int((i + 1) / len(messages) * 92), 'RETRAINING THE AI', msg), s, gr.Button(visible=False), gr.Button(visible=False)
        await asyncio.sleep(1.4)

    s['interstate_fix_trained'] = True
    done = training_html(100, 'UPDATED AI READY', 'Your five new labelled examples are now part of the training data.', done=True)
    yield done, s, gr.Button(visible=False), gr.Button(visible=True)


def start_interstate_fix_redeploy(state):
    s = dict(state or initial_state())
    if not s.get('interstate_fix_trained'):
        return s, gr.Group(visible=True), gr.Group(visible=False), "<div class='warning-card'>Retrain the AI first.</div>", scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
    s['interstate_fix_screened'] = 0
    s['interstate_fix_hired'] = 0
    s['interstate_fix_round'] = 0
    s['interstate_fix_start_value'] = float(s.get('company_value', 0.0))
    s['applicants_remaining'] = INTERSTATE_POOL_SIZE
    s['display_applicants'] = INTERSTATE_POOL_SIZE
    s['last_round_hires'] = []
    s['last_round_applicants'] = []
    s['last_round_hired_count'] = 0

    # Recalibrate the updated AI on the new 100-person pool.  The local-model
    # threshold is retained unless it would still be excessively strict after
    # adding the five new positive examples.
    selected = np.asarray(s.get('interstate_selected', [False] * 20), dtype=bool)
    model, base_threshold, uses_guild, X4 = augmented_interstate_model(s, selected)
    adjusted, _, _ = support_adjusted_scores(model, INTERSTATE_TEST_POOL, uses_guild, X4)
    s['interstate_fix_threshold'] = float(min(base_threshold, np.quantile(adjusted, 0.85)))

    screen = interstate_intro_html() + growth_screen_html(s, 'Updated AI ready to re-test')
    return s, gr.Group(visible=False), gr.Group(visible=True), screen, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)


def deploy_interstate_fix_round(state):
    s = dict(state or initial_state())
    selected = np.asarray(s.get('interstate_selected', [False] * 20), dtype=bool)
    if selected.sum() != 5 or not s.get('interstate_fix_trained'):
        yield "<div class='warning-card'>Label five applicants and retrain the AI first.</div>", s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
        return

    start = int(s.get('interstate_fix_screened', 0))
    end = min(start + 50, INTERSTATE_POOL_SIZE)
    applicants = INTERSTATE_TEST_POOL[start:end].copy()
    if len(applicants) == 0:
        yield growth_screen_html(s, 'All applicants screened'), s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)
        return

    model, base_threshold, uses_guild, X4 = augmented_interstate_model(s, selected)
    threshold = float(s.get('interstate_fix_threshold', base_threshold))
    adjusted, _, _ = support_adjusted_scores(model, applicants, uses_guild, X4)
    hired_mask = adjusted >= threshold
    n_hired = int(hired_mask.sum())
    new_hires = applicants[hired_mask]

    existing = np.asarray(s.get('growth_workforce') or [], dtype=int)
    if len(new_hires):
        workforce = np.vstack([existing, new_hires]) if len(existing) else new_hires.copy()
    else:
        workforce = existing.copy()
    eff, cul = workplace_metrics(workforce) if len(workforce) else (0.0, 0.0)
    gain = deployment_value_gain(eff, cul, n_hired)
    start_value = float(s.get('company_value', 0.0))
    start_emp = int(s.get('employees', 0))
    frames = 7
    statuses = ['Opening applications…', 'Updated AI scoring candidates…', 'Comparing with the expanded training data…', 'Checking candidate profiles…', 'Preparing offers…', 'Onboarding hires…', 'Hiring round complete']
    for f in range(1, frames + 1):
        frac = f / frames
        temp = dict(s)
        temp['display_applicants'] = max(0, INTERSTATE_POOL_SIZE - int(round(start + len(applicants) * frac)))
        temp['applicants_remaining'] = temp['display_applicants']
        temp['employees'] = int(round(start_emp + n_hired * frac))
        temp['company_value'] = start_value + gain * frac
        temp['last_efficiency'] = eff
        temp['last_culture'] = cul
        yield interstate_intro_html() + growth_screen_html(temp, statuses[f-1], int(round(len(applicants) * frac)), int(round(n_hired * frac)), gain * frac), s, scoreboard_html(temp), gr.Button(visible=False), gr.Button(visible=False)
        time.sleep(INTERSTATE_ANIMATION_SECONDS / frames)

    s['interstate_fix_screened'] = end
    s['interstate_fix_hired'] = int(s.get('interstate_fix_hired', 0)) + n_hired
    s['interstate_fix_round'] = int(s.get('interstate_fix_round', 0)) + 1
    s['applicants_screened'] = int(s.get('applicants_screened', 0)) + len(applicants)
    s['applicants_remaining'] = INTERSTATE_POOL_SIZE - end
    s['display_applicants'] = INTERSTATE_POOL_SIZE - end
    s['employees'] = start_emp + n_hired
    s['company_value'] = start_value + gain
    s['growth_workforce'] = workforce.tolist() if len(workforce) else []
    s['last_efficiency'] = eff
    s['last_culture'] = cul
    s['last_round_hires'] = new_hires.tolist() if len(new_hires) else []
    s['last_round_applicants'] = applicants.tolist()
    s['last_round_hired_count'] = n_hired

    if end < INTERSTATE_POOL_SIZE:
        result = interstate_intro_html() + growth_screen_html(s, f'Hiring round complete — {n_hired} applicants hired', 50, n_hired, gain)
        yield result, s, scoreboard_html(s), gr.Button(visible=True), gr.Button(visible=False)
    else:
        total = int(s.get('interstate_fix_hired', 0))
        total_gain = float(s.get('company_value', 0.0)) - float(s.get('interstate_fix_start_value') or 0.0)
        result = interstate_intro_html() + growth_screen_html(s, 'All applicants screened', 50, n_hired, gain) + f"""
        <div class='repair-success-card'><div class='round-comparison-kicker'>UPDATED HIRING COMPLETE</div>
        <h2>Well done, your updated AI hired <b>{total} of {INTERSTATE_POOL_SIZE}</b> applicants.</h2>
        <p>Company value grew by <b>+${total_gain:.2f}M</b>.</p></div>"""
        s['final_interstate_hires'] = total
        s['final_redeploy_gain'] = total_gain
        yield result, s, scoreboard_html(s), gr.Button(visible=False), gr.Button(visible=True)


def labelling_priorities_html(state):
    s = state or initial_state()
    founding_mask = np.asarray(s.get('initial_hires') or [False] * 20, dtype=bool)
    new_mask = np.asarray(s.get('interstate_selected') or [False] * 20, dtype=bool)

    def one_profile(title, X):
        if len(X) == 0:
            return ''
        m = np.asarray(X, dtype=float).mean(axis=0)
        labels = ['Experience', 'Qualification', 'Work ethic', 'Teamwork']
        strongest = labels[int(np.argmax(m))]
        items = ''.join(f"<div><span>{lab}</span><b>{val:.1f}/5</b></div>" for lab, val in zip(labels, m))
        return f"<div class='final-label-profile'><h3>{title}</h3><p>Highest average: <b>{strongest}</b></p><div class='final-label-grid'>{items}</div></div>"

    founding = LEVEL1_VALUES[founding_mask] if founding_mask.sum() else np.empty((0, 4), dtype=int)
    newer = INTERSTATE_LABEL_BATCH[new_mask] if new_mask.sum() else np.empty((0, 4), dtype=int)
    return f"""<div class='final-priorities'><div class='round-comparison-kicker'>WHAT DID YOUR LABELS PRIORITISE?</div><h2>Compare the profiles you chose to label Hire.</h2><div class='final-priority-grid'>{one_profile('Founding Hire labels', founding)}{one_profile('New applicant Hire labels', newer)}</div></div>"""


def final_report(state):
    s = state or initial_state()
    value = float(s.get('company_value', STARTING_COMPANY_VALUE))
    employees = int(s.get('employees', 0))
    hired_workers = max(0, employees - 5)
    eff = s.get('last_efficiency')
    cul = s.get('last_culture')
    screened = int(s.get('applicants_screened', 0))
    return f"""<div class='final-screen'>
    <div class='final-hero'><div class='final-trophy'>★</div><small>SIMULATION COMPLETE</small><h1>Final company valuation and stats</h1>
    <p>Compare your results with the people around you.</p></div>
    <div class='final-stats five'>
      <div><span>Company value</span><b>${value:.2f}M</b></div>
      <div><span>Workers hired</span><b>{hired_workers}</b></div>
      <div><span>Efficiency</span><b>{fmt_metric(eff)}/100</b></div>
      <div><span>Culture</span><b>{fmt_metric(cul)}/100</b></div>
      <div><span>Applicants screened</span><b>{screened:,}</b></div>
    </div>
    {labelling_priorities_html(s)}
    <div class='takeaway'><h2>Compare with the class</h2><p>Who built the highest-value company? Who hired the most people? How do your Efficiency and Culture scores compare? Did different early hiring decisions lead to different final companies?</p></div>
    <p class='muted'>Company value and workplace scores are fictional teaching mechanics used to make the consequences of hiring decisions visible.</p>
    </div>"""

def finish_simulation(state):
    return final_report(state), gr.Walkthrough(selected=9)


# ============================================================
# Mission 6 — sampling / representation
# ============================================================

def sampling_experiment(graduate_mix, state):
    s = dict(state or initial_state())
    graduate_mix = int(graduate_mix)
    total_training = 60
    repeats = 12

    trad_test = generate_traditional(np.random.default_rng(1333), 1000)
    grad_test = generate_graduates(np.random.default_rng(1334), 1000)
    y_trad = agreed_label(trad_test)
    y_grad = agreed_label(grad_test)
    trad_accs, grad_accs = [], []
    representative_model = None

    for rep in range(repeats):
        rng = np.random.default_rng(1000 + rep * 101 + graduate_mix)
        n_grad = max(1, int(round(total_training * graduate_mix / 100)))
        n_trad = total_training - n_grad
        X_train = np.vstack([generate_traditional(rng, n_trad), generate_graduates(rng, n_grad)])
        y_train = agreed_label(X_train)
        if len(np.unique(y_train)) < 2:
            continue
        model = train_sensitive_model(X_train, y_train)
        trad_accs.append(100 * (model.predict(trad_test) == y_trad).mean())
        grad_accs.append(100 * (model.predict(grad_test) == y_grad).mean())
        if representative_model is None:
            representative_model = model

    trad_acc = float(np.mean(trad_accs))
    grad_acc = float(np.mean(grad_accs))
    deploy_pool, _ = mixed_population(seed=7123, n=200, graduate_fraction=0.5)
    hired = shortlist(representative_model, deploy_pool, 50)
    workforce = deploy_pool[hired]
    eff, cul = workplace_metrics(workforce)

    s["sampling_mix"] = graduate_mix
    # This is a controlled teaching experiment, not another company deployment.
    # Keep the persistent scoreboard tied to the actual company built above.
    s["sampling_bias_unlocked"] = True

    n_grad = int(round(60 * graduate_mix / 100))
    n_trad = 60 - n_grad
    allocation = f"""<div class='allocation-card'><div class='panel-kicker'>YOUR 60-LABEL BUDGET</div>
    <div class='allocation-bar'><i class='trad' style='width:{100-graduate_mix}%'></i><i class='grad' style='width:{graduate_mix}%'></i></div>
    <div class='allocation-labels'><span><b>{n_trad}</b> traditional</span><span><b>{n_grad}</b> graduates</span></div></div>"""
    accuracy = f"""<div class='accuracy-grid'><div><small>ACCURACY · TRADITIONAL</small><b>{trad_acc:.1f}%</b><div class='accuracy-track'><i style='width:{trad_acc}%'></i></div></div>
    <div><small>ACCURACY · NEW GRADUATES</small><b>{grad_acc:.1f}%</b><div class='accuracy-track alt'><i style='width:{grad_acc}%'></i></div></div></div>"""
    result = allocation + accuracy + results_cards_html(eff, cul, "One deployment") + concept_unlock_html("SAMPLING BIAS", "A model may work much better for the kinds of examples that are well represented in its training data, even when every label follows the same agreed rule.", "◫") + bias_badges_html(s)
    return result, s, scoreboard_html(s)


def legacy_final_report(state):
    s = state or initial_state()
    value = float(s.get("company_value", STARTING_COMPANY_VALUE))
    employees = int(s.get("employees", 0))
    eff = s.get("last_efficiency")
    cul = s.get("last_culture")
    screened = int(s.get("applicants_screened", 0))
    return f"""<div class='final-screen'>
    <div class='final-hero'><div class='final-trophy'>🏁</div><small>SIMULATION COMPLETE</small><h1>Well done!</h1>
    <p>You built a company, trained a hiring AI, watched the profession change, and then adapted the system.</p></div>
    <div class='final-summary-sentence'>After screening <b>{screened:,} applicants</b>, your company finished with:</div>
    <div class='final-stats five'>
      <div><span>💰 Company value</span><b>${value:.2f}M</b></div>
      <div><span>⚙ Efficiency</span><b>{fmt_metric(eff)}/100</b></div>
      <div><span>🙂 Culture</span><b>{fmt_metric(cul)}/100</b></div>
      <div><span>👥 Employees</span><b>{employees}</b></div>
      <div><span>🔎 Applicants screened</span><b>{screened:,}</b></div>
    </div>
    {bias_badges_html(s)}
    <div class='takeaway'><h2>Three questions to take back to a real AI system</h2><p><b>Who decided the labels?</b><br><b>What past does the data reflect?</b><br><b>Who or what is represented in the data?</b></p></div>
    <p class='muted'>Company value, workplace scores and the Zylometry scenario are fictional teaching mechanics. “Applicants screened” counts AI deployment pools, not the small training batches you labelled manually.</p>
    </div>"""


def legacy_finish_simulation(state):
    return final_report(state), gr.Walkthrough(selected=8)




def briefing_intro_html():
    return f"""
    <section class="briefing-screen briefing-page-one">
      <div class="briefing-glow glow-one"></div>
      <div class="briefing-glow glow-two"></div>

      <div class="briefing-brand">QUT001 · ZYLOMETRY LAB</div>
      <div class="briefing-kicker">MISSION BRIEFING</div>
      <h1>Build. Train. Grow.</h1>

      <div class="briefing-story">
        <p><strong>As a keen enthusiast of Zylometry, you've noticed increasing demand from the market.</strong>
        You've decided now is the perfect time to launch a new Zylometry company.</p>
        <p><strong>As a very busy founder, your least favourite job is reading long résumés.</strong>
        You want to use AI to help build the company by hiring the best zylometrists in the field.</p>
      </div>

      <div class="briefing-cards">
        <div class="briefing-card hire-card">
          <div class="briefing-card-icon svg-mode">{icon_svg('hire')}</div>
          <div class="briefing-card-step">STEP 1</div>
          <h2>HIRE</h2>
          <p>Choose the people you think will make the strongest founding team.</p>
        </div>
        <div class="briefing-card train-card">
          <div class="briefing-card-icon svg-mode">{icon_svg('train')}</div>
          <div class="briefing-card-step">STEP 2</div>
          <h2>TRAIN</h2>
          <p>Turn your hiring decisions into labels and teach an AI what a good zylometrist looks like.</p>
        </div>
        <div class="briefing-card grow-card">
          <div class="briefing-card-icon svg-mode">{icon_svg('grow')}</div>
          <div class="briefing-card-step">STEP 3</div>
          <h2>GROW</h2>
          <p>Deploy your AI, hire at scale, and build the most valuable company you can.</p>
        </div>
      </div>

      <div class="briefing-page-hint">Next: learn how your company dashboard works</div>
    </section>
    """


def briefing_dashboard_html():
    return f"""
    <section class="briefing-screen briefing-page-two">
      <div class="briefing-glow glow-one"></div>
      <div class="briefing-glow glow-two"></div>

      <div class="briefing-brand">QUT001 · ZYLOMETRY LAB</div>
      <div class="briefing-kicker">MISSION BRIEFING</div>
      <h1>Watch Your Dashboard</h1>

      <div class="briefing-explainer">
        <div class="briefing-explainer-head">
          <div class="briefing-section-title">WATCH THE DASHBOARD</div>
          <p>Your hiring decisions will affect your company's <strong>Efficiency</strong> and <strong>Culture</strong>.
          Both are important for gaining <strong>Company Value</strong>. As you deploy your AI, the <strong>Applicants</strong> counter shows how much talent remains in the current market.</p>
        </div>
        {briefing_score_preview()}
      </div>

      <div class="briefing-goal briefing-goal-hero">
        <div class="briefing-goal-icon svg-mode">{icon_svg('goal', color='#ffffff')}</div>
        <div>
          <div class="briefing-goal-label">YOUR GOAL</div>
          <h2>BUILD THE MOST VALUABLE COMPANY YOU CAN</h2>
          <p><strong>Grow company value as quickly as possible.</strong> There is no perfect hiring formula — choose who you think the best zylometrists for the company are, then see what your decisions teach the AI.</p>
        </div>
      </div>
    </section>
    """


def show_briefing_page_two(state):
    s = state or initial_state()
    return gr.update(visible=False), gr.update(visible=True), gr.HTML(value=scoreboard_html(s), visible=True)


def show_briefing_page_one():
    return gr.update(visible=True), gr.update(visible=False), gr.HTML(visible=False)



def start_company(state):
    s = dict(state or initial_state())
    return gr.Walkthrough(selected=1), s, gr.HTML(value=scoreboard_html(s), visible=True)


# ============================================================
# CSS
# ============================================================

CSS = r"""
:root {
  --ink:#17223b; --muted:#667085; --line:#dce4f0; --panel:#ffffff;
  --blue:#2d7de9; --navy:#163a70; --purple:#6d5dfc; --teal:#16a394;
  --green:#1a9b62; --orange:#f39a2c; --red:#d9485f; --soft:#f5f8fd;
}
.gradio-container {max-width: 1320px !important; background:linear-gradient(180deg,#f5f8fd 0,#eef3fa 100%);}
.briefing-screen{position:relative;overflow:hidden;max-width:1120px;margin:20px auto 26px;padding:48px 54px 42px;border-radius:30px;background:linear-gradient(145deg,#12284a 0%,#183d70 53%,#24568b 100%);color:white;box-shadow:0 24px 60px rgba(24,54,96,.24);text-align:center;border:1px solid rgba(255,255,255,.12)}
.briefing-screen>*{position:relative;z-index:2}.briefing-glow{position:absolute!important;z-index:1!important;border-radius:50%;filter:blur(4px);opacity:.32}.glow-one{width:330px;height:330px;right:-110px;top:-150px;background:#6d5dfc}.glow-two{width:280px;height:280px;left:-120px;bottom:-160px;background:#16a394}
.briefing-brand{font-size:13px;font-weight:900;letter-spacing:.18em;color:#d7e8ff;margin-bottom:22px}.briefing-kicker{display:inline-block;padding:7px 12px;border-radius:99px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.15);font-size:10px;font-weight:900;letter-spacing:.16em;color:#f2f8ff}.briefing-screen h1{font-size:48px!important;line-height:1;margin:12px 0 24px!important;color:#fff}.briefing-story{max-width:850px;margin:0 auto 28px;font-size:18px;line-height:1.6;color:#f4f8ff}.briefing-story p{margin:0 0 13px;color:#f4f8ff}.briefing-story strong,.briefing-goal strong,.briefing-explainer strong{color:#fff}
.briefing-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0 22px;text-align:left}.briefing-card{min-height:210px;padding:22px;border-radius:20px;background:rgba(255,255,255,.97);color:#233754;border:1px solid rgba(255,255,255,.55);box-shadow:0 10px 28px rgba(7,27,55,.16)}.briefing-card-icon{width:48px;height:48px;margin-bottom:17px;color:inherit}.briefing-card-icon svg{width:48px;height:48px;display:block}.briefing-card-icon.svg-mode{color:currentColor}.briefing-card-step{font-size:9px;font-weight:900;letter-spacing:.14em;color:#8190a4}.briefing-card h2{font-size:24px!important;margin:4px 0 8px!important;color:#17223b}.briefing-card p{font-size:13px;line-height:1.45;margin:0;color:#5e6d83}.hire-card{border-top:5px solid #16a394;color:#138577}.train-card{border-top:5px solid #6d5dfc;color:#5d4df0}.grow-card{border-top:5px solid #f39a2c;color:#d9861d}
.briefing-explainer{max-width:940px;margin:4px auto 0;text-align:left;padding:22px 24px;border-radius:20px;background:rgba(9,27,52,.36);border:1px solid rgba(255,255,255,.14)}.briefing-explainer-head p{font-size:15px;line-height:1.55;margin:6px 0 0;color:#f1f6ff}.score-preview-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:16px}.score-preview-tile{display:flex;gap:12px;align-items:flex-start;padding:14px;border-radius:16px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.14)}.score-preview-icon{width:44px;height:44px;flex:0 0 auto;border-radius:14px;background:rgba(255,255,255,.12);display:grid;place-items:center;color:#fff}.score-preview-icon svg{width:26px;height:26px}.score-preview-label{font-size:11px;font-weight:900;letter-spacing:.14em;color:#d8e9ff}.score-preview-short{font-size:15px;font-weight:800;color:#fff;margin:3px 0}.score-preview-detail{font-size:12px;line-height:1.45;color:#dbe9fb}
.briefing-goal{max-width:940px;margin:24px auto 0;display:flex;gap:16px;text-align:left;align-items:flex-start;padding:18px 20px;border-radius:17px;background:rgba(5,22,43,.28);border:1px solid rgba(255,255,255,.13)}.briefing-goal-icon{width:44px;height:44px;color:#fff}.briefing-goal-icon svg{width:44px;height:44px}.briefing-goal-label{font-size:10px;font-weight:900;letter-spacing:.14em;color:#bce2ff;margin-bottom:4px}.briefing-goal p{font-size:15px;line-height:1.5;margin:0;color:#f4f8ff}
#start-company-btn button{min-height:58px!important;font-size:17px!important;font-weight:900!important;border-radius:16px!important;box-shadow:0 10px 25px rgba(45,125,233,.2)!important}
@media (max-width: 900px){.briefing-cards{grid-template-columns:1fr}.score-preview-grid{grid-template-columns:1fr}}
footer {display:none !important;}
#global-scoreboard {position:sticky; top:0; z-index:50; background:rgba(245,248,253,.96); backdrop-filter:blur(12px); padding:10px 0 8px;}
.scoreboard-shell{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 10px;}
.score-tile{min-height:104px;background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px 18px;display:flex;gap:14px;align-items:center;box-shadow:0 8px 22px rgba(27,50,93,.08);overflow:hidden;position:relative;}
.score-tile:after{content:'';position:absolute;inset:auto -30px -50px auto;width:100px;height:100px;border-radius:50%;background:currentColor;opacity:.07;}
.score-icon{width:48px;height:48px;border-radius:15px;display:grid;place-items:center;background:#edf4ff;color:var(--blue);flex:0 0 auto;}.score-icon svg{width:26px;height:26px;display:block}
.value-tile .score-icon{background:#e9fbf4;color:var(--green)} .applicant-tile .score-icon{background:#fff4e7;color:var(--orange)}
.score-label{font-size:12px;font-weight:900;letter-spacing:.075em;color:#667085}.score-number{font-size:31px;line-height:1;font-weight:900;color:var(--ink);margin:5px 0}.score-denom{font-size:14px;color:#8792a6}.score-sub{font-size:12px;color:var(--green);font-weight:700}
.mini-track{height:6px;border-radius:99px;background:#e8edf5;overflow:hidden;width:150px;max-width:100%}.mini-fill{height:100%;border-radius:99px;background:var(--blue);animation:barIn .55s ease-out}.score-tile.good .mini-fill{background:var(--green)}.score-tile.fair .mini-fill{background:#a7bf42}.score-tile.mid .mini-fill{background:#e6c84a}.score-tile.low .mini-fill{background:var(--orange)}.score-tile.bad .mini-fill{background:var(--red)}.applicant-tile .mini-fill{background:var(--orange)}
@keyframes barIn{from{width:0}}
.mission-banner{min-height:150px;border-radius:24px;padding:24px 30px;margin:10px 0 20px;display:flex;align-items:center;justify-content:space-between;overflow:hidden;box-shadow:0 12px 32px rgba(34,73,131,.12);border:1px solid #cfe0f6;background:linear-gradient(120deg,#eaf4ff,#f5f2ff);}
.mission-banner.orange{background:linear-gradient(120deg,#fff0e6,#fff8ef);border-color:#ffd4ae}.mission-banner.red{background:linear-gradient(120deg,#7d1d2b,#c4383f);color:white;border-color:#db5660}.mission-banner.red .mission-kicker,.mission-banner.red p{color:#ffd9d9}.mission-banner.purple{background:linear-gradient(120deg,#efeaff,#f8f2ff);border-color:#d9cfff}
.mission-kicker{font-size:12px;font-weight:900;letter-spacing:.14em;color:var(--blue)}.mission-banner h1{font-size:34px!important;line-height:1.05;margin:5px 0 8px!important;color:inherit}.mission-banner p{font-size:17px;margin:0;color:#4e607c;max-width:760px}.banner-art{width:235px;height:110px;flex:0 0 235px}
.main-two-col{align-items:flex-start!important}.candidate-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;}
.candidate-card{background:white;border:2px solid #e2e8f1;border-radius:18px;padding:14px;box-shadow:0 6px 18px rgba(35,60,100,.07);cursor:pointer;transition:transform .15s ease,border-color .15s ease,box-shadow .15s ease;min-height:355px;outline:none}.candidate-card:hover,.candidate-card:focus{transform:translateY(-3px);border-color:#8bb9f4;box-shadow:0 12px 26px rgba(35,60,100,.14)}.candidate-card.selected{border-color:var(--teal);box-shadow:0 0 0 3px rgba(22,163,148,.12),0 12px 26px rgba(22,163,148,.16);background:linear-gradient(180deg,#fff,#f2fffb)}
.candidate-head{display:flex;gap:10px;align-items:center;margin-bottom:14px}.abstract-avatar,.reject-avatar{width:48px;height:48px;border-radius:16px;display:grid;place-items:center;font-weight:900;color:white;font-size:17px;box-shadow:inset 0 -10px 20px rgba(0,0,0,.08)}.avatar-0{background:#6d5dfc}.avatar-1{background:#16a394}.avatar-2{background:#e58c35}.avatar-3{background:#2d7de9}.avatar-4{background:#be5f97}.avatar-5{background:#6f8b3d}.avatar-6{background:#d9485f}.avatar-7{background:#6073a8}
.rating-dots{display:inline-flex!important;gap:2px;white-space:nowrap!important;flex-wrap:nowrap!important}.rating-dots span{display:inline-block!important;white-space:nowrap!important}
.candidate-name{font-size:19px;font-weight:900;color:var(--ink);line-height:1.05}.guild-badge{font-size:10px;font-weight:900;margin-top:6px;border-radius:99px;padding:4px 7px;background:#fff4cf;color:#8a5b00;border:1px solid #efd47a;letter-spacing:.02em;line-height:1.15}.pathway-badge{font-size:10px;font-weight:800;margin-top:5px;border-radius:99px;padding:3px 7px;background:#edf3ff;color:#3c5d99}.trait-row{display:block;border-top:1px solid #edf0f5;padding:10px 0 8px;font-size:13px;font-weight:800;color:#48566c}.trait-row>span:first-child{display:block;margin-bottom:5px}.rating{display:block;font-size:18px;line-height:1;letter-spacing:2px;white-space:nowrap}.dots-on{color:var(--purple)}.dots-off{color:#cbd3df}.hire-pill{margin-top:8px;border-radius:11px;padding:10px;text-align:center;font-weight:900;font-size:13px;color:#637087;background:#eef2f7;border:1px solid #dce3ed}.hire-pill.selected{color:white;background:var(--teal);border-color:var(--teal)}
.selection-status{display:flex;align-items:center;gap:12px;border-radius:13px;padding:11px 15px;background:#eef3fb;color:#516078;margin-bottom:10px}.selection-status strong{font-size:18px;color:var(--ink)}.selection-status.complete{background:#e9fbf4;color:#167b59}.selection-status.complete strong{color:#0b6f4c}.status-kicker{font-size:10px;font-weight:900;letter-spacing:.1em}.page-label{display:block;text-align:center;color:#667085;font-size:12px;padding-top:8px}
.side-panel{background:white;border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:0 8px 24px rgba(34,58,96,.08)}.side-panel-title{display:flex;gap:12px;align-items:center;border-bottom:1px solid #e9edf3;padding-bottom:13px}.side-panel-title>span{font-size:28px}.side-panel-title small,.panel-kicker{font-size:10px;font-weight:900;letter-spacing:.12em;color:#69778e}.side-panel-title h2{margin:2px 0 0!important;font-size:21px!important}.gauge-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0}.gauge-card{border:1px solid #e0e6ef;border-radius:15px;padding:13px;text-align:center;background:#fafcff}.gauge-label{font-size:12px;font-weight:900;color:#5b6980}.gauge-ring{--score:0;width:92px;height:92px;margin:9px auto;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--teal) calc(var(--score)*1%),#e4eaf2 0);position:relative}.gauge-ring:after{content:'';position:absolute;width:70px;height:70px;border-radius:50%;background:white}.gauge-ring span{position:relative;z-index:2;font-size:25px;font-weight:900;color:var(--ink)}.gauge-card.fair .gauge-ring{background:conic-gradient(#a7bf42 calc(var(--score)*1%),#e4eaf2 0)}.gauge-card.mid .gauge-ring{background:conic-gradient(#e6c84a calc(var(--score)*1%),#e4eaf2 0)}.gauge-card.low .gauge-ring{background:conic-gradient(var(--orange) calc(var(--score)*1%),#e4eaf2 0)}.gauge-card.bad .gauge-ring{background:conic-gradient(var(--red) calc(var(--score)*1%),#e4eaf2 0)}.gauge-number{font-size:35px;font-weight:900;margin:16px}.gauge-note{font-size:11px;font-weight:800;color:#69778e}
.profile-panel{border-top:1px solid #e9edf3;padding-top:13px;margin-top:8px}.profile-panel h3{font-size:15px!important;margin:0 0 9px!important}.profile-item{display:grid;grid-template-columns:1fr 52px;gap:4px 8px;align-items:center;font-size:12px;margin:8px 0;color:#536078}.profile-item b{text-align:right;color:#263651}.profile-track{grid-column:1/-1;height:5px;background:#e8edf4;border-radius:99px;overflow:hidden}.profile-track i{display:block;height:100%;background:var(--blue);border-radius:99px}.game-tip{background:#eef5ff;border:1px solid #d7e7fa;color:#47617f;border-radius:12px;padding:11px 12px;font-size:12px;margin-top:14px}
.diagnosis-panel{margin-top:15px;padding:15px;border-radius:16px;background:#111f38;color:#eef5ff;border:1px solid #2f486d;text-align:left}
.diagnosis-title{display:flex;align-items:center;justify-content:space-between;gap:10px;font-family:Arial,Helvetica,sans-serif;font-size:14px;font-weight:900;letter-spacing:.08em;color:#fff}.diagnosis-title span{font-size:10px;letter-spacing:.08em;padding:4px 7px;border-radius:99px;background:#f0a43a;color:#17223b}.diagnosis-panel>p,.diagnosis-explainer{font-size:13px;line-height:1.45;color:#cfdef2;margin:7px 0 0}.diagnosis-section{margin-top:15px;padding-top:13px;border-top:1px solid #314865}.diagnosis-section h4{font-family:Arial,Helvetica,sans-serif;font-size:15px!important;margin:0 0 10px!important;color:#fff}.diagnosis-row{margin:10px 0}.diagnosis-row-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;font-size:13px;color:#eaf2ff}.diagnosis-row-head b{font-weight:800}.diagnosis-row-head span{font-weight:800;color:#9fc7ff;white-space:nowrap}.diagnosis-row small{display:block;font-size:11px;color:#aebfd6;margin:2px 0 5px}.diagnosis-track{height:7px;background:#263953;border-radius:99px;overflow:hidden}.diagnosis-track i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#5c8ff2,#7d6cf6)}.diagnosis-penalty{margin-top:10px;padding:8px 10px;border-radius:10px;background:#47293a;color:#ffd9e1;font-size:12px}
.pathway-split{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:14px 0}.pathway-split div{border-radius:14px;background:#f3f6fb;text-align:center;padding:13px}.pathway-split b{font-size:27px;display:block;color:var(--ink)}.pathway-split span{font-size:11px;color:#68748a}
button.primary{font-weight:900!important}.success-card,.warning-card{border-radius:14px;padding:14px 17px;font-weight:800;margin:10px 0}.success-card{background:#e7f9f1;color:#136d4b;border:1px solid #bfead7}.warning-card{background:#fff4e8;color:#935815;border:1px solid #f3d4ad}
.training-console{display:flex;gap:18px;align-items:center;background:#172746;color:white;border-radius:20px;padding:24px;margin:14px 0;box-shadow:0 12px 30px rgba(19,36,67,.18)}.training-icon{width:58px;height:58px;border-radius:18px;background:#263d69;display:grid;place-items:center;font-size:31px}.training-console.running .training-icon{animation:pulse 1s infinite}.training-console.done{background:linear-gradient(120deg,#104c43,#177b62)}@keyframes pulse{50%{transform:scale(1.08);opacity:.7}}.training-copy{flex:1}.training-title{font-size:20px;font-weight:900;letter-spacing:.04em}.training-msg{color:#b8c8e5;margin:3px 0 11px}.training-console.done .training-msg{color:#c9f3e5}.training-track{height:11px;background:#2e4267;border-radius:99px;overflow:hidden}.training-track div{height:100%;background:linear-gradient(90deg,#6d5dfc,#4bc6ff);border-radius:99px;transition:width .25s}.training-pct{text-align:right;font-size:12px;margin-top:4px;color:#a9bee2}.ai-ready-card{display:flex;gap:18px;align-items:center;background:white;border:1px solid #dce5f0;border-radius:18px;padding:20px}.ai-ready-icon{font-size:42px}.ai-ready-card h2{margin:0!important}.ai-ready-card p{margin:5px 0}.callout{background:#eef5ff;padding:10px 12px;border-radius:10px;color:#405b80}
.growth-grid{display:grid;grid-template-columns:1.25fr .8fr 1fr;gap:13px}.pipeline-card,.ticker-card,.growth-card{border-radius:20px;padding:18px;border:1px solid #dce5f0;background:white;box-shadow:0 7px 22px rgba(33,58,98,.08)}.pipeline-flow{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:19px 0}.pipeline-flow div{flex:1;text-align:center}.pipeline-flow i{width:49px;height:49px;margin:auto;display:grid;place-items:center;border-radius:16px;background:#eeeaff;font-style:normal;font-size:23px}.pipeline-flow b{display:block;font-size:13px;margin-top:6px}.pipeline-flow span{font-size:10px;color:#758096}.pipeline-flow em{font-size:24px;color:#a3afc1}.pipeline-status{background:#f3f6fb;padding:10px;border-radius:11px;text-align:center;font-size:12px;font-weight:800;color:#53627b}.ticker-card{background:#132641;color:white;text-align:center}.ticker-card .panel-kicker{color:#8fb5e5}.ticker-number{font-size:58px;font-weight:900;line-height:1;margin-top:10px}.ticker-label{color:#9fb4d2;font-size:11px}.ticker-track{height:12px;background:#2c405c;border-radius:99px;margin:18px 0 5px;overflow:hidden}.ticker-track div{height:100%;background:linear-gradient(90deg,#23c6d5,#48a0ff);border-radius:99px}.ticker-ends{display:flex;justify-content:space-between;font-weight:900;font-size:12px}.ticker-ends small{display:block;color:#879bb7;font-size:8px}.growth-stat-row{display:grid;grid-template-columns:1fr 1.4fr;gap:8px;margin:14px 0}.growth-stat-row>div,.mini-result-row>div{background:#f2f8f5;border:1px solid #d7eadf;border-radius:13px;padding:12px}.growth-stat-row small{display:block;color:#68798b;font-size:9px;font-weight:900}.growth-stat-row b{font-size:24px;color:#17754c}.delta{display:block;color:#24875d;font-size:9px;font-weight:800}.mini-result-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mini-result-row span{font-size:10px;color:#647286;display:block}.mini-result-row b{font-size:22px;color:var(--ink)}.event-trigger-card{margin-top:13px;background:linear-gradient(120deg,#29205f,#441b6c);color:white;border-radius:18px;padding:16px 20px;display:flex;gap:14px;align-items:center}.event-trigger-card>span{font-size:35px}.event-trigger-card small{color:#d7b8ff;font-weight:900;letter-spacing:.12em}.event-trigger-card h3{margin:2px 0!important;color:white}.event-trigger-card p{margin:0;color:#d8d3ec}
.concept-unlock{display:flex;gap:15px;align-items:center;margin:15px 0;padding:17px 20px;border:1px solid #e3ca71;background:linear-gradient(120deg,#fff9df,#fff2c3);border-radius:18px;box-shadow:0 8px 22px rgba(149,118,24,.08);animation:unlockPop .45s ease-out}.unlock-icon{font-size:31px;width:52px;height:52px;border-radius:16px;display:grid;place-items:center;background:#ffe68f}.concept-unlock small{font-size:9px;font-weight:900;letter-spacing:.13em;color:#92731c}.concept-unlock h3{margin:1px 0!important}.concept-unlock p{margin:2px 0;color:#685a35}@keyframes unlockPop{from{transform:scale(.96);opacity:.3}}
.bias-strip{display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:#172746;border-radius:14px;padding:10px 12px;margin-top:12px}.bias-strip-label{font-size:9px;color:#9eb1d0;font-weight:900;letter-spacing:.12em;margin-right:6px}.bias-badge{display:flex;align-items:center;gap:6px;border-radius:99px;padding:7px 10px;font-size:11px;background:#263a5b;color:#97a7c2}.bias-badge.unlocked{background:#2f3861;color:#f4e9ff;border:1px solid #725fa7}.bias-badge i{font-style:normal;margin-left:3px}
.history-results-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:14px}.history-pool-card,.standout-panel,.results-shell,.allocation-card,.accuracy-grid{background:white;border:1px solid #dce5f0;border-radius:18px;padding:18px;box-shadow:0 7px 22px rgba(33,58,98,.07)}.history-pool-card h2{margin:4px 0 15px!important}.pool-split{display:grid;grid-template-columns:1fr 45px 1fr;gap:9px;align-items:center}.pool-split>div:not(.vs){text-align:center;background:#f3f6fb;border-radius:13px;padding:13px}.pool-split b{font-size:30px;display:block}.pool-split span{font-size:11px;color:#667085}.vs{text-align:center;font-weight:900;color:#61708a}.hire-rate{display:grid;grid-template-columns:135px 1fr 45px;gap:9px;align-items:center;margin-top:13px;font-size:11px}.hire-rate>div{height:9px;background:#e9edf3;border-radius:99px;overflow:hidden}.hire-rate i{display:block;height:100%;background:var(--blue)}.hire-rate.alt i{background:var(--purple)}.standout-panel{margin-top:14px}.standout-panel>div:first-child small{font-size:9px;letter-spacing:.12em;font-weight:900;color:var(--red)}.standout-panel h2{margin:3px 0 12px!important}.rejected-stack{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.rejected-card{border:1px solid #f2cbd1;background:#fff5f6;border-radius:14px;padding:12px;display:flex;align-items:center;gap:9px}.rejected-card h4{margin:2px 0!important;font-size:13px}.rejected-card p,.rejected-card small{font-size:9px;margin:0;color:#6c6470}.rejected-card>b{color:var(--red);font-size:10px;margin-left:auto}.reject-avatar{width:38px;height:38px;border-radius:12px;flex:0 0 38px}
.rate-grid,.accuracy-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:13px}.rate-grid>div,.accuracy-grid>div{border-radius:13px;padding:13px;background:#f4f7fb}.rate-grid small,.accuracy-grid small{font-size:9px;color:#67758b;font-weight:900}.rate-grid b,.accuracy-grid b{display:block;font-size:25px;color:var(--ink)}.allocation-card{margin-bottom:12px}.allocation-bar{display:flex;height:24px;border-radius:99px;overflow:hidden;background:#edf1f6;margin:12px 0}.allocation-bar i{display:block;height:100%}.allocation-bar .trad{background:var(--blue)}.allocation-bar .grad{background:var(--purple)}.allocation-labels{display:flex;justify-content:space-between;font-size:12px;color:#68758a}.accuracy-track{height:8px;background:#e4e9f0;border-radius:99px;overflow:hidden;margin-top:8px}.accuracy-track i{display:block;height:100%;background:var(--blue)}.accuracy-track.alt i{background:var(--purple)}
.final-screen{background:white;border:1px solid #dbe4ef;border-radius:26px;padding:32px;box-shadow:0 14px 34px rgba(33,58,98,.10)}.final-hero{text-align:center;max-width:760px;margin:0 auto 24px}.final-trophy{font-size:64px}.final-hero small{font-weight:900;letter-spacing:.15em;color:#6d5dfc}.final-hero h1{font-size:42px!important;margin:4px 0 8px!important}.final-hero p{font-size:17px;color:#61708a}.final-summary-sentence{text-align:center;font-size:20px;color:#34445f;margin:18px 0}.final-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.final-stats div{background:#f4f7fb;border-radius:16px;padding:18px;text-align:center}.final-stats span{display:block;font-size:11px;color:#68758a;margin-bottom:6px}.final-stats b{font-size:25px;color:var(--ink)}.takeaway{margin-top:14px;border-left:4px solid var(--teal);padding:9px 14px;background:#f0faf7}.takeaway h2{font-size:17px!important;margin:0 0 4px!important}.takeaway p{margin:0}.muted{color:#7a8699}
@media(max-width:1050px){.candidate-grid{grid-template-columns:repeat(3,1fr)}.growth-grid{grid-template-columns:1fr 1fr}.growth-card{grid-column:1/-1}.scoreboard-shell{grid-template-columns:repeat(2,1fr)}.banner-art{display:none}}
@media(max-width:700px){.candidate-grid{grid-template-columns:1fr 1fr}.scoreboard-shell{grid-template-columns:1fr 1fr}.score-tile{min-height:90px;padding:12px}.score-number{font-size:23px}.score-icon{display:none}.growth-grid,.history-results-grid{grid-template-columns:1fr}.rejected-stack{grid-template-columns:1fr}.mission-banner h1{font-size:27px!important}.candidate-card{min-height:300px}}

.graduate-only-result{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.graduate-only-result>div{background:#f4f7fb;border-radius:14px;padding:14px;text-align:center}.graduate-only-result small{display:block;font-size:9px;font-weight:900;color:#6c7890}.graduate-only-result b{display:block;font-size:28px;color:var(--ink)}
.value-impact{margin-top:7px;padding:7px 8px;border-radius:8px;background:#fff0d8;color:#8d5a13;font-size:9px;line-height:1.3}.discussion-card{display:flex;gap:18px;align-items:flex-start;margin:16px 0;background:linear-gradient(120deg,#eef5ff,#f5f0ff);border:1px solid #d5e1f4;border-radius:19px;padding:20px}.discussion-icon{width:52px;height:52px;border-radius:16px;background:#6d5dfc;color:white;display:grid;place-items:center;font-size:28px;flex:0 0 52px}.discussion-card small{font-size:9px;font-weight:900;letter-spacing:.13em;color:#6d5dfc}.discussion-card h2{margin:3px 0 6px!important}.discussion-card p{margin:0 0 8px;color:#53627b}.discussion-card ul{margin:7px 0 0 18px;color:#53627b}.discussion-card li{margin:4px 0}.rejected-card{align-items:flex-start}.rejected-card>div:nth-child(2){min-width:0;flex:1}.rejected-card>b{padding-top:3px}
@media(max-width:1100px){.final-stats{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){.final-stats{grid-template-columns:1fr 1fr}.graduate-only-result{grid-template-columns:1fr}}

/* v11 readability + icon consistency pass */
.briefing-explainer{background:#0b203c;border:1px solid rgba(255,255,255,.28);box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.briefing-goal{background:#0b203c;border:1px solid rgba(255,255,255,.28);box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.briefing-explainer-head p,.briefing-goal p{color:#ffffff!important;font-size:17px!important;line-height:1.58!important}
.briefing-goal-label{color:#d8efff!important;font-size:12px!important;letter-spacing:.12em}
.score-preview-tile{background:#17375f;border:1px solid rgba(255,255,255,.24);padding:17px}
.score-preview-icon{background:#244d7b;color:#fff!important;width:50px;height:50px}.score-preview-icon svg{width:30px;height:30px}
.score-preview-label{font-size:13px!important;color:#dff1ff!important}
.score-preview-short{font-size:17px!important;color:#fff!important;line-height:1.35}
.score-preview-detail{font-size:14px!important;color:#e5f1ff!important;line-height:1.5}
.briefing-goal-icon{color:#fff!important;width:50px;height:50px}.briefing-goal-icon svg{width:50px;height:50px}
.briefing-story{font-size:19px!important}.briefing-card p{font-size:15px!important;line-height:1.5!important}.briefing-card-step{font-size:11px!important}
.score-label{font-size:13px!important}.score-sub{font-size:13px!important}.score-denom{font-size:15px!important}
.score-icon{width:50px;height:50px}.score-icon svg{width:29px;height:29px}.culture-tile .score-icon{background:#f4edff;color:#7c54d8}.applicant-tile .score-icon{background:#fff3e4;color:#d7821e}.efficiency-tile .score-icon{background:#eaf3ff;color:#2d7de9}
.mission-kicker{font-size:13px!important}.mission-banner p{font-size:18px!important;line-height:1.45}
.candidate-name{font-size:20px!important}.pathway-badge{font-size:12px!important}.trait-row{font-size:15px!important}.rating{font-size:20px!important}.hire-pill{font-size:15px!important}.status-kicker{font-size:12px!important}.page-label{font-size:14px!important}
.side-panel-title small,.panel-kicker{font-size:12px!important}.side-panel-title h2{font-size:23px!important}.gauge-label{font-size:14px!important}.gauge-note{font-size:13px!important}.profile-panel h3{font-size:17px!important}.profile-item{font-size:14px!important}.game-tip{font-size:14px!important;line-height:1.5}
.pathway-split span{font-size:13px!important}.training-kicker{font-size:12px!important}.training-msg{font-size:15px!important}.training-pct{font-size:14px!important}.callout{font-size:14px!important;line-height:1.45}
.pipeline-flow b{font-size:15px!important}.pipeline-flow span{font-size:12px!important}.pipeline-status{font-size:14px!important}.ticker-label{font-size:13px!important}.ticker-ends{font-size:14px!important}.ticker-ends small{font-size:10px!important}.growth-stat-row small{font-size:11px!important}.delta{font-size:11px!important}.mini-result-row span{font-size:12px!important}
.concept-unlock small{font-size:11px!important}.concept-unlock p{font-size:14px!important}.bias-strip-label{font-size:11px!important}.bias-badge{font-size:13px!important}
.pool-split span{font-size:13px!important}.hire-rate{font-size:13px!important}.standout-panel>div:first-child small{font-size:11px!important}.rejected-card h4{font-size:15px!important}.rejected-card p,.rejected-card small{font-size:12px!important}.rejected-card>b{font-size:12px!important}.value-impact{font-size:12px!important}
.rate-grid small,.accuracy-grid small{font-size:11px!important}.allocation-labels{font-size:14px!important}.final-stats span{font-size:13px!important}.muted{font-size:13px!important;line-height:1.45}.graduate-only-result small{font-size:11px!important}.discussion-card small{font-size:11px!important}.discussion-card p,.discussion-card li{font-size:14px!important;line-height:1.45}

@media(max-width:800px){.briefing-screen{padding:34px 22px}.briefing-screen h1{font-size:38px!important}.briefing-story{font-size:16px}.briefing-cards{grid-template-columns:1fr}.briefing-card{min-height:0}.briefing-goal{align-items:flex-start}}

/* v12 onboarding readability + two-card briefing */
.briefing-screen{max-width:1080px!important;padding:52px 58px 48px!important;min-height:610px;display:flex;flex-direction:column;justify-content:center}
.briefing-brand{font-size:15px!important;letter-spacing:.17em!important;margin-bottom:24px!important}
.briefing-kicker{font-size:15px!important;padding:9px 15px!important;letter-spacing:.14em!important}
.briefing-screen h1{font-size:54px!important;margin:16px 0 28px!important}
.briefing-story{max-width:900px!important;font-size:22px!important;line-height:1.6!important;margin-bottom:30px!important}
.briefing-story p{font-size:22px!important;line-height:1.6!important}
.briefing-cards{gap:20px!important;margin:28px 0 20px!important}
.briefing-card{min-height:245px!important;padding:26px!important}
.briefing-card-icon,.briefing-card-icon svg{width:58px!important;height:58px!important}
.briefing-card-step{font-size:14px!important;letter-spacing:.12em!important;margin-top:4px}
.briefing-card h2{font-size:30px!important;margin:7px 0 10px!important}
.briefing-card p{font-size:18px!important;line-height:1.55!important}
.briefing-page-hint{margin-top:14px;font-size:16px;font-weight:800;color:#d7e9ff;letter-spacing:.02em}
.briefing-explainer{max-width:980px!important;padding:28px 30px!important;border-radius:22px!important;background:#071a31!important;border:2px solid rgba(157,210,255,.36)!important;box-shadow:0 12px 30px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.08)!important}
.briefing-section-title{font-size:19px!important;font-weight:900;letter-spacing:.12em;color:#bfe4ff;margin-bottom:8px}
.briefing-explainer-head p{font-size:21px!important;line-height:1.58!important;color:#fff!important;margin:8px 0 0!important}
.score-preview-grid{gap:16px!important;margin-top:22px!important}
.score-preview-tile{padding:18px!important;border-radius:18px!important;background:rgba(255,255,255,.115)!important}
.score-preview-icon{width:52px!important;height:52px!important;border-radius:15px!important}
.score-preview-icon svg{width:31px!important;height:31px!important}
.score-preview-label{font-size:14px!important;letter-spacing:.11em!important}
.score-preview-short{font-size:19px!important;line-height:1.3!important;margin:5px 0!important}
.score-preview-detail{font-size:16px!important;line-height:1.48!important}
.briefing-goal-hero{max-width:980px!important;margin:28px auto 0!important;padding:28px 30px!important;border-radius:22px!important;background:linear-gradient(120deg,#6248dd 0%,#7650ee 48%,#a66f22 140%)!important;border:2px solid rgba(255,225,143,.76)!important;box-shadow:0 15px 38px rgba(32,16,89,.34),0 0 0 5px rgba(255,216,112,.06)!important;align-items:center!important}
.briefing-goal-hero .briefing-goal-icon{width:66px!important;height:66px!important;flex:0 0 66px;padding:9px;border-radius:18px;background:rgba(255,255,255,.13)}
.briefing-goal-hero .briefing-goal-icon svg{width:48px!important;height:48px!important}
.briefing-goal-hero .briefing-goal-label{font-size:17px!important;letter-spacing:.14em!important;color:#ffe9a9!important;margin-bottom:4px!important}
.briefing-goal-hero h2{font-size:27px!important;line-height:1.15!important;color:#fff!important;margin:4px 0 9px!important}
.briefing-goal-hero p{font-size:20px!important;line-height:1.55!important;color:#fff!important;margin:0!important}
.briefing-flip-row{align-items:center!important}
.briefing-arrow-col{display:flex!important;align-items:center!important;justify-content:center!important}
.briefing-arrow-button button{width:58px!important;height:76px!important;min-width:58px!important;border-radius:18px!important;font-size:34px!important;font-weight:900!important;color:#214a7b!important;background:#fff!important;border:2px solid #cbd9ea!important;box-shadow:0 10px 25px rgba(28,60,104,.16)!important;padding:0!important}
.briefing-arrow-button button:hover{transform:translateY(-2px);border-color:#7baeea!important;box-shadow:0 14px 30px rgba(28,60,104,.22)!important}
.briefing-arrow-spacer{height:76px}
#start-company-btn button{min-height:64px!important;font-size:20px!important;letter-spacing:.03em!important}
@media(max-width:900px){.briefing-screen{padding:36px 26px!important;min-height:0}.briefing-screen h1{font-size:42px!important}.briefing-story,.briefing-story p{font-size:19px!important}.briefing-cards{grid-template-columns:1fr!important}.score-preview-grid{grid-template-columns:1fr!important}.briefing-arrow-button button{width:48px!important;min-width:48px!important;height:64px!important;font-size:28px!important}.briefing-goal-hero{align-items:flex-start!important}.briefing-goal-hero h2{font-size:23px!important}}


/* v13 polish: larger onboarding arrows, stable capitals, full founding-team grid */
.briefing-arrow-button button{
  width:82px!important;height:96px!important;min-width:82px!important;
  border-radius:22px!important;font-size:52px!important;line-height:1!important;
}
.briefing-arrow-spacer{height:96px!important}

/* Use standard supported weights/features on display text. This prevents
   variable-font synthesis from making isolated capitals appear mismatched. */
.briefing-brand,.briefing-kicker,.briefing-card-step,.briefing-card h2,
.briefing-section-title,.briefing-goal-label,.briefing-goal-hero h2,
.mission-kicker,.mission-banner h1,.status-kicker,
.app-header-brand,.app-header-subtitle{
  font-family:"Segoe UI",Arial,Helvetica,sans-serif!important;
  font-variant:normal!important;font-variant-caps:normal!important;
  font-feature-settings:"liga" 0,"calt" 0!important;
  font-synthesis:none!important;
}
.briefing-brand,.briefing-kicker,.briefing-card-step,.briefing-section-title,
.briefing-goal-label,.mission-kicker,.status-kicker{font-weight:900!important}
.briefing-card h2,.briefing-goal-hero h2,.mission-banner h1{font-weight:800!important}

/* Mission 1: all 20 applicants visible at once, five columns × four rows. */
.candidate-grid.founding-grid{
  grid-template-columns:repeat(5,minmax(0,1fr))!important;
  gap:14px!important;
  align-items:stretch;
}
.founding-grid .candidate-card{
  min-height:330px!important;
  padding:15px!important;
}
.founding-grid .candidate-name{font-size:18px!important}
.founding-grid .trait-row{font-size:14px!important;padding:9px 0 7px!important}
.founding-grid .rating{font-size:18px!important;letter-spacing:1.5px!important}
.founding-bottom-row{margin-top:16px!important;align-items:stretch!important}
.founding-lock-column{display:flex!important;flex-direction:column!important;justify-content:center!important}
#lock-founding-btn button{min-height:62px!important;font-size:18px!important;font-weight:900!important;border-radius:16px!important}

@media(max-width:1100px){
  .candidate-grid.founding-grid{grid-template-columns:repeat(4,minmax(0,1fr))!important}
}
@media(max-width:800px){
  .briefing-arrow-button button{width:64px!important;min-width:64px!important;height:78px!important;font-size:42px!important}
  .candidate-grid.founding-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
}


/* v14: force stable glyph rendering on the Start button itself. */
#start-company-btn button,#start-company-btn button *{
  font-family:Verdana,Arial,Helvetica,sans-serif!important;
  font-weight:700!important;
  font-style:normal!important;
  font-variant:normal!important;
  font-feature-settings:normal!important;
  font-synthesis:none!important;
  text-transform:none!important;
  letter-spacing:.02em!important;
}

/* Founding-team classroom pause. */
.founding-discussion-shell{background:#fff;border:1px solid #dbe5f1;border-radius:26px;padding:26px;box-shadow:0 12px 32px rgba(31,58,99,.10)}
.discussion-pause-banner{padding:22px 24px;border-radius:20px;background:linear-gradient(120deg,#172b50,#274d83);color:#fff;margin-bottom:16px}
.discussion-pause-kicker{font-size:14px;font-weight:800;letter-spacing:.14em;color:#b9dcff}
.discussion-pause-banner h1{font-size:34px!important;line-height:1.1!important;color:#fff!important;margin:7px 0 8px!important}
.discussion-pause-banner p{font-size:18px!important;line-height:1.5!important;color:#f2f7ff!important;margin:0!important}
.discussion-score-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0 18px}
.discussion-score-row>div{padding:14px 18px;border-radius:15px;background:#f2f6fc;border:1px solid #dce6f2;display:flex;align-items:center;justify-content:space-between}
.discussion-score-row span{font-size:14px;font-weight:800;letter-spacing:.08em;color:#63728a}
.discussion-score-row b{font-size:27px;color:#17223b}
.discussion-selected-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}
.discussion-hire-card{border:2px solid #dfe7f2;border-radius:18px;padding:15px;background:linear-gradient(180deg,#fff,#f8fbff)}
.discussion-hire-head{display:flex;gap:10px;align-items:center;margin-bottom:12px}.discussion-guild-badge{display:inline-block;margin-top:5px;padding:4px 7px;border-radius:999px;background:#fff4cf;color:#8a5b00;border:1px solid #efd47a;font-size:10px;font-weight:900;letter-spacing:.02em}
.discussion-hire-name{font-size:19px;font-weight:800;color:#17223b}.discussion-hire-sub{font-size:11px;font-weight:800;letter-spacing:.10em;color:#7b89a0;margin-top:3px}
.discussion-trait{border-top:1px solid #e8edf4;padding:9px 0 7px}.discussion-trait span{display:block;font-size:14px;font-weight:700;color:#56657b;margin-bottom:4px}.discussion-trait b{font-size:17px;letter-spacing:1px}
.discussion-prompts{margin-top:18px;padding:16px 18px;border-radius:14px;background:#fff6df;border:1px solid #edd59c;color:#604c22;font-size:17px;line-height:1.5}
#continue-training-btn button{min-height:58px!important;font-size:18px!important;font-weight:800!important;border-radius:15px!important}
@media(max-width:1000px){.discussion-selected-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:700px){.discussion-selected-grid{grid-template-columns:1fr 1fr}}




/* v15 typography cleanup: use a single web-safe glyph set for compact UI labels.
   This avoids the mixed-looking capital A seen in some browser/font combinations. */
.score-preview-label,.score-label,#start-company-btn button,#start-company-btn button *,
.briefing-brand,.briefing-kicker,.briefing-card-step,.briefing-section-title,
.briefing-goal-label,.mission-kicker,.status-kicker,.panel-kicker,
.discussion-pause-kicker,.discussion-hire-sub,.bias-strip-label,
.concept-unlock small,.standout-panel>div:first-child small,
.growth-stat-row small,.graduate-only-result small,.rate-grid small,.accuracy-grid small,
#lock-founding-btn button,#continue-training-btn button{
  font-family:Verdana,Arial,Helvetica,sans-serif!important;
  font-variant:normal!important;
  font-variant-caps:normal!important;
  font-feature-settings:normal!important;
  font-synthesis:none!important;
}
.score-preview-label,.score-label{font-weight:700!important}
#start-company-btn button,#start-company-btn button *{font-weight:700!important}

/* Keep each five-dot rating together. The filled and empty dots are separate
   spans, so without nowrap the browser can split them across two lines. */
.discussion-trait b,.discussion-trait .rating{
  display:block!important;
  white-space:nowrap!important;
  word-break:keep-all!important;
  overflow-wrap:normal!important;
}
.discussion-trait b{letter-spacing:0!important}
.discussion-trait .dots-on,.discussion-trait .dots-off{white-space:nowrap!important}

/* Discussion prompt readability. */
.discussion-prompts{font-size:18px!important;line-height:1.55!important}
.discussion-prompts strong{font-size:18px!important}


/* v16 Mission 1 discussion + Mission 2 training polish */
.discussion-rating{display:inline-flex!important;flex-wrap:nowrap!important;white-space:nowrap!important;word-break:keep-all!important;overflow-wrap:normal!important;align-items:center;gap:0;font-size:18px;line-height:1;letter-spacing:1px}
.discussion-rating .dots-on,.discussion-rating .dots-off{display:inline-block!important;white-space:nowrap!important;word-break:keep-all!important;flex:0 0 auto!important}
.discussion-team-profile{margin:8px 0 18px;padding:17px 19px;border-radius:16px;background:#f7f9fd;border:1px solid #dce5f1}
.discussion-team-profile .profile-panel{border-top:0!important;padding-top:0!important;margin-top:0!important}
.discussion-team-profile .profile-panel h3{font-size:20px!important;margin-bottom:13px!important;color:#17223b!important}
.discussion-team-profile .profile-item{font-size:15px!important;margin:10px 0!important}

.label-training-explainer{background:#fff;border:1px solid #d9e3f1;border-radius:22px;padding:24px 26px;margin:10px 0 18px;box-shadow:0 9px 26px rgba(32,58,98,.08)}
.label-explainer-kicker{font-family:Arial,Helvetica,sans-serif!important;font-size:13px;font-weight:800;letter-spacing:.12em;color:#6252dc}
.label-training-explainer h2{font-family:Arial,Helvetica,sans-serif!important;font-size:27px!important;margin:5px 0 7px!important;color:#17223b!important}
.label-training-explainer>p{font-size:17px!important;line-height:1.5!important;color:#53627b!important;margin:0 0 18px!important}
.label-flow{display:grid;grid-template-columns:1fr auto 1.25fr auto 1fr auto 1.35fr;gap:10px;align-items:stretch}
.label-count-card,.label-ai-card{min-height:118px;border-radius:16px;padding:15px 16px;display:flex;flex-direction:column;justify-content:center;border:1px solid #dae3ef}
.label-count-card span{font-family:Arial,Helvetica,sans-serif!important;font-size:12px;font-weight:800;letter-spacing:.08em}.label-count-card b{font-size:34px;line-height:1;margin:5px 0;color:#17223b}.label-count-card small,.label-ai-card small{font-size:13px;line-height:1.35;color:#617087}
.hire-label-card{background:#ebfaf5;border-color:#c9ecdf}.hire-label-card span{color:#16835f}.reject-label-card{background:#f3f5f8}.reject-label-card span{color:#59687d}.data-label-card{background:#f0efff;border-color:#d9d5ff}.data-label-card span{color:#6252dc}
.label-flow-arrow{display:grid;place-items:center;font-size:26px;font-weight:800;color:#97a4b7}
.label-ai-card{flex-direction:row;gap:12px;align-items:center;background:#edf5ff;border-color:#cfe1f8}.label-ai-icon{width:46px;height:46px;border-radius:14px;background:#dceaff;color:#315f9d;display:grid;place-items:center;flex:0 0 auto}.label-ai-icon svg{width:29px;height:29px}.label-ai-card b{display:block;font-family:Arial,Helvetica,sans-serif!important;font-size:18px;color:#17223b;margin-bottom:4px}
.label-explainer-callout{margin-top:15px;padding:13px 15px;border-radius:12px;background:#fff7e8;border:1px solid #f0d8a7;color:#624b1d;font-size:15px;line-height:1.45}

.training-console{background:linear-gradient(120deg,#172746,#203b68)!important;color:#fff!important;min-height:150px!important;padding:28px!important}
.training-console.done{background:linear-gradient(120deg,#1c3158,#403a79)!important;color:#fff!important;border:1px solid #675ca0!important}
.training-title,.training-title *{font-family:Arial,Helvetica,sans-serif!important;color:#fff!important;font-size:22px!important;font-weight:800!important;font-feature-settings:normal!important;font-synthesis:none!important}
.training-msg{color:#d8e4f7!important;font-size:17px!important;line-height:1.45!important;margin:6px 0 14px!important}
.training-console.done .training-msg{color:#e5e1ff!important}.training-pct{color:#c2d1ea!important;font-size:15px!important}
.training-icon{color:#fff!important}.training-console.done .training-icon{background:#5c55a5!important;color:#fff!important}
.ai-ready-card{border:1px solid #d8dfef!important;background:linear-gradient(120deg,#fff,#f6f4ff)!important}.ai-ready-icon{width:58px;height:58px;border-radius:18px;background:#ece9ff;color:#5f54c9;display:grid;place-items:center;flex:0 0 auto}.ai-ready-icon svg{width:34px;height:34px}.ai-ready-card h2,.ai-ready-card h2 *{font-family:Arial,Helvetica,sans-serif!important;font-feature-settings:normal!important;font-synthesis:none!important;font-size:25px!important;font-weight:800!important;color:#17223b!important}.ai-ready-card p{font-size:16px!important;line-height:1.5!important;color:#4d5d76!important}

#continue-training-btn button,#continue-training-btn button *,#train-ai-btn button,#train-ai-btn button *,#start-scaling-btn button,#start-scaling-btn button *,button.gr-button,button.gr-button *{font-family:Arial,Helvetica,sans-serif!important;font-feature-settings:normal!important;font-synthesis:none!important;font-variant:normal!important;font-weight:700!important}
#start-scaling-btn button{min-height:58px!important;font-size:17px!important;border-radius:15px!important}
@media(max-width:1050px){.label-flow{grid-template-columns:1fr 1fr}.label-flow-arrow{display:none}.label-ai-card{grid-column:1/-1}}

/* v17 Mission 2-3 cleanup */
.ticker-number{color:#fff!important}
.ticker-simple-note{margin-top:10px;color:#9fb4d2;font-size:13px;font-weight:700}
.training-console,.training-console *{font-family:"Trebuchet MS",Arial,sans-serif!important}
.training-console .training-title{color:#fff!important}
.training-console .training-msg{color:#d7e4f7!important;font-size:17px!important;line-height:1.45!important}
.training-console.done .training-msg{color:#ece9ff!important}
#continue-training-btn button,#continue-training-btn button *,#train-ai-btn button,#train-ai-btn button *,#start-scaling-btn button,#start-scaling-btn button *,#growth-round-btn button,#growth-round-btn button *,#degree-news-btn button,#degree-news-btn button *{font-family:"Trebuchet MS",Arial,sans-serif!important;font-feature-settings:normal!important;font-synthesis:none!important;font-variant:normal!important;font-variant-caps:normal!important;font-weight:700!important;line-height:1.15!important}
#growth-round-btn button,#degree-news-btn button,#post-guild-round-btn button,#post-guild-next-btn button{min-height:58px!important;font-size:17px!important;border-radius:15px!important;font-family:Arial,Helvetica,sans-serif!important;font-weight:700!important}



/* v18: candidate-name glyph consistency.
   Some browser/theme font combinations were rendering initial capitals (A/B)
   with a visibly smaller first-letter glyph. Force the whole candidate identity
   area to one conservative font, and explicitly neutralise ::first-letter. */
.candidate-name,.candidate-name *,
.discussion-hire-name,.discussion-hire-name *,
.abstract-avatar,.reject-avatar{
  font-family:Arial,Helvetica,sans-serif!important;
  font-style:normal!important;
  font-variant:normal!important;
  font-variant-caps:normal!important;
  font-feature-settings:normal!important;
  font-kerning:normal!important;
  font-synthesis:none!important;
  text-transform:none!important;
}
.candidate-name,.discussion-hire-name{
  font-weight:700!important;
  letter-spacing:0!important;
}
.candidate-name::first-letter,.discussion-hire-name::first-letter{
  font-family:inherit!important;
  font-size:1em!important;
  line-height:inherit!important;
  font-weight:inherit!important;
  letter-spacing:inherit!important;
}
.abstract-avatar::first-letter,.reject-avatar::first-letter{
  font-size:1em!important;
  line-height:inherit!important;
  font-weight:inherit!important;
}

/* v22 Guild-news + diagnosis flow */
.breaking-news-card{margin:10px 0 20px;padding:34px 38px;border-radius:24px;background:linear-gradient(125deg,#8a6200 0%,#c58a08 55%,#e2a91b 100%);border:1px solid #f2c450;box-shadow:0 14px 34px rgba(103,70,0,.20);color:#fff;position:relative;overflow:hidden}
.breaking-news-card:after{content:'';position:absolute;width:260px;height:260px;border-radius:50%;right:-90px;top:-110px;background:rgba(255,255,255,.10)}
.breaking-news-kicker{font-family:Arial,Helvetica,sans-serif;font-size:14px;font-weight:900;letter-spacing:.15em;color:#fff4c2;margin-bottom:8px}.breaking-news-card h1{font-family:Arial,Helvetica,sans-serif!important;font-size:34px!important;line-height:1.1!important;color:#fff!important;margin:0 0 12px!important;font-weight:900!important}.breaking-news-card p{font-size:18px;line-height:1.55;color:#fff;margin:0;max-width:900px}.breaking-news-detail{margin-top:18px;padding:13px 16px;border-radius:13px;background:rgba(62,39,0,.25);border:1px solid rgba(255,255,255,.20);font-size:15px;line-height:1.45;color:#fff}
.round-comparison-card{margin:16px 0;padding:20px 24px;border-radius:18px;background:linear-gradient(120deg,#fff7df,#fffdf5);border:1px solid #efce73;box-shadow:0 8px 22px rgba(122,89,10,.08)}.round-comparison-kicker{font-size:12px;font-weight:900;letter-spacing:.13em;color:#9b6e00}.round-comparison-card h2{font-family:Arial,Helvetica,sans-serif!important;font-size:24px!important;color:#2b3548!important;margin:5px 0 7px!important}.round-comparison-card p{font-size:16px;line-height:1.5;color:#596579;margin:0}
.diagnosis-full-card-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:18px 0}.diagnosis-candidate-card{background:#fff;border:2px solid #e1e7f0;border-radius:18px;padding:15px;box-shadow:0 7px 20px rgba(32,53,90,.08)}.guild-missing-badge{display:inline-flex;margin-top:5px;padding:4px 8px;border-radius:99px;background:#eef1f5;color:#68758a;font-size:10px;font-weight:900;letter-spacing:.06em}.diagnosis-ai-decision{margin-top:12px;padding:10px;border-radius:12px;background:#f7e9ec;border:1px solid #efcdd3;display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center}.diagnosis-ai-decision span{font-size:11px;color:#697487}.diagnosis-ai-decision b{font-size:17px;color:#29374f}.diagnosis-ai-decision strong{grid-column:1/-1;font-size:12px;color:#b6324a;letter-spacing:.08em}.value-impact.full{margin-top:10px;padding:9px 10px;border-radius:10px;background:#eef8f3;color:#35644f;font-size:12px;line-height:1.4}.discussion-card.large{margin-top:20px}.diagnosis-screen-shell .discussion-pause-banner p{font-size:17px}
#understood-news-btn button,#understood-news-btn button *,#diagnose-hiring-btn button,#diagnose-hiring-btn button *{font-family:Arial,Helvetica,sans-serif!important;font-weight:800!important;font-feature-settings:normal!important;font-synthesis:none!important;font-variant:normal!important;font-variant-caps:normal!important}
@media(max-width:1100px){.diagnosis-full-card-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){.diagnosis-full-card-grid{grid-template-columns:1fr 1fr}}

/* v23 robust historical-bias + repair flow */
.scaling-complete-card,.repair-success-card{margin-top:18px;padding:20px 22px;border-radius:18px;background:linear-gradient(135deg,#eef7ff,#f4f0ff);border:1px solid #cbdcf3;color:#213552}.scaling-complete-card h2,.repair-success-card h2{margin:5px 0 8px!important;color:#17365f}.scaling-complete-card p,.repair-success-card p{font-size:16px;color:#435875}.scaling-reassurance{margin-top:12px;padding:12px 14px;border-radius:12px;background:#fff5df;border:1px solid #f3d58a;color:#79520a;font-size:15px;font-weight:800}
.training-example-review{margin-top:24px;padding:20px;border-radius:20px;background:#f5f8fd;border:1px solid #dae3f0}.training-example-review h2{margin:4px 0 16px!important}.training-example-columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.training-example-columns h3{font-size:16px!important}.training-sample-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.training-sample-card{background:#fff;border:1px solid #dde5ef;border-radius:14px;padding:11px}.training-sample-card .candidate-name{font-size:16px}.training-sample-card .trait-row{font-size:11px;padding:6px 0}.training-sample-card .rating{font-size:14px}.training-label-pill{margin-top:8px;border-radius:9px;padding:7px;text-align:center;font-weight:900;font-size:11px}.training-sample-card.hire .training-label-pill{background:#e6f7ef;color:#147252}.training-sample-card.reject .training-label-pill{background:#f8e9ed;color:#a5364b}.diagnosis-training-badge{font-size:9px!important}
.fix-choice-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}.fix-choice-card{min-height:210px;background:#fff;border:2px solid #dce5f1;border-radius:20px;padding:22px;box-shadow:0 8px 22px rgba(30,55,95,.08)}.fix-choice-card h2{font-size:22px!important;color:#172f55}.fix-choice-card p{font-size:15px;line-height:1.5;color:#51627c}.fix-choice-number{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:#edf3ff;color:#315f9e;font-weight:900;font-size:20px}.fix-selected-card{margin:12px 0;padding:14px 16px;border-radius:14px;background:#edf5ff;border:1px solid #cbdcf2;color:#294769;font-size:15px}.fresh-training-wrap{margin-top:16px}.repair-success-card{background:linear-gradient(135deg,#edf7ff,#f3efff);border-color:#cbd8ee}
#wait-applicants-btn button,#wait-applicants-btn button *,#fix-ai-btn button,#fix-ai-btn button *,#repair-train-btn button,#repair-train-btn button *,#redeploy-ai-btn button,#redeploy-ai-btn button *,#repair-round-btn button,#repair-round-btn button *{font-family:Arial,Helvetica,sans-serif!important;font-weight:800!important;font-feature-settings:normal!important;font-synthesis:none!important}
.breaking-news-card{background:linear-gradient(135deg,#d8a31d,#f1c84f)!important;border-color:#f6dc83!important;color:#fff!important}.breaking-news-card h1,.breaking-news-card p,.breaking-news-card .breaking-news-detail,.breaking-news-card .breaking-news-kicker{color:#fff!important}
@media(max-width:900px){.training-example-columns,.fix-choice-grid{grid-template-columns:1fr}.training-sample-grid{grid-template-columns:1fr}}



/* v25: workforce profile + hire inspection */
.workforce-average-panel{margin-top:14px;padding-top:13px;border-top:1px solid #e6ebf2}.workforce-average-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:9px}.workforce-average-item{font-size:12px;color:#5c697d}.workforce-average-item span{display:block;font-weight:700}.workforce-average-item b{display:block;font-size:15px;color:#24344d;margin:2px 0 5px}.workforce-mini-track{height:5px;background:#e8edf4;border-radius:99px;overflow:hidden}.workforce-mini-track i{display:block;height:100%;background:#2d7de9;border-radius:99px}.hire-inspection{margin:16px 0 22px;background:linear-gradient(120deg,#eef4ff,#f5f0ff);border:2px solid #9ab7ef;border-radius:17px;box-shadow:0 8px 22px rgba(45,81,145,.13);overflow:hidden}.hire-inspection summary{cursor:pointer;padding:17px 20px;font-size:16px;font-weight:900;color:#203b68;list-style:none;position:relative;padding-right:54px}.hire-inspection summary::-webkit-details-marker{display:none}.hire-inspection summary:after{content:'+';position:absolute;right:20px;top:50%;transform:translateY(-50%);font-size:30px;line-height:1;color:#6d5dfc}.hire-inspection[open] summary:after{content:'−'}.hire-inspect-cta{display:block;font-size:17px;font-weight:900;letter-spacing:.04em;color:#284f92}.hire-inspect-sub{display:block;margin-top:3px;font-size:13px;font-weight:600;color:#667792}.hire-inspect-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:0 16px 16px}.hire-inspect-stat{padding:12px;border-radius:12px;background:#f6f8fc;border:1px solid #e2e8f1}.hire-inspect-stat span{display:block;font-size:12px;font-weight:800;color:#657286}.hire-inspect-stat b{display:block;font-size:16px;color:#24344d;margin:3px 0}.hire-inspect-stat small{font-size:12px;color:#7a8799}@media(max-width:900px){.hire-inspect-grid{grid-template-columns:1fr 1fr}}

/* v24: stable button glyphs + diagnosis/readability refinements */
button.primary,button.primary *,#growth-round-btn button,#growth-round-btn button *,#wait-applicants-btn button,#wait-applicants-btn button *,#understood-news-btn button,#understood-news-btn button *,#diagnose-hiring-btn button,#diagnose-hiring-btn button *,#fix-ai-btn button,#fix-ai-btn button *,#repair-train-btn button,#repair-train-btn button *,#redeploy-ai-btn button,#redeploy-ai-btn button *,#repair-round-btn button,#repair-round-btn button *{
  font-family:Verdana,Geneva,Tahoma,sans-serif!important;
  font-weight:700!important;
  font-feature-settings:"liga" 0,"calt" 0!important;
  font-kerning:normal!important;
  font-synthesis:none!important;
  font-variant:normal!important;
  font-variant-caps:normal!important;
  letter-spacing:0!important;
}
button.primary::first-letter,#growth-round-btn button::first-letter,#wait-applicants-btn button::first-letter,#understood-news-btn button::first-letter,#diagnose-hiring-btn button::first-letter,#fix-ai-btn button::first-letter,#repair-train-btn button::first-letter,#redeploy-ai-btn button::first-letter,#repair-round-btn button::first-letter{font-size:1em!important;font-family:inherit!important;font-weight:inherit!important}
.diagnosis-ai-decision.rejected-only{display:flex!important;align-items:center;justify-content:center;padding:12px!important}.diagnosis-ai-decision.rejected-only strong{font-size:15px!important;letter-spacing:.1em!important;color:#b6324a!important}
.value-impact.full{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:12px!important;padding:12px 13px!important;font-size:14px!important}.value-impact.full span{font-size:13px!important;font-weight:800!important;color:#456454!important}.value-impact.full b{font-size:22px!important;color:#19744f!important;white-space:nowrap!important}
.crisis-summary-card h2{font-size:25px!important;line-height:1.35!important}.crisis-summary-card p{font-size:17px!important}
.training-sample-card .training-label-pill.hire{background:#e6f7ef;color:#147252}.training-sample-card .training-label-pill.reject{background:#f8e9ed;color:#a5364b}


/* v27: interstate expansion + sampling-bias finale */
.interstate-warning-card{margin:18px 0;padding:30px 34px;border-radius:24px;background:linear-gradient(135deg,#fff5d9,#fffaf0);border:2px solid #edc55b;box-shadow:0 12px 28px rgba(126,91,10,.10);color:#26354c}.interstate-warning-card h1{font-family:Arial,Helvetica,sans-serif!important;font-size:31px!important;margin:6px 0 10px!important;color:#273752!important}.interstate-warning-card p{font-size:19px;line-height:1.55;color:#536078;margin:0}
.interstate-intro-card{margin:14px 0 18px;padding:26px 30px;border-radius:22px;background:linear-gradient(130deg,#edf5ff,#f7f1ff);border:1px solid #cbdcf2;box-shadow:0 9px 24px rgba(42,73,119,.10)}.interstate-intro-card h1{font-size:29px!important;color:#1e3c6c!important;margin:6px 0 9px!important}.interstate-intro-card p{font-size:17px;line-height:1.55;color:#53637d;margin:6px 0}.interstate-profile-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:17px}.interstate-profile-row>div{padding:13px;border-radius:13px;background:#fff;border:1px solid #dce5f0;text-align:center}.interstate-profile-row small{display:block;font-size:11px;font-weight:900;letter-spacing:.08em;color:#718097}.interstate-profile-row b{display:block;font-size:23px;color:#27476f;margin-top:3px}
.interstate-badge{display:inline-flex;margin-top:5px;padding:4px 8px;border-radius:99px;background:#eaf2ff;color:#426695;font-size:10px;font-weight:900;letter-spacing:.05em}.interstate-reject-card{min-height:330px}.interstate-preview-card{margin:18px 0;padding:23px 26px;border-radius:20px;background:linear-gradient(135deg,#eef7ff,#f4f0ff);border:2px solid #a9bfe8;display:grid;grid-template-columns:1fr auto;gap:8px 22px;align-items:center}.interstate-preview-card h2{font-size:21px!important;line-height:1.4!important;color:#263b60!important;margin:4px 0!important}.interstate-preview-card p{grid-column:1/-1;font-size:15px;color:#60708a;margin:5px 0 0}.interstate-preview-rate{text-align:center;padding:10px 17px;border-radius:15px;background:#fff;border:1px solid #d9e3f2;min-width:155px}.interstate-preview-rate b{display:block;font-size:34px;color:#5b51dc}.interstate-preview-rate span{display:block;font-size:11px;font-weight:800;color:#718097}
#repair-wait-btn button,#open-interstate-btn button,#interstate-hire-btn button,#interstate-diagnose-btn button,#interstate-fix-btn button,#interstate-train-btn button,#final-results-btn button{font-family:Verdana,Geneva,Tahoma,sans-serif!important;font-weight:700!important;font-feature-settings:"liga" 0,"calt" 0!important;font-synthesis:none!important}
@media(max-width:900px){.interstate-profile-row{grid-template-columns:1fr 1fr}.interstate-preview-card{grid-template-columns:1fr}.interstate-preview-rate{text-align:left}}


/* v28: force stable capitals throughout compact UI labels and progression controls */
.round-comparison-kicker,.discussion-pause-kicker,.mission-kicker,.briefing-kicker,.briefing-brand,.briefing-card-step,.briefing-goal-label,.score-label,.panel-kicker,.status-kicker,
.round-comparison-kicker::first-letter,.discussion-pause-kicker::first-letter,.mission-kicker::first-letter,.briefing-kicker::first-letter,.score-label::first-letter,
button,button *{
  font-family:Arial,Helvetica,sans-serif!important;
  font-variant:normal!important;font-variant-caps:normal!important;
  font-feature-settings:normal!important;font-synthesis:none!important;
}
.round-comparison-kicker::first-letter,.discussion-pause-kicker::first-letter,.mission-kicker::first-letter,.briefing-kicker::first-letter,.score-label::first-letter{font-size:1em!important;font-weight:inherit!important}
.discussion-pause-banner p b,.discussion-pause-banner p strong{color:#fff!important}
.interstate-result-card h2,.interstate-result-card p{color:#2b3548!important}
.prominent-preview{grid-template-columns:1fr!important}.interstate-preview-metrics{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}.interstate-preview-metrics>div{padding:16px;border-radius:15px;background:#fff;border:1px solid #d8e3f2}.interstate-preview-metrics span{display:block;font-size:12px;font-weight:900;letter-spacing:.08em;color:#65748b}.interstate-preview-metrics b{display:block;font-size:34px;color:#554bd6;margin:4px 0}.interstate-preview-metrics small{display:block;font-size:14px;color:#617087}
.learning-curve-shell{margin:16px 0;padding:28px 30px;border-radius:24px;background:linear-gradient(135deg,#eef5ff,#f8f3ff);border:1px solid #cfdcf0;box-shadow:0 10px 28px rgba(41,66,107,.10)}.learning-curve-shell h1{font-size:29px!important;color:#233b64!important;margin:7px 0 20px!important}.learning-curve-list{display:grid;gap:12px}.learning-curve-row{display:grid;grid-template-columns:100px 1fr 170px;gap:16px;align-items:center;padding:15px 17px;border-radius:16px;background:#fff;border:1px solid #dce5f0}.curve-sample{text-align:center}.curve-sample b{display:block;font-size:29px;color:#5b51dc}.curve-sample span{font-size:13px;color:#66758b}.curve-label{display:flex;justify-content:space-between;gap:14px;font-size:14px;color:#52627a}.curve-label b{color:#263a5d}.curve-track{height:10px;margin-top:7px;background:#e8edf5;border-radius:99px;overflow:hidden}.curve-track i{display:block;height:100%;background:linear-gradient(90deg,#6d5dfc,#2d7de9);border-radius:99px}.curve-value{text-align:right}.curve-value span{display:block;font-size:12px;font-weight:850;color:#6a778b}.curve-value b{display:block;font-size:24px;color:#1d8b63}.curve-value small{font-size:13px;color:#65748b}.learning-discussion{margin-top:22px}.final-redeploy-summary{margin:18px 0;padding:17px 20px;border-radius:16px;background:#eef4ff;border:1px solid #cfdcf3;text-align:center}.final-redeploy-summary span{display:block;font-size:12px;font-weight:900;letter-spacing:.08em;color:#6c7990}.final-redeploy-summary b{display:block;font-size:22px;color:#253e69;margin:4px 0}.final-redeploy-summary small{font-size:14px;color:#5f6f88}
@media(max-width:900px){.interstate-preview-metrics{grid-template-columns:1fr}.learning-curve-row{grid-template-columns:70px 1fr}.curve-value{grid-column:2;text-align:left}}

/* v29 tuning: clearer hire inspection and wider-market diagnosis */
.diagnosis-value-callout{margin-top:15px;padding:14px 18px;border-radius:14px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);font-size:17px!important;line-height:1.45;color:#fff!important}.diagnosis-value-callout b{font-size:20px!important;color:#fff!important}
.prominent-preview{margin-top:4px!important;border:3px solid #8c84f0!important;box-shadow:0 14px 34px rgba(76,68,190,.18)!important}.prominent-preview .round-comparison-kicker{font-size:14px!important}.prominent-preview h2{font-size:23px!important}
.hire-inspect-stat b{font-size:18px!important;color:#203f75!important}.hire-inspect-stat small{font-size:13px!important}

"""

CSS += r'''
/* v33: final label-priority reflection */
.final-priorities{margin:24px 0;padding:24px;border-radius:22px;background:#f2f6fc;border:1px solid #d9e3f0}.final-priorities h2{font-size:22px!important;color:#263b60!important;margin:6px 0 16px!important}.final-priority-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.final-label-profile{padding:18px;border-radius:16px;background:#fff;border:1px solid #dce5ef}.final-label-profile h3{font-size:18px!important;margin:0 0 5px!important;color:#263b60!important}.final-label-profile p{font-size:14px;color:#627087}.final-label-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}.final-label-grid>div{padding:10px 12px;border-radius:11px;background:#f7f9fd;border:1px solid #e4e9f1}.final-label-grid span{display:block;font-size:12px;color:#6a788c}.final-label-grid b{display:block;font-size:18px;color:#243958;margin-top:2px}@media(max-width:900px){.final-priority-grid{grid-template-columns:1fr}}
'''

# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="QUT001 Zylometry Lab — Visual Game") as demo:
    state = gr.State(initial_state())

    scoreboard = gr.HTML(scoreboard_html(initial_state()), elem_id="global-scoreboard", container=False, visible=False)

    gr.HTML("""<div style='text-align:center;margin:2px 0 12px'><div class='app-header-brand' style='font-weight:900;letter-spacing:.12em;color:#294b7c'>QUT001 · ZYLOMETRY LAB</div><div class='app-header-subtitle' style='color:#667085;font-size:14px'>AI Hiring & Company-Building Simulation</div></div>""", container=False)

    with gr.Walkthrough(selected=0) as walkthrough:
        # ------------------------------------------------------
        # Mission Briefing
        # ------------------------------------------------------
        with gr.Step("Mission Briefing", id=0):
            with gr.Group(visible=True) as briefing_page_one:
                with gr.Row(elem_classes="briefing-flip-row"):
                    with gr.Column(scale=1, min_width=70):
                        gr.HTML("<div class='briefing-arrow-spacer'></div>", container=False)
                    with gr.Column(scale=12):
                        gr.HTML(briefing_intro_html(), container=False)
                    with gr.Column(scale=1, min_width=70, elem_classes="briefing-arrow-col"):
                        briefing_next = gr.Button("→", elem_classes="briefing-arrow-button", variant="secondary")

            with gr.Group(visible=False) as briefing_page_two:
                with gr.Row(elem_classes="briefing-flip-row"):
                    with gr.Column(scale=1, min_width=70, elem_classes="briefing-arrow-col"):
                        briefing_back = gr.Button("←", elem_classes="briefing-arrow-button", variant="secondary")
                    with gr.Column(scale=12):
                        gr.HTML(briefing_dashboard_html(), container=False)
                    with gr.Column(scale=1, min_width=70):
                        gr.HTML("<div class='briefing-arrow-spacer'></div>", container=False)
                with gr.Row():
                    gr.Column(scale=2)
                    with gr.Column(scale=3):
                        start_btn = gr.Button("START COMPANY →", variant="primary", elem_id="start-company-btn")
                    gr.Column(scale=2)

            briefing_next.click(show_briefing_page_two, inputs=state, outputs=[briefing_page_one, briefing_page_two, scoreboard], queue=False)
            briefing_back.click(show_briefing_page_one, outputs=[briefing_page_one, briefing_page_two, scoreboard], queue=False)
            start_btn.click(start_company, inputs=state, outputs=[walkthrough, state, scoreboard], queue=False)

        # ------------------------------------------------------
        # Mission 1
        # ------------------------------------------------------
        with gr.Step("1 · Hire founders", id=1):
            with gr.Group(visible=True) as founding_selection_group:
                gr.HTML(mission_banner("MISSION 1", "Hire Your Founders", "Review all 20 applicants and choose exactly five people to launch your Zylometry start-up.", "team"), container=False)
                initial_status = gr.HTML(selection_status_html([False] * 20), container=False)
                initial_cards = gr.HTML(
                    make_all_card_data(NAMES, LEVEL1_VALUES, [False] * 20, guild_flags=LEVEL1_GUILD),
                    html_template=FOUNDING_CANDIDATE_TEMPLATE,
                    css_template="",
                    js_on_load=CANDIDATE_JS,
                    container=False,
                )
                with gr.Row(elem_classes="founding-bottom-row"):
                    with gr.Column(scale=3):
                        founding_panel = gr.HTML(founding_preview_html(initial_state()), container=False)
                    with gr.Column(scale=2, elem_classes="founding-lock-column"):
                        lock_btn = gr.Button("LOCK FOUNDING TEAM", variant="primary", elem_id="lock-founding-btn")
                        lock_status = gr.HTML(container=False)

                initial_cards.click(
                    toggle_initial_candidate,
                    inputs=state,
                    outputs=[initial_cards, state, initial_status, founding_panel, scoreboard],
                    queue=False,
                )

            # This is deliberately part of Mission 1 rather than a separate
            # Walkthrough tab. Locking the team swaps the content in-place.
            with gr.Group(visible=False) as founding_discussion_group:
                discussion_team = gr.HTML(
                    "<div class='founding-discussion-shell'><div class='discussion-pause-banner'><div class='discussion-pause-kicker'>CLASSROOM PAUSE</div><h1>Pause here for classroom discussion.</h1><p>Your five hires will appear here after you lock the founding team.</p></div></div>",
                    container=False,
                )
                continue_training_btn = gr.Button("CONTINUE TO MODEL TRAINING →", variant="primary", elem_id="continue-training-btn")
                continue_training_btn.click(go_training, outputs=walkthrough, queue=False)

            lock_btn.click(
                lock_initial_team,
                inputs=state,
                outputs=[lock_status, state, scoreboard, discussion_team, founding_selection_group, founding_discussion_group],
                queue=False,
            )

        # ------------------------------------------------------
        # Mission 2
        # ------------------------------------------------------
        with gr.Step("2 · Train the hiring AI", id=2):
            gr.HTML(mission_banner("MISSION 2", "Train Your Hiring AI", "", "ai", "purple"), container=False)
            gr.HTML(training_labels_html(), container=False)
            train_result = gr.HTML(training_html(0, "TRAINING HIRING MODEL", "Press Train when you are ready."), container=False)
            train_btn = gr.Button("TRAIN MODEL", variant="primary", elem_id="train-ai-btn")
            to_growth_btn = gr.Button("START SCALING COMPANY →", visible=False, variant="primary", elem_id="start-scaling-btn")
            train_btn.click(train_hiring_ai, inputs=state, outputs=[train_result, state, scoreboard, train_btn, to_growth_btn])

        # ------------------------------------------------------
        # Mission 3 — scale, news, then observe the frozen AI fail
        # ------------------------------------------------------
        with gr.Step("3 · Scale the company", id=3):
            gr.HTML(mission_banner("MISSION 3", "Scale the Company", f"Each round your AI screens {GROWTH_BATCH_SIZE} applicants. It only hires people whose AI score is comparable to the people you originally labelled Hire.", "growth"), container=False)
            growth_screen = gr.HTML(growth_screen_html(initial_state()), container=False)
            deploy_btn = gr.Button("RUN ANOTHER HIRING ROUND", variant="primary", elem_id="growth-round-btn")
            news_btn = gr.Button("WAIT FOR MORE APPLICANTS →", visible=False, variant="primary", elem_id="wait-applicants-btn")
            understood_btn = gr.Button("UNDERSTOOD — LET'S KEEP GROWING THE COMPANY →", visible=False, variant="primary", elem_id="understood-news-btn")
            diagnose_btn = gr.Button("Diagnose the AI →", visible=False, variant="primary", elem_id="diagnose-hiring-btn")
            to_growth_btn.click(go_growth, inputs=state, outputs=[walkthrough, growth_screen], queue=False)
            deploy_btn.click(deploy_growth_round, inputs=state, outputs=[growth_screen, state, scoreboard, deploy_btn, news_btn, understood_btn, diagnose_btn])
            news_btn.click(open_guild_event, inputs=state, outputs=[growth_screen, state, scoreboard, deploy_btn, news_btn, understood_btn, diagnose_btn], queue=False)
            understood_btn.click(acknowledge_guild_event, inputs=state, outputs=[growth_screen, state, scoreboard, deploy_btn, news_btn, understood_btn, diagnose_btn], queue=False)

        # ------------------------------------------------------
        # Mission 4 — diagnosis pause
        # ------------------------------------------------------
        with gr.Step("4 - Diagnose the AI", id=4):
            diagnosis_screen = gr.HTML("<div class='diagnosis-screen-shell'><div class='discussion-pause-banner'><div class='discussion-pause-kicker'>CLASSROOM PAUSE</div><h1>Diagnose the AI.</h1><p>Run a few hiring rounds after the industry news first.</p></div></div>", container=False)
            diagnosis_continue = gr.Button("I THINK I KNOW WHAT'S GOING ON →", variant="primary", elem_id="fix-ai-btn")
            diagnosis_continue.click(go_update, outputs=walkthrough, queue=False)
            diagnose_btn.click(go_diagnose, inputs=state, outputs=[walkthrough, diagnosis_screen], queue=False)

        # ------------------------------------------------------
        # Mission 5 — choose a repair strategy
        # ------------------------------------------------------
        with gr.Step("5 · Fix the AI", id=5):
            gr.HTML(mission_banner("MISSION 5", "Fix the Hiring AI", "You have diagnosed a problem. Choose one of two ways to update the system, then test whether hiring recovers.", "ai", "purple"), container=False)
            gr.HTML("""<div class='fix-choice-grid'><div class='fix-choice-card'><div class='fix-choice-number'>1</div><h2>Remove Guild accreditation as a feature</h2><p>Keep your original training data and original Hire / Do not hire labels, but do not let the AI use Guild accreditation when making decisions.</p></div><div class='fix-choice-card'><div class='fix-choice-number'>2</div><h2>Collect completely fresh training data</h2><p>Start again with 20 current applicants and create a new set of 5 Hire and 15 Do not hire labels.</p></div></div>""", container=False)
            with gr.Row():
                remove_guild_btn = gr.Button("CHOOSE OPTION 1", variant="secondary")
                fresh_data_btn = gr.Button("CHOOSE OPTION 2", variant="secondary")
            fix_status = gr.HTML(container=False)

            with gr.Group(visible=False) as fresh_data_group:
                fresh_status = gr.HTML(selection_status_html([False]*20, label="FRESH TRAINING LABELS"), container=False)
                fresh_cards = gr.HTML(make_all_card_data(CURRENT_NAMES, CURRENT_BATCH, [False]*20, guild_flags=CURRENT_GUILD), html_template=FOUNDING_CANDIDATE_TEMPLATE, js_on_load=CANDIDATE_JS, container=False)
                fresh_cards.click(toggle_fresh_candidate, inputs=state, outputs=[fresh_cards, state, fresh_status], queue=False)

            repair_train_result = gr.HTML(container=False)
            repair_train_btn = gr.Button("TRAIN THE AI", visible=False, variant="primary", elem_id="repair-train-btn")
            redeploy_ai_btn = gr.Button("RE-DEPLOY YOUR AI →", visible=False, variant="primary", elem_id="redeploy-ai-btn")

            remove_guild_btn.click(choose_remove_guild, inputs=state, outputs=[state, fix_status, fresh_data_group, repair_train_btn], queue=False)
            fresh_data_btn.click(choose_fresh_data, inputs=state, outputs=[state, fix_status, fresh_data_group, repair_train_btn, fresh_cards, fresh_status], queue=False)
            repair_train_btn.click(train_repaired_ai, inputs=state, outputs=[repair_train_result, state, repair_train_btn, redeploy_ai_btn])

        # ------------------------------------------------------
        # Mission 6 — re-deploy repaired AI
        # ------------------------------------------------------
        with gr.Step("6 · Re-deploy the AI", id=6):
            gr.HTML(mission_banner("RE-DEPLOY", "Test Your Updated Hiring AI", "A fresh pool of 200 local applicants is available. Run hiring rounds and see whether your fix worked.", "growth"), container=False)
            repaired_growth = gr.HTML(repaired_growth_html(initial_state()), container=False)
            repair_round_btn = gr.Button("RUN ANOTHER HIRING ROUND", variant="primary", elem_id="repair-round-btn")
            repair_wait_btn = gr.Button("WAIT FOR MORE APPLICANTS →", visible=False, variant="primary", elem_id="repair-wait-btn")
            redeploy_ai_btn.click(start_repaired_deployment, inputs=state, outputs=[walkthrough, state, repaired_growth], queue=False)
            repair_round_btn.click(deploy_repaired_round, inputs=state, outputs=[repaired_growth, state, scoreboard, repair_round_btn, repair_wait_btn])

        # ------------------------------------------------------
        # Mission 7 — expand interstate
        # ------------------------------------------------------
        with gr.Step("7 · Hire more widely", id=7):
            gr.HTML(mission_banner("MISSION 7", "Hire More Widely", "Your local hiring pipeline is working again. Now the company needs a larger talent pool.", "growth", "orange"), container=False)

            with gr.Group(visible=True) as interstate_wait_group:
                interstate_landing = gr.HTML(interstate_waiting_html(), container=False)
                open_interstate_btn = gr.Button("OPEN THE APPLICATION PORTAL TO INTERSTATE CANDIDATES →", variant="primary", elem_id="open-interstate-btn")

            with gr.Group(visible=False) as interstate_pool_group:
                interstate_pool_screen = gr.HTML(interstate_intro_html(), container=False)
                interstate_hire_btn = gr.Button("RUN HIRING ROUND · NEXT 50", visible=True, variant="primary", elem_id="interstate-hire-btn")
                interstate_diagnose_btn = gr.Button("INSPECT THE REJECTED APPLICANTS →", visible=False, variant="primary", elem_id="interstate-diagnose-btn")

            with gr.Group(visible=False) as interstate_diagnosis_group:
                interstate_diagnosis_screen = gr.HTML(container=False)
                interstate_fix_btn = gr.Button("I THINK I KNOW WHAT’S GOING ON →", variant="primary", elem_id="interstate-fix-btn")

            open_interstate_btn.click(
                open_interstate_portal,
                inputs=state,
                outputs=[state, interstate_wait_group, interstate_pool_group, interstate_pool_screen, scoreboard, interstate_hire_btn, interstate_diagnose_btn],
                queue=False,
            )
            interstate_hire_btn.click(
                deploy_interstate_pool,
                inputs=state,
                outputs=[interstate_pool_screen, state, scoreboard, interstate_hire_btn, interstate_diagnose_btn],
            )
            interstate_diagnose_btn.click(
                show_interstate_diagnosis,
                inputs=state,
                outputs=[interstate_pool_group, interstate_diagnosis_group, interstate_diagnosis_screen],
                queue=False,
            )

        # ------------------------------------------------------
        # Mission 8 — add wider-market examples
        # ------------------------------------------------------
        with gr.Step("8 · Fix the AI", id=8):
            gr.HTML(mission_banner("MISSION 8", "Teach the AI About This Applicant Profile", "Choose five applicants you would hire, add those labels to the training data, then re-test the AI on another 100 applicants.", "ai", "purple"), container=False)

            with gr.Group(visible=True) as interstate_fix_select_group:
                interstate_fix_status = gr.HTML(selection_status_html([False]*20, label="NEW HIRE EXAMPLES"), container=False)
                interstate_fix_cards = gr.HTML(
                    make_all_card_data(INTERSTATE_NAMES, INTERSTATE_LABEL_BATCH, [False]*20),
                    html_template=FOUNDING_CANDIDATE_TEMPLATE,
                    js_on_load=CANDIDATE_JS,
                    container=False,
                )
                interstate_train_btn = gr.Button("RETRAIN THE AI", interactive=False, variant="primary", elem_id="interstate-train-btn")
                interstate_train_status = gr.HTML(container=False)
                redeploy_interstate_btn = gr.Button("RE-TEST THE UPDATED AI ON 100 APPLICANTS →", visible=False, variant="primary", elem_id="full-redeploy-btn")

            with gr.Group(visible=False) as interstate_fix_deploy_group:
                interstate_fix_deploy_screen = gr.HTML(container=False)
                interstate_fix_round_btn = gr.Button("RUN HIRING ROUND · NEXT 50", visible=True, variant="primary", elem_id="interstate-fix-round-btn")
                final_results_btn = gr.Button("SEE MY FINAL COMPANY VALUATION AND STATS →", visible=False, variant="primary", elem_id="final-results-btn")

            interstate_fix_cards.click(
                toggle_interstate_training_candidate,
                inputs=state,
                outputs=[interstate_fix_cards, state, interstate_fix_status, interstate_train_btn],
                queue=False,
            )
            interstate_train_btn.click(
                train_interstate_fix_ai,
                inputs=state,
                outputs=[interstate_train_status, state, interstate_train_btn, redeploy_interstate_btn],
            )
            redeploy_interstate_btn.click(
                start_interstate_fix_redeploy,
                inputs=state,
                outputs=[state, interstate_fix_select_group, interstate_fix_deploy_group, interstate_fix_deploy_screen, scoreboard, interstate_fix_round_btn, final_results_btn],
                queue=False,
            )
            interstate_fix_round_btn.click(
                deploy_interstate_fix_round,
                inputs=state,
                outputs=[interstate_fix_deploy_screen, state, scoreboard, interstate_fix_round_btn, final_results_btn],
            )

        # ------------------------------------------------------
        # Final screen
        # ------------------------------------------------------
        with gr.Step("9 · Final company valuation and stats", id=9):
            final_screen = gr.HTML("<div class='final-screen'><div class='final-hero'><div class='final-trophy'>★</div><h1>Final company valuation and stats</h1><p>Complete the final re-deployment to reveal your company results.</p></div></div>", container=False)

        # Cross-step navigation wired after all target components exist.
        repair_wait_btn.click(
            go_hire_more_widely,
            inputs=state,
            outputs=[walkthrough, state, interstate_landing, interstate_wait_group, interstate_pool_group, scoreboard],
            queue=False,
        )
        interstate_fix_btn.click(
            go_fix_interstate,
            inputs=state,
            outputs=[walkthrough, state, interstate_fix_cards, interstate_fix_status, interstate_train_btn],
            queue=False,
        )
        final_results_btn.click(
            lambda s: (gr.Walkthrough(selected=9), final_report(s)),
            inputs=state,
            outputs=[walkthrough, final_screen],
            queue=False,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=CSS
    )
