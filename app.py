import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="HealthWise Smart Premium Engine", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("healthwise_model.joblib")


M = load_model()
clf, regs, FEATURES, NUM, med = M["clf"], M["regs"], M["features"], M["num"], M["med"]

# ─────────────────────────────────────────────────────────────────────────────
# Relationships learned from healthwise.csv (de-duplicated, 1,164 complete rows).
# Only pairs with a real correlation are linked. Checked and NOT linked:
#   age–bmi r=+0.04 · age–exercise r=-0.01 · age–children r=0.00 · bmi–children r=-0.01
#   sex–smoker (Cramér's V 0.08) · children–region (eta² 0.014)  → all negligible.
# To link another pair later, add one row to LINKS (and its mean/sd to STATS).
# ─────────────────────────────────────────────────────────────────────────────
STATS = {"bmi": (27.895, 5.813), "exercise_freq": (4.003, 1.233)}      # (mean, sd)
LINKS = [("bmi", "exercise_freq", -0.786, "BMI ↔ Exercise")]           # (a, b, r, label)

SPEC = {  # label, min, max, step
    "age": ("Age", 18, 64, 1),
    "bmi": ("BMI", 16.0, 50.0, 0.1),
    "children": ("Children", 0, 4, 1),
    "exercise_freq": ("Exercise / week", 0, 7, 1),
}
DEFAULTS = {"age": int(med["age"]), "bmi": round(float(med["bmi"]), 1),
            "children": int(med["children"]), "exercise_freq": int(med["exercise_freq"]),
            "sex": "male", "smoker": "no", "region": "north"}
TIERS = ["Low", "Medium", "High"]


def implied(target, source, value, r):
    """Expected value of `target` given `source` (simple regression line implied by r)."""
    mu_s, sd_s = STATS[source]
    mu_t, sd_t = STATS[target]
    return mu_t + r * (sd_t / sd_s) * (value - mu_s)


def on_slider_change(changed):
    """When a linked slider moves, nudge its partner to the value typical for that customer.
    Skipped when that link's checkbox is off, so the user can move one feature on its own."""
    for a, b, r, _ in LINKS:
        if not st.session_state.get(f"link_{a}_{b}", True):
            continue
        if changed not in (a, b):
            continue
        src, tgt = (a, b) if changed == a else (b, a)
        _, lo, hi, step = SPEC[tgt]
        new = min(max(implied(tgt, src, st.session_state[src], r), lo), hi)
        st.session_state[tgt] = round(new, 1) if isinstance(step, float) else int(round(new))


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def bmi_band(b):
    return "Underweight" if b < 18.5 else "Normal" if b < 25 else "Overweight" if b < 30 else "Obese"


def predict(c):
    row = pd.DataFrame([c])
    for k in NUM:
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = med[k]
    p = clf.predict_proba(row[FEATURES])[0]
    proba = dict(zip(clf.classes_, p))
    t = clf.classes_[p.argmax()]                                          # Stage 1
    return t, max(float(regs[t].predict(row[FEATURES])[0]), 0.0), proba   # Stage 2


# ── State: start from the *typical* customer (dataset medians) ───────────────
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.title("🏥 HealthWise — Smart Premium Engine")
st.caption("Stage 1 classifies the risk tier; Stage 2 prices the customer with that tier's own model. "
           "Change any input and the result updates live.")

# ── Inputs ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Customer profile")
    st.markdown("**Link correlated features**")
    for a, b, r, label in LINKS:
        st.checkbox(f"{label}  (r = {r:+.2f})", value=True, key=f"link_{a}_{b}",
                    help="ON: moving one slider moves the other to the value typical for that customer, "
                         "so you don't build unrealistic profiles.\n\nOFF: each slider moves independently "
                         "(use this for clean 'change only one thing' what-ifs).")
    st.button("↺ Reset to typical customer", on_click=reset)
    st.caption("Age, children, sex, smoker and region are not meaningfully correlated with anything else "
               "in the data, so they always move independently.")

c1, c2 = st.columns(2)
with c1:
    for k, (label, lo, hi, step) in SPEC.items():
        st.slider(label, lo, hi, step=step, key=k, on_change=on_slider_change, args=(k,))
        if k == "bmi":
            st.caption(f"BMI category: {bmi_band(st.session_state['bmi'])}")
with c2:
    st.selectbox("Sex", ["male", "female"], key="sex")
    st.selectbox("Smoker", ["no", "yes"], key="smoker")
    st.selectbox("Region", ["north", "south", "east", "west"], key="region")

cust = dict(age=st.session_state["age"], bmi=st.session_state["bmi"], children=st.session_state["children"],
            exercise_freq=st.session_state["exercise_freq"], sex=st.session_state["sex"],
            smoker=st.session_state["smoker"], region=st.session_state["region"])

# Warn when the profile is an unusual combination (mostly happens with links switched off)
for a, b, r, label in LINKS:
    typical = implied(b, a, cust[a], r)
    resid_sd = STATS[b][1] * (1 - r ** 2) ** 0.5
    if abs(cust[b] - typical) > 2 * resid_sd:
        st.warning(f"Unusual combination: {SPEC[a][0]} {cust[a]} with {cust[b]} {SPEC[b][0].lower()} is rare in the "
                   f"training data (typical is about {typical:.0f}). The estimate is less reliable here.")

# ── Results ──────────────────────────────────────────────────────────────────
tier, premium, proba = predict(cust)
base = st.session_state.get("baseline")
delta = None
if base:
    d = premium - base["premium"]
    delta = f"{'+' if d >= 0 else '-'}${abs(d):,.0f} vs baseline"

st.divider()
m1, m2 = st.columns(2)
m1.metric("Predicted Risk Tier", tier)
m2.metric("Estimated Annual Premium", f"${premium:,.0f}", delta=delta, delta_color="inverse")

for t in TIERS:
    st.progress(float(proba.get(t, 0.0)), text=f"{t}: {proba.get(t, 0.0):.0%}")
if max(proba.values()) < 0.6:
    st.caption("Borderline: the model is not confident about this tier, so a small change in inputs may switch "
               "the tier and cause a jump in premium.")

b1, b2 = st.columns([1, 3])
if b1.button("📌 Pin as baseline"):
    st.session_state["baseline"] = dict(tier=tier, premium=premium, cust=dict(cust))
    st.rerun()
if base:
    changed = [f"{k}: {base['cust'][k]} → {cust[k]}" for k in cust if base["cust"][k] != cust[k]]
    b2.caption(f"Baseline was **{base['tier']}, ${base['premium']:,.0f}**. "
               f"Changed: {', '.join(changed) if changed else 'nothing yet'}.")

# ── Batch scoring ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Batch scoring — upload unseen customers (CSV)")
up = st.file_uploader("CSV with columns: age, bmi, children, exercise_freq, sex, smoker, region", type="csv")
if up:
    new = pd.read_csv(up)
    missing = [c for c in FEATURES if c not in new.columns]
    if missing:
        st.error(f"Missing column(s): {', '.join(missing)}")
    else:
        allowed = {"sex": ["male", "female"], "smoker": ["no", "yes"], "region": ["north", "south", "east", "west"]}
        for c, ok in allowed.items():
            new[c] = new[c].astype(str).str.strip().str.lower()
            bad = (~new[c].isin(ok)).sum()
            if bad:
                st.warning(f"{bad} row(s) have an unrecognised '{c}' value (expected {', '.join(ok)}).")
        out = [predict(dict(r)) for _, r in new.iterrows()]
        new["predicted_tier"] = [o[0] for o in out]
        new["predicted_premium"] = [round(o[1]) for o in out]
        st.dataframe(new)
        st.download_button("Download predictions", new.to_csv(index=False), "predictions.csv", "text/csv")
