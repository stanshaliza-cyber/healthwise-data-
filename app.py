import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="HealthWise Smart Premium Engine", page_icon="🏥")
M = joblib.load('healthwise_model.joblib')
clf, regs, FEATURES, NUM, med = M['clf'], M['regs'], M['features'], M['num'], M['med']

def predict(c):
    row = pd.DataFrame([c])
    for k in NUM:
        if k not in row or pd.isna(row.at[0, k]): row[k] = med[k]
    proba = dict(zip(clf.classes_, clf.predict_proba(row[FEATURES])[0]))
    t = clf.predict(row[FEATURES])[0]                                  # Stage 1
    return t, max(float(regs[t].predict(row[FEATURES])[0]), 0.0), proba  # Stage 2

st.title("🏥 HealthWise — Smart Premium Engine")
st.caption("Stage 1 classifies the risk tier; Stage 2 prices the customer with that tier's own model. Change any input and the result updates live.")

c1, c2 = st.columns(2)
with c1:
    age      = st.slider("Age", 18, 64, 40)
    bmi      = st.slider("BMI", 16.0, 50.0, 27.0, 0.1)
    children = st.slider("Children", 0, 4, 1)
    exercise = st.slider("Exercise / week", 0, 7, 3)
with c2:
    sex    = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])

tier, premium, proba = predict(dict(age=age, bmi=bmi, children=children, exercise_freq=exercise,
                                    sex=sex, smoker=smoker, region=region))
m1, m2 = st.columns(2)
m1.metric("Predicted Risk Tier", tier)
m2.metric("Estimated Annual Premium", f"${premium:,.0f}")
st.write("Tier probabilities:", {k: f"{v:.0%}" for k, v in proba.items()})

st.divider()
st.subheader("Batch scoring — upload unseen customers (CSV)")
up = st.file_uploader("CSV with columns: age, bmi, children, exercise_freq, sex, smoker, region", type="csv")
if up:
    new = pd.read_csv(up)
    out = [predict(dict(r)) for _, r in new.iterrows()]
    new["predicted_tier"] = [o[0] for o in out]
    new["predicted_premium"] = [round(o[1]) for o in out]
    st.dataframe(new)
    st.download_button("Download predictions", new.to_csv(index=False), "predictions.csv")
