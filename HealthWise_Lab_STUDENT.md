# 🏥 HealthWise Insurance — Building a Smart Premium Engine
### End-to-End Machine Learning Lab Assignment

**Course:** Machine Learning — MAIB
**Dataset provided:** `healthwise.csv` (use the exact file given to you)
**What you submit:** Your completed notebook — **every code cell run with its output visible** — your **Reflection** answers inline, and your **working GUI app** (Part E).
**You will also receive:** a small set of **unseen customers** from your instructor to score with your finished app (Task E3).
**Marks:** 35

---

## 1. The Business Scenario

**HealthWise Insurance** sells annual health-insurance policies. Today the company uses a **single flat pricing model** for everyone. This is hurting the business in two ways:

- **Low-risk customers are overcharged** → they leave for cheaper competitors.
- **High-risk customers are undercharged** → the company loses money on claims.

The leadership team believes that **customers are not all the same** — a young, non-smoking, active customer and an older, smoking, high-BMI customer have completely different cost structures, and pricing them with one formula is a mistake.

You have been hired as the **data science team** to design a **two-stage "Smart Premium Engine":**

> **Stage 1 — CLASSIFY:** Sort each customer into a **risk tier** — `Low`, `Medium`, or `High` — based on their profile.
>
> **Stage 2 — PREDICT:** Estimate the customer's **exact annual charge** using a **regression model trained only on customers in that same tier.**

At the end, the **board wants a clear, evidence-backed answer to two questions:**

1. **Does this two-stage approach actually beat a single one-size-fits-all model?**
2. **Which customer attributes truly drive cost?**

Your job is to build the engine, **prove your answer with numbers you generate**, and write a short recommendation.

---

## 2. Learning Objectives

| Stage | Skills exercised |
|---|---|
| Data handling | Loading, inspection, **cleaning**, duplicates, missing values |
| Understanding | **EDA**, distributions, target skew, correlations |
| Feature work | **Feature engineering**, encoding, **identifying & dropping useless features**, spotting **redundant** features |
| Inference | **Hypothesis testing** |
| Diagnostics | **Multicollinearity (VIF)**, **bias–variance** reasoning |
| Selection | **Automated feature selection** with Lasso |
| Modelling 1 | **Classification** (Stage 1) |
| Modelling 2 | **Simple vs. Multiple Linear Regression**, **Ridge**, **Lasso** (Stage 2) |
| Tuning | **Fine-tuning** the regularisation strength (alpha) |
| Judgement | Model comparison, feature importance, business recommendation |
| Deployment | Saving a model, building an **interactive GUI** that predicts tier + premium and **recalculates live** as inputs change |

---

## 3. The Dataset — `healthwise.csv`

Each row is one customer. **~1,200 rows, 16 columns.**

> ⚠️ **Important:** HealthWise's database team exported *every* field they had on file — including some that have **nothing to do with health cost**, and some that **duplicate** information already present in another column. **Part of your job is to decide which columns to keep, which to drop as useless, and which to drop as redundant.** Do **not** assume every column is a useful predictor.

| Column | Meaning |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Age in years |
| `sex` | male / female |
| `bmi` | Body Mass Index (has some missing values) |
| `weight_kg` | Body weight in kilograms |
| `children` | Number of dependents |
| `smoker` | yes / no |
| `region` | north / south / east / west |
| `exercise_freq` | Workouts per week, 0–7 (has some missing values) |
| `date_of_birth` | Date of birth |
| `favorite_color` | Customer's favourite colour |
| `zodiac_sign` | Customer's star sign (12 categories) |
| `lucky_number` | A number the customer picked (1–99) |
| `preferred_contact` | email / phone / sms |
| `marketing_opt_in` | Has the customer opted into marketing emails? |
| `annual_charge` | **TARGET for Stage 2** — annual medical charge |
| `risk_tier` | **TARGET for Stage 1** — Low / Medium / High |

> 🔎 The data is deliberately a little "messy" (missing values, duplicate rows). Handling this correctly is part of the marks.

---

## 4. How to Work Through This Lab

- Work through the parts **in order** — each builds on the previous one.
- For **every task**: run the code, **keep the output visible**, then answer the **Reflection** questions in a markdown cell below.
- The reflections are where most of the marks are. **"It ran successfully" is not an answer** — we want *what you saw* and *why it matters for HealthWise*.
- Where you see **`# YOUR CHOICE`** / **`# TODO`**, you must make and justify a decision.
- Use **one fixed `random_state`** everywhere so results are reproducible.

**Setup — run this first:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

---

# PART A — Foundations (Understand, Clean & Select)

## Task A1 — Load and Inspect

```python
df = pd.read_csv('healthwise.csv')
print("Shape:", df.shape)
display(df.head())
print(df.info())
display(df.describe(include='all'))
```

**Reflection A1**
1. How many rows and columns are there? What is the unit of one row?
2. Which columns are numeric and which are categorical?
3. Glancing at the columns, list any that **already look suspicious** as predictors of medical cost (you'll test this properly later).

---

## Task A2 — Clean the Data

```python
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

print(df.isna().sum())
# YOUR CHOICE: handle missing bmi and exercise_freq (justify in reflection)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df['exercise_freq'] = df['exercise_freq'].fillna(df['exercise_freq'].median())
print("Remaining missing:", df.isna().sum().sum())
```

**Reflection A2**
1. How many duplicate rows did you remove? Why are duplicates dangerous in modelling?
2. Which columns had missing values, and how many?
3. **Justify** your imputation choice. Why might *median* be safer than *mean* for a skewed column like `bmi`?

---

## Task A3 — Exploratory Data Analysis (EDA)

```python
plt.figure(figsize=(6,4))
sns.histplot(df['annual_charge'], kde=True); plt.title('Distribution of Annual Charge'); plt.show()
print("Skew of annual_charge:", round(df['annual_charge'].skew(), 2))

sns.boxplot(data=df, x='smoker', y='annual_charge'); plt.title('Charge by Smoker'); plt.show()
sns.boxplot(data=df, x='risk_tier', y='annual_charge', order=['Low','Medium','High']); plt.show()
print(df['risk_tier'].value_counts())
```

**Reflection A3**
1. Is `annual_charge` symmetric or **skewed**? What is the skew value, and what does it imply for a linear model?
2. Roughly how much more do smokers cost? Large or small gap?
3. Are the three tiers **balanced**? Why does class balance matter for Stage 1?

---

## Task A4 — Feature Audit: Identify Useless & Redundant Columns ⭐

> This is a key step. Before modelling, you must **audit every column** and sort it into one of three buckets:
> **(a) Useful predictor**, **(b) Redundant** (duplicates another column), or **(c) Irrelevant** (no plausible link to medical cost).

**Step 1 — Reason about it first (business sense).**
Look at the column list and ask: *could this plausibly affect a person's medical cost?*

**Step 2 — Back your reasoning with evidence.**

```python
# (i) Numeric features vs the target — weak/zero correlation = suspect
num_cols = ['age','bmi','weight_kg','children','exercise_freq','lucky_number']
print("Correlation with annual_charge:")
print(df[num_cols + ['annual_charge']].corr()['annual_charge'].drop('annual_charge').round(3))

# (ii) Categorical features vs the target — do group averages actually differ?
for c in ['sex','smoker','region','favorite_color','zodiac_sign','preferred_contact','marketing_opt_in']:
    grp = df.groupby(c)['annual_charge'].mean()
    spread = (grp.max() - grp.min()) / df['annual_charge'].mean()
    print(f"  {c:18} charge varies {spread*100:4.1f}% across its categories")
```

**Step 3 — Decide and drop.**

```python
# Drop the identifier
drop_cols = ['customer_id']

# YOUR CHOICE: after studying the evidence above, add the columns YOU judge to be
# IRRELEVANT (no plausible link to medical cost). Justify each one in Reflection A4.
# drop_cols += [ ... ]          # <-- you decide which columns go here

df_model = df.drop(columns=drop_cols)
print("Kept columns:", list(df_model.columns))
```

**Reflection A4**
1. Which columns did you drop as **irrelevant**, and what is your business reasoning for each?
2. Look at `lucky_number`'s correlation with charge — what is it, and why is that expected?
3. Some categorical columns may show a surprisingly large "spread" in average charge across their categories. Before trusting any such pattern, consider: how many categories does that column have, and how many customers fall in each? What would you check before concluding the pattern is real?
4. You still have `weight_kg`, `bmi`, `exercise_freq`, `age`, and `date_of_birth` — some of these overlap. You'll deal with that redundancy in Task A6.

---

## Task A5 — Hypothesis Testing

> **Business question:** *"Do smokers really pay significantly more, or could the difference be random chance?"*

```python
smokers    = df[df['smoker']=='yes']['annual_charge']
nonsmokers = df[df['smoker']=='no']['annual_charge']
t_stat, p_val = stats.ttest_ind(smokers, nonsmokers, equal_var=False)
print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")

r, p_corr = stats.pearsonr(df['age'], df['annual_charge'])
print(f"age vs charge: r = {r:.3f}, p = {p_corr:.5f}")
```

**Reflection A5**
1. State your **null (H₀)** and **alternative (H₁)** hypotheses for the smoker test.
2. At α = 0.05, do you **reject or fail to reject** H₀? Explain the p-value in plain business language.
3. What does the age–charge correlation add?

---

## Task A6 — Redundancy & Multicollinearity (VIF)

> Multicollinearity = two or more features carrying **duplicate information**. It makes regression coefficients **unstable and untrustworthy**. You have some obvious candidates: `age` vs `date_of_birth`, and `bmi` vs `weight_kg`.

```python
# Investigate the redundant pairs
df['age_from_dob'] = 2026 - pd.to_datetime(df['date_of_birth']).dt.year
print("age vs date_of_birth corr:", round(df['age'].corr(df['age_from_dob']), 3))
print("bmi vs weight_kg  corr   :", round(df['bmi'].corr(df['weight_kg']), 3))

from statsmodels.stats.outliers_influence import variance_inflation_factor  # pip install statsmodels
num_feats = ['age','bmi','weight_kg','children','exercise_freq']
Xv = df[num_feats].dropna()
vif = pd.DataFrame({'feature': num_feats,
                    'VIF': [variance_inflation_factor(Xv.values, i) for i in range(len(num_feats))]})
print(vif)
sns.heatmap(df[num_feats].corr(), annot=True, cmap='coolwarm'); plt.show()
```

**Reflection A6**
1. What is the correlation between `bmi` and `weight_kg`? Between `age` and `date_of_birth`?
2. Which features have a **VIF above 5** (or above 10)? Which pair is the worst offender?
3. For each redundant pair, **which column do you keep and which do you drop**, and why? Update `df_model` accordingly:

```python
# YOUR CHOICE: for EACH redundant pair you found, drop ONE column (keep the more
# interpretable one) and justify it in Reflection A6.
# df_model = df_model.drop(columns=[ ... ])
print("Final feature columns:", list(df_model.columns))
```

---

## Task A7 — Encode & Frame Bias vs. Variance

```python
df_model = pd.get_dummies(df_model, columns=['sex','smoker','region'], drop_first=True)
display(df_model.head())
```

**Reflection A7**
1. What does `drop_first=True` do, and why does it avoid a new form of redundancy?
2. In your own words: what would **underfitting (high bias)** vs **overfitting (high variance)** look like for HealthWise's pricing model — give a business consequence of each?
3. If **training error is low but test error is high**, which problem is it, and what would you do?

---

## Task A8 — Automated Feature Selection with Lasso (confirm your audit) ⭐

> In A4 you dropped junk using **judgement**. Now let a machine check your work: **Lasso** shrinks useless coefficients to **exactly zero**. Run it on the **full feature set including the junk** to see whether it independently agrees with you.

```python
# Rebuild a feature set that STILL INCLUDES the suspected-junk columns, for this test only
audit = df.drop(columns=['customer_id','date_of_birth','annual_charge','risk_tier','age_from_dob'])
audit = pd.get_dummies(audit, columns=['sex','smoker','region','favorite_color',
                                       'zodiac_sign','preferred_contact','marketing_opt_in'],
                       drop_first=True)
feat_names = audit.columns
X, y = audit.values, df['annual_charge'].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
sc = StandardScaler().fit(Xtr)

for a in [10, 100, 400, 800]:
    l = Lasso(alpha=a, max_iter=50000).fit(sc.transform(Xtr), ytr)
    kept = [f for f, c in zip(feat_names, l.coef_) if abs(c) > 1]
    r2 = r2_score(yte, l.predict(sc.transform(Xte)))
    print(f"alpha={a:4} | R²={r2:.3f} | features kept ({len(kept)}): {kept}")
```

**Reflection A8**
1. As `alpha` increases, which features **survive** and which get **zeroed**? At a strong alpha, are any of your "junk" columns (`favorite_color`, `zodiac_sign`, `lucky_number`, etc.) still there?
2. Does R² **fall much** when all the junk is removed? What does that prove about those features?
3. Did the machine (Lasso) **agree** with your manual audit in A4? Note any disagreement.

*(From here on, use your cleaned `df_model` — junk and redundant columns removed.)*

---

# PART B — STAGE 1: Classify the Customer (Risk Tier)

## Task B1 — Build a Classifier

> Predict `risk_tier` (Low/Medium/High) from the profile.
> **`# YOUR CHOICE`:** pick a classifier you've studied (Logistic Regression, Decision Tree, …) and justify it.

```python
from sklearn.linear_model import LogisticRegression      # or DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_cls = df_model.drop(columns=['annual_charge','risk_tier'])
y_cls = df_model['risk_tier']
Xtr, Xte, ytr, yte = train_test_split(X_cls, y_cls, test_size=0.2,
                                      random_state=RANDOM_STATE, stratify=y_cls)

clf = LogisticRegression(max_iter=1000)   # YOUR CHOICE
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
```

**Reflection B1**
1. Which classifier did you choose and why?
2. Why do we use `stratify=y_cls` in the split?

---

## Task B2 — Evaluate the Classifier

```python
print("Accuracy:", round(accuracy_score(yte, pred), 3))
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred))
```

**Reflection B2**
1. What is your test accuracy?
2. From the confusion matrix, **which tier is most confused with which**?
3. **Business risk:** which is worse for HealthWise — labelling a truly **High**-risk customer as **Low**, or a **Low**-risk one as **High**? Explain the financial consequence of each.

---

## Task B3 — Bias/Variance Check

```python
print("Train accuracy:", round(accuracy_score(ytr, clf.predict(Xtr)), 3))
print("Test  accuracy:", round(accuracy_score(yte, pred), 3))
```

**Reflection B3**
1. Compare train vs. test accuracy — is there a large gap?
2. Is your classifier **overfitting, underfitting, or well-balanced**?

---

# PART C — STAGE 2: Predict the Charge (Regression per Tier)

## Task C1 — The Baseline: ONE Global Regression

```python
X_reg = df_model.drop(columns=['annual_charge','risk_tier'])
y_reg = df_model['annual_charge']
Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)

global_model = LinearRegression().fit(Xtr, ytr)
gp = global_model.predict(Xte)
print("GLOBAL model")
print("  MAE :", round(mean_absolute_error(yte, gp)))
print("  RMSE:", round(np.sqrt(mean_squared_error(yte, gp))))
print("  R²  :", round(r2_score(yte, gp), 3))
```

**Reflection C1**
1. Record the global MAE, RMSE, and R² — this is your **benchmark**.
2. How good does one model for everyone look so far?

---

## Task C2 — Simple vs. Multiple Regression (within a tier)

> Pick **one tier** (start with `High`) and build two models on the same split:
> **simple** (single strongest feature) vs **multiple** (all features).

```python
tier = 'High'      # YOUR CHOICE: repeat for other tiers
d = df_model[df_model['risk_tier']==tier].drop(columns=['risk_tier'])
Xd, yd = d.drop(columns=['annual_charge']), d['annual_charge']
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)

single = '____'    # YOUR CHOICE: pick the single strongest driver for this tier
                   # (decide it from your EDA / correlations — do not guess)
simple = LinearRegression().fit(Xtr[[single]], ytr)
multi  = LinearRegression().fit(Xtr, ytr)
print(f"[{tier}] SIMPLE ({single}) R²:", round(r2_score(yte, simple.predict(Xte[[single]])), 3))
print(f"[{tier}] MULTIPLE        R²:", round(r2_score(yte, multi.predict(Xte)), 3))
```

**Reflection C2**
1. How much did R² improve from **one feature** to **all features** for this tier?
2. Repeat for the `Low` tier and **compare the size of the jump**. Is it bigger for High-risk or Low-risk customers — and **why**?
3. Higher R² with more features — but does *every* added feature deserve credit? How would you check? (Lasso, next.)

---

## Task C3 — Regression WITHIN EACH Tier (the two-stage engine)

```python
tiers = ['Low','Medium','High']
overall_true, overall_pred = [], []
for t in tiers:
    d = df_model[df_model['risk_tier']==t].drop(columns=['risk_tier'])
    Xd, yd = d.drop(columns=['annual_charge']), d['annual_charge']
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)
    m = LinearRegression().fit(Xtr, ytr); p = m.predict(Xte)
    overall_true.extend(yte); overall_pred.extend(p)
    print(f"[{t:6}] R²={r2_score(yte,p):.3f}  MAE={mean_absolute_error(yte,p):,.0f}  (n={len(d)})")

print("\nTWO-STAGE ENGINE (all tiers combined)")
print("  MAE:", round(mean_absolute_error(overall_true, overall_pred)))
print("  R² :", round(r2_score(overall_true, overall_pred), 3))
```

**Reflection C3**
1. Fill in with **your** numbers:

   | Model | MAE | R² |
   |---|---|---|
   | Global (C1) | | |
   | Two-stage per-tier (C3) | | |

2. **Board question 1:** *Does the two-stage approach beat the single global model?* Quote your MAE improvement.
3. Which tier fits **best** and which **worst**? What does that say about how predictable each customer type is?

---

## Task C4 — Ridge & Lasso + Fine-Tuning (per tier)

```python
tier = 'High'      # YOUR CHOICE
d = df_model[df_model['risk_tier']==tier].drop(columns=['risk_tier'])
Xd, yd = d.drop(columns=['annual_charge']), d['annual_charge']
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(Xtr); Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

alphas = [0.01, 0.1, 1, 10, 50, 100, 200]
print("alpha :  Ridge R²   Lasso R²")
for a in alphas:
    r = Ridge(alpha=a).fit(Xtr_s, ytr)
    l = Lasso(alpha=a, max_iter=10000).fit(Xtr_s, ytr)
    print(f"{a:6} : {r2_score(yte, r.predict(Xte_s)):.3f}     {r2_score(yte, l.predict(Xte_s)):.3f}")

best_alpha = 10    # YOUR CHOICE from the sweep
lasso = Lasso(alpha=best_alpha, max_iter=10000).fit(Xtr_s, ytr)
coefs = pd.Series(lasso.coef_, index=Xd.columns).sort_values(key=abs, ascending=False)
print("\nLasso coefficients:\n", coefs.round(1))
```

**Reflection C4**
1. Which `alpha` gave the best test R² for Ridge and Lasso? What happens when alpha gets **too large**?
2. Among the **real** features, which are the strongest drivers of cost in this tier? Does `region` matter much?
3. Why is regularisation useful even after you've already removed the obvious junk?

---

## Task C5 — The Verdict

**Reflection C5**
1. Complete the **model ladder** (High tier, or averaged across tiers):

   | Model | R² | What it tells you |
   |---|---|---|
   | Simple linear (1 feature) | | |
   | Multiple linear (all features) | | |
   | Ridge (tuned) | | |
   | Lasso (tuned) | | |

2. Where was the **biggest** gain — cleaning out junk features, adding features, splitting into tiers, or tuning alpha? What does that reveal about this problem?

---

# PART D — Business Recommendation

Write a short **memo to the HealthWise board** (150–250 words).

**Reflection D — The Board Memo**
1. **Does the two-stage engine beat the flat model?** Cite your MAE/R² improvement (C3).
2. **Which attributes drive cost, and which are worthless?** Name the top 3 drivers, and mention the columns you dropped as irrelevant.
3. **Recommendation:** deploy or not? Note **one risk or limitation** to watch for.

---

# PART E — Deploy: Build the Interactive Premium Calculator (GUI) ⭐

Now turn your two-stage engine into a **working application**. A HealthWise staff member should be able to type in a customer's details, see the **predicted risk tier** and **estimated premium**, and — crucially — **change a value (e.g. add a child, or switch smoker to "yes") and watch the premium recalculate instantly.**

> 💡 **How the app works (the two-stage logic lives inside it):** when the user enters/changes a value, the app first runs **Stage 1** (classify the tier), then feeds the customer to **that tier's Stage-2 regressor** to compute the premium. Change smoker to "yes" and the customer may jump to a higher tier → a different regressor → a higher premium. That is your engine in action.

## Task E1 — Save your trained two-stage model

An app can't retrain every time it opens — so save your trained models to a file the app will load. Use an sklearn **`Pipeline`** (encoder + model) so that **single-customer inputs are encoded correctly** (a plain `get_dummies` on one row silently breaks one-hot columns — avoid it here).

```python
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

NUM = ['age','bmi','children','exercise_freq']      # your kept numeric features
CAT = ['sex','smoker','region']                     # your kept categorical features
FEATURES = NUM + CAT
med = {c: float(df[c].median()) for c in NUM}       # for filling any missing input

pre = ColumnTransformer([('num','passthrough',NUM),
                         ('cat',OneHotEncoder(handle_unknown='ignore',drop='first'),CAT)])

# Stage 1 — classifier (whole dataset)
clf = Pipeline([('pre',pre),('model',DecisionTreeClassifier(max_depth=5,random_state=RANDOM_STATE))])
clf.fit(df[FEATURES], df['risk_tier'])

# Stage 2 — one regressor per tier
regs = {}
for t in ['Low','Medium','High']:
    sub = df[df['risk_tier']==t]
    regs[t] = Pipeline([('pre',pre),('model',LinearRegression())]).fit(sub[FEATURES], sub['annual_charge'])

joblib.dump({'clf':clf,'regs':regs,'features':FEATURES,'num':NUM,'cat':CAT,'med':med},
            'healthwise_model.joblib')
print("Saved healthwise_model.joblib")
```

**Reflection E1**
1. Why must the **same encoder** be used at training time and at prediction time?
2. Why do we save **three regressors** (one per tier) plus **one classifier**, rather than a single model?

## Task E2 — Build the app

Choose **one** option. Both call the same two-stage `predict()` function.

**The prediction function (used by either option):**
```python
import joblib, pandas as pd
M = joblib.load('healthwise_model.joblib')
clf, regs, FEATURES, NUM, med = M['clf'], M['regs'], M['features'], M['num'], M['med']

def predict(customer: dict):
    row = pd.DataFrame([customer])
    for k in NUM:                                   # fill anything the user left blank
        if k not in row or pd.isna(row.at[0,k]): row[k] = med[k]
    tier = clf.predict(row[FEATURES])[0]            # Stage 1
    premium = float(regs[tier].predict(row[FEATURES])[0])   # Stage 2 (that tier's model)
    return tier, round(premium, 2)
```

**Option A — Streamlit** (recommended; a real web app). Save as `app.py`, run with `streamlit run app.py`:
```python
import streamlit as st, pandas as pd, joblib
M = joblib.load('healthwise_model.joblib')
clf, regs, FEATURES, NUM, med = M['clf'], M['regs'], M['features'], M['num'], M['med']

def predict(c):
    row=pd.DataFrame([c])
    for k in NUM:
        if k not in row or pd.isna(row.at[0,k]): row[k]=med[k]
    t=clf.predict(row[FEATURES])[0]
    return t, float(regs[t].predict(row[FEATURES])[0])

st.title("🏥 HealthWise — Smart Premium Engine")
age      = st.slider("Age", 18, 64, 40)
bmi      = st.slider("BMI", 16.0, 50.0, 27.0, 0.1)
children = st.slider("Children", 0, 4, 1)
exercise = st.slider("Exercise/week", 0, 7, 3)
sex      = st.selectbox("Sex", ["male","female"])
smoker   = st.selectbox("Smoker", ["no","yes"])
region   = st.selectbox("Region", ["north","south","east","west"])

tier, premium = predict(dict(age=age,bmi=bmi,children=children,exercise_freq=exercise,
                             sex=sex,smoker=smoker,region=region))
st.metric("Predicted Risk Tier", tier)
st.metric("Estimated Annual Premium", f"${premium:,.0f}")

# BONUS: let staff upload a CSV of unseen customers for batch scoring
up = st.file_uploader("Score unseen customers (CSV)", type="csv")
if up:
    new = pd.read_csv(up)
    out = [predict(dict(r)) for _, r in new.iterrows()]
    new["predicted_tier"]=[t for t,_ in out]; new["predicted_premium"]=[round(p) for _,p in out]
    st.dataframe(new); st.download_button("Download", new.to_csv(index=False), "predictions.csv")
```

**Option B — ipywidgets** (stays inside your Jupyter/Colab notebook):
```python
import ipywidgets as widgets
from IPython.display import display

w = dict(age=widgets.IntSlider(40,18,64,description='Age'),
         bmi=widgets.FloatSlider(27,16,50,description='BMI'),
         children=widgets.IntSlider(1,0,4,description='Children'),
         exercise_freq=widgets.IntSlider(3,0,7,description='Exercise'),
         sex=widgets.Dropdown(options=['male','female'],description='Sex'),
         smoker=widgets.Dropdown(options=['no','yes'],description='Smoker'),
         region=widgets.Dropdown(options=['north','south','east','west'],description='Region'))
out = widgets.Output()
def update(_=None):
    with out:
        out.clear_output()
        t,p = predict({k:x.value for k,x in w.items()})
        print(f"Predicted tier: {t}   |   Estimated premium: ${p:,.0f}")
for x in w.values(): x.observe(update,'value')
display(*w.values(), out); update()
```

**Reflection E2**
1. Screenshot (or paste the output of) your app showing a prediction.
2. In your app, start from a **Low-risk** customer and switch **smoker → yes**. What happens to the **tier** and the **premium**? Explain *why*, using the two-stage logic.
3. Which single change moves the premium the **most** — smoker, age, or bmi? Does that match what your Stage-2 coefficients (Task C4) said?

## Task E3 — Score the unseen customers ⭐

Your instructor will give you a small set of **unseen customers** (or a `test_customers.csv`). For each one, use your app to report the **predicted tier** and **premium**. Then, for at least one customer, **adjust a parameter your instructor names** (e.g. *"make this customer a smoker"*, or *"add two children"*) and report the **updated premium**.

| Customer | Predicted tier | Premium | Change applied | Updated premium |
|---|---|---|---|---|
| 1 | | | e.g. smoker → yes | |
| 2 | | | e.g. children +2 | |
| … | | | | |

**Reflection E3**
1. Did the predictions look **reasonable** for each unseen customer? Any surprises?
2. For the adjusted customer — did the premium move in the direction and by roughly the magnitude you expected? What does this tell the business about that risk factor?

---

# ⭐ Bonus Tasks (optional — up to +2 marks)

**Bonus 1 — Error propagation.** Stage 2 trusts Stage 1's tier, but B2 showed misclassifications.
- *Reflection:* If a truly **High**-risk customer is misclassified as **Low** and priced by the Low-tier model, what happens to their premium and to HealthWise's finances? What safeguard would you add?

**Bonus 2 — When would you use PCA?** This dataset had few *useful* features, so we used VIF + Lasso, not PCA.
- *Reflection (3 sentences):* When **would** you reach for PCA instead, and what trade-off does it force you to accept?

---

# 📋 Marking Rubric (Total: 35)

| Criterion | Marks |
|---|---|
| **A. Data cleaning** (duplicates, missing values, justified imputation) | 3 |
| **B. Feature audit — dropping useless & redundant columns** (reasoning + evidence in A4; VIF redundancy in A6; zodiac trap understood) | 4 |
| **C. EDA & hypothesis testing** (skew, plots, correct H₀/H₁ and p-value reading) | 3 |
| **D. Automated feature selection with Lasso** (A8: reads which features survive; agrees/disagrees with manual audit) | 3 |
| **E. Stage-1 classification** (built, evaluated, confusion matrix, business-risk reasoning) | 3 |
| **F. Simple vs. Multiple regression** (both built, gap quantified, cross-tier insight) | 3 |
| **G. Two-stage engine vs. global model** (per-tier models, correct comparison, board question 1 answered with numbers) | 4 |
| **H. Ridge/Lasso & fine-tuning** (alpha sweep, feature importance read correctly) | 2 |
| **I. Business recommendation memo** (evidence-based, both board questions, limitation noted) | 2 |
| **J. GUI app** (model saved; app runs; predicts tier + premium; **recalculates live when a parameter changes**) | 5 |
| **K. Unseen-data scoring** (E3: correct predictions on instructor's unseen customers + updated premium after a parameter change) | 2 |
| **L. Quality of reflections** (own interpretation; all cells executed with visible output) | 1 |
| **Bonus** (error propagation and/or PCA reasoning) | +2 |

> *Prefer a /30 scheme? Treat Parts J+K (the GUI, 7 marks) as a separate practical grade, or as bonus, and the analysis stays at /28–30.*

---

## ✅ Submission Checklist

- [ ] Every code cell **run** with **output visible** (unexecuted notebooks lose marks).
- [ ] All **Reflection** questions answered in markdown cells, in your own words.
- [ ] You **audited and dropped** the useless and redundant columns, with justification.
- [ ] You used the **provided `healthwise.csv`** and a single fixed `random_state`.
- [ ] The **Board Memo (Part D)** is included.
- [ ] Your **saved model** (`healthwise_model.joblib`) and **app** (`app.py` or the ipywidgets cell) are included and **run**.
- [ ] Your **GUI recalculates the premium live** when a parameter (e.g. smoker, children) is changed.
- [ ] The **unseen-customer results table (Task E3)** is filled in.
- [ ] Files named: `HealthWise_<YourName>_<RollNo>.ipynb` (+ `app.py` if used).

> **Academic integrity:** Individual work on a shared dataset. Copied notebooks, or notebooks submitted without outputs, will be flagged. Your **reflections and feature-audit decisions must be your own** — that is where the real marks are.
