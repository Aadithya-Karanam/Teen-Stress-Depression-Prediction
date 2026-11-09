
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
DATA_PATH = BASE_DIR / "Teen_Stress.csv"

st.set_page_config(
    page_title="TeenWell | ML Dashboard",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#0B1120; color:#E5EAF2; font-family:Inter,sans-serif; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1450px; padding-top:2rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:#101A2B; border-right:1px solid #25334A; }
h1 { color:#F1F5F9 !important; font-size:2.25rem !important; font-weight:700 !important; letter-spacing:-1px; }
h2,h3 { color:#E8EEF7 !important; }
p,label,.stCaption { color:#A8B4C7; }
.metric-card { background:linear-gradient(145deg,#17243A,#121D30); border:1px solid #293B55; border-radius:14px; padding:20px; min-height:115px; }
.metric-label { font-size:.76rem; color:#91A3BA; margin-bottom:10px; }
.metric-value { font-size:1.55rem; font-weight:700; color:#F1F5F9; }
.metric-sub { font-size:.75rem; color:#43C6AC; margin-top:7px; }
.panel { background:#131F32; border:1px solid #293B55; border-radius:15px; padding:22px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#17243A; border-color:#34465F; border-radius:8px; }
.stButton > button,.stFormSubmitButton > button { background:#36B9A2; color:#071B20; border:none; border-radius:9px; min-height:44px; font-weight:700; }
.stButton > button:hover,.stFormSubmitButton > button:hover { background:#57D6BB; color:#071B20; }
hr { border-color:#26364C; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# Fail clearly instead of silently training inside the app.
missing_artifacts = [
    path.name for path in (MODEL_PATH, METRICS_PATH, DATA_PATH) if not path.exists()
]
if missing_artifacts:
    st.error("Required project files are missing: " + ", ".join(missing_artifacts))
    st.markdown(
        "Place `Teen_Stress.csv` beside `app.py`, then run "
        "`python train_model.py` once to create the saved model and metrics."
    )
    st.stop()

try:
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    metrics = load_metrics()
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load the saved project artifacts: {exc}")
    st.info("If the model file is outdated or corrupted, run `python train_model.py` again.")
    st.stop()


def slider_values(column):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    low, high = float(values.min()), float(values.max())
    default = float(values.median())
    if low == high:
        return low, high + 1.0, low
    return low, high, min(max(default, low), high)


with st.sidebar:
    st.markdown("<h2 style='color:#43C6AC;'>✦ TeenWell</h2>", unsafe_allow_html=True)
    st.caption("TEEN STRESS & DEPRESSION ML PROJECT")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Assessment", "Model Analytics"])
    st.divider()
    st.markdown("#### PROJECT INFORMATION")
    st.caption("Machine Learning")
    st.caption("Random Forest Classifier")
    st.caption("GridSearchCV · 5-fold CV")
    st.divider()
    st.caption("Educational project · Not a diagnostic tool")

st.markdown(
    "<span style='color:#43C6AC;font-size:12px;letter-spacing:2px;font-weight:700;'>"
    "TEENWELL / MACHINE LEARNING</span>",
    unsafe_allow_html=True,
)

if page == "Overview":
    st.title("Overview")
    st.write("Explore the dataset and the saved machine-learning model.")
elif page == "Assessment":
    st.title("Lifestyle Assessment")
    st.write("Enter the requested information to receive the model's predicted dataset label.")
else:
    st.title("Model Analytics")
    st.write("Review saved model performance, classification results, and influential features.")

st.divider()

if page == "Overview":
    cols = st.columns(4)
    cards = [
        ("DATASET RECORDS", f"{metrics['dataset_records']:,}", "Rows used"),
        ("TEST ACCURACY", f"{metrics['test_accuracy'] * 100:.2f}%", "Held-out test set"),
        ("CROSS-VALIDATION", f"{metrics['cv_accuracy'] * 100:.2f}%", "Best 5-fold CV score"),
        ("FEATURES", str(metrics["feature_count"]), "Input variables"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Depression label distribution")
        counts = df["depression_label"].value_counts().reindex([0, 1], fill_value=0)
        counts.index = ["No depression (0)", "Depression (1)"]
        st.bar_chart(counts, color="#36B9A2")
        st.caption("Counts of the two target labels in the dataset.")
    with right:
        st.markdown("### Dataset information")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write(f"**Total records:** {metrics['dataset_records']:,}")
        st.write(f"**Input features:** {metrics['feature_count']}")
        st.write(f"**Missing values:** {metrics['missing_values']:,}")
        st.write("**Target:** depression_label")
        st.write("**0:** No · **1:** Yes")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### About TeenWell")
    st.markdown("""
    <div class="panel">
      <h3 style="color:#43C6AC;">Understanding lifestyle patterns through machine learning</h3>
      <p>TeenWell uses a Random Forest classification model to explore associations between
      social media usage, sleep, academic performance, physical activity, stress, anxiety
      and the dataset's depression label.</p>
      <p>The preprocessing pipeline and tuned model were trained separately and are loaded
      from saved artifacts. Opening this dashboard does not start model training.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Assessment":
    st.markdown("### Enter your information")
    st.caption("Input ranges are based on the values available in the dataset.")

    with st.form("assessment_form"):
        st.markdown("#### 01 / Personal information")
        c1, c2 = st.columns(2)
        with c1:
            lo, hi, default = slider_values("age")
            age = st.slider("Age", min_value=lo, max_value=hi, value=default)
        with c2:
            gender_options = sorted(df["gender"].dropna().unique().tolist())
            gender = st.selectbox("Gender", gender_options)

        st.divider()
        st.markdown("#### 02 / Digital lifestyle")
        c1, c2 = st.columns(2)
        with c1:
            lo, hi, default = slider_values("daily_social_media_hours")
            social_hours = st.slider("Daily social media hours", lo, hi, default)
            platform_options = sorted(df["platform_usage"].dropna().unique().tolist())
            platform = st.selectbox("Most used platform", platform_options)
        with c2:
            lo, hi, default = slider_values("screen_time_before_sleep")
            screen_time = st.slider("Screen time before sleep", lo, hi, default)
            interaction_options = sorted(df["social_interaction_level"].dropna().unique().tolist())
            interaction = st.selectbox("Social interaction level", interaction_options)

        st.divider()
        st.markdown("#### 03 / Lifestyle and well-being")
        c1, c2 = st.columns(2)
        with c1:
            lo, hi, default = slider_values("sleep_hours")
            sleep = st.slider("Sleep hours", lo, hi, default)
            lo, hi, default = slider_values("academic_performance")
            academic = st.slider("Academic performance", lo, hi, default)
            lo, hi, default = slider_values("physical_activity")
            physical = st.slider("Physical activity", lo, hi, default)
        with c2:
            lo, hi, default = slider_values("stress_level")
            stress = st.slider("Stress level", lo, hi, default)
            lo, hi, default = slider_values("anxiety_level")
            anxiety = st.slider("Anxiety level", lo, hi, default)
            lo, hi, default = slider_values("addiction_level")
            addiction = st.slider("Addiction level", lo, hi, default)

        submitted = st.form_submit_button("ANALYZE MY RESULTS", use_container_width=True)

    if submitted:
        # Apply exactly the same encoding used in train_model.py.
        user_input = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "daily_social_media_hours": social_hours,
            "platform_usage": platform,
            "sleep_hours": sleep,
            "screen_time_before_sleep": screen_time,
            "academic_performance": academic,
            "physical_activity": physical,
            "social_interaction_level": interaction,
            "stress_level": stress,
            "anxiety_level": anxiety,
            "addiction_level": addiction,
        }])

        user_input["gender"] = user_input["gender"].map({
            "male": 1,
            "female": 0
        })
        user_input["social_interaction_level"] = user_input[
            "social_interaction_level"
        ].map({
            "low": 0,
            "medium": 1,
            "high": 2
        })
        user_input = pd.get_dummies(
            user_input,
            columns=["platform_usage"],
            dtype=int
        )
        # Match the training feature names and their exact column order.
        user_input = user_input.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = int(model.predict(user_input)[0])
        probabilities = model.predict_proba(user_input)[0]
        probability_by_class = {
            int(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

        st.divider()
        st.markdown("## Your Assessment Results")
        if prediction == 0:
            st.success("No depression-associated label predicted")
            st.write(
                "The model predicted the negative class (0) from the information entered. "
                "This does not rule out depression or replace a professional assessment."
            )
        else:
            st.warning("Depression-associated pattern predicted")
            st.write(
                "The model predicted the positive class (1) based on patterns in its "
                "training data. This is not a diagnosis. If you are concerned about your "
                "well-being, consider speaking with a qualified mental health professional."
            )

        st.markdown("#### Model output probabilities")
        probability_df = pd.DataFrame({
            "Outcome": ["No depression label (0)", "Depression-associated label (1)"],
            "Model probability": [
                probability_by_class.get(0, 0.0),
                probability_by_class.get(1, 0.0),
            ],
        }).set_index("Outcome")
        st.bar_chart(probability_df, color="#36B9A2")
        st.caption(
            "These are model output probabilities, not clinical risk scores or a measure "
            "of the likelihood that someone has depression."
        )

else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Test accuracy")
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">HELD-OUT TEST ACCURACY</div>'
            f'<div class="metric-value">{metrics["test_accuracy"] * 100:.2f}%</div>'
            f'<div class="metric-sub">20% test split</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("### Cross-validation score")
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">BEST 5-FOLD CV ACCURACY</div>'
            f'<div class="metric-value">{metrics["cv_accuracy"] * 100:.2f}%</div>'
            f'<div class="metric-sub">Training-set cross-validation</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Confusion matrix")
    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual: No (0)", "Actual: Yes (1)"],
        columns=["Predicted: No (0)", "Predicted: Yes (1)"],
    )
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown("### Classification report")
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("### Feature importance")
    importance_df = pd.DataFrame(metrics["feature_importances"]).head(12).copy()
    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    st.bar_chart(importance_df.set_index("feature"), color="#36B9A2")

    with st.expander("View best hyperparameters"):
        st.json(metrics["best_params"])

    st.caption(f"Model trained (UTC): {metrics['trained_at_utc']}")

st.divider()
st.markdown("""
<div style="text-align:center;color:#65758B;font-size:12px;padding:15px;">
  TEENWELL · MACHINE LEARNING PROJECT<br>
  Educational and research purposes only · Not a clinical diagnostic tool
</div>
""", unsafe_allow_html=True)
