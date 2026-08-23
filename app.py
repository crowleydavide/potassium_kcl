import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(
    page_title="Avocado Potassium Decision Support",
    page_icon="🥑",
    layout="wide",
)

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "yield_model.joblib"
METADATA_PATH = "model_metadata.json"

MODEL_HIGH_YIELD_K_LOW = 0.72
MODEL_HIGH_YIELD_K_HIGH = 1.00
GENERAL_K_SUFF_LOW = 0.75
GENERAL_K_SUFF_HIGH = 2.00

# Chloride interpretation used by the K × Cl adjunct.
# Chloride zones are based on avocado leaf-analysis context developed in the
# research versions; the K modifier remains an observational research finding.
CL_BACKGROUND_HIGH = 0.25
CL_ELEVATED_HIGH = 0.40
CL_HIGH_HIGH = 0.50

# Soft K interaction zones. These DO NOT redefine K sufficiency.
K_CL_TRANSITION_LOW = 1.10
K_CL_TRANSITION_HIGH = 1.20

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 1220px; padding-top: 1.3rem; padding-bottom: 3rem;}
    .hero {
        background: linear-gradient(135deg,#173f2a,#3f7d44);
        color:white; padding:28px 32px; border-radius:22px;
        margin-bottom:18px;
        box-shadow: 0 12px 34px rgba(23,63,42,.13);
    }
    .hero h1 {margin:0; font-size:2.35rem;}
    .hero p {opacity:.93; margin:.55rem 0 0; max-width:900px; line-height:1.55;}
    .notice {
        background:#fff8df; border:1px solid #eadca8; color:#66531d;
        padding:13px 15px; border-radius:13px; margin-bottom:18px;
    }
    .result {
        background:#f1f8ed; border:1px solid #bdd7ac;
        padding:18px; border-radius:16px; margin-top:10px;
    }
    .warningbox {
        background:#fff2e6; border:1px solid #efc196;
        padding:16px; border-radius:14px;
    }
    .goodbox {
        background:#edf7ec; border:1px solid #bedbbd;
        padding:16px; border-radius:14px;
    }
    .bluebox {
        background:#edf4fb; border:1px solid #c4d7eb;
        padding:16px; border-radius:14px;
    }
    .redbox {
        background:#faecec; border:1px solid #e7bcbc;
        padding:16px; border-radius:14px;
    }
    .kcl-panel {
        margin-top:16px;
        padding:17px 18px;
        border-radius:15px;
        line-height:1.5;
    }
    .kcl-green {
        background:#edf7ec;
        border:1px solid #bedbbd;
    }
    .kcl-blue {
        background:#edf4fb;
        border:1px solid #c4d7eb;
    }
    .kcl-yellow {
        background:#fff8df;
        border:1px solid #ead28a;
    }
    .kcl-orange {
        background:#fff0e5;
        border:1px solid #edbd91;
    }
    .kcl-red {
        background:#faecec;
        border:1px solid #e7bcbc;
    }
    .kcl-title {
        font-weight:800;
        font-size:1.02rem;
        margin-bottom:5px;
    }
    .small {font-size:.86rem; color:#5c685f; line-height:1.45;}
    div[data-testid="stMetric"] {
        background:#fbfcfa; border:1px solid #e0e7de;
        padding:10px 14px; border-radius:14px;
    }
    </style>
    <div class="hero">
      <h1>🥑 Avocado Potassium Decision Support</h1>
      <p>
        Uses the trained avocado yield model to vary leaf potassium while holding the
        grower's other leaf nutrients constant, then combines that model response
        with soil K, CEC, texture, salinity and root-zone information.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
    <b>Research decision-support prototype.</b>
    Model predictions and the K × Cl interaction are associations in historical
    avocado data, not proof that changing fertilizer will cause the predicted response.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_assets()
except Exception as e:
    st.error(
        "The trained model files could not be loaded. Place this file in the same "
        "folder as `yield_model.joblib` and `model_metadata.json`."
    )
    st.code(str(e))
    st.stop()

features = list(meta["features"])
defaults = meta.get("defaults", {})

def clean_name(name):
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace(".", "")
    )

def find_feature(candidates):
    normalized = {clean_name(f): f for f in features}
    for cand in candidates:
        c = clean_name(cand)
        for norm, original in normalized.items():
            if norm == c:
                return original
    for cand in candidates:
        c = clean_name(cand)
        for norm, original in normalized.items():
            if c in norm or norm in c:
                return original
    return None

K_FEATURE = find_feature(["Potassium (%)", "Potassium", "K (%)", "K(%)", "K"])
MG_FEATURE = find_feature(["Magnesium (%)", "Magnesium", "Mg (%)", "Mg(%)", "Mg"])
CA_FEATURE = find_feature(["Calcium (%)", "Calcium", "Ca (%)", "Ca(%)", "Ca"])
CL_FEATURE = find_feature(["Chloride (%)", "Chloride", "Cl (%)", "Cl(%)", "Cl"])

if K_FEATURE is None:
    st.error("Could not identify the potassium feature in model_metadata.json.")
    st.write("Features found:", features)
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def predict_yield(values_by_feature):
    x = np.array([[float(values_by_feature[f]) for f in features]], dtype=float)
    return max(0.0, float(model.predict(x)[0]))

def yield_potential(pred):
    ref = float(meta.get("yield_practical_max_95th_percentile", 0.0))
    if ref <= 0:
        return np.nan
    return min(100.0, 100.0 * pred / ref)

def soil_k_class(soil_k, cec, clay):
    if cec < 10 or clay < 10:
        low, high = 100.0, 200.0
    elif cec <= 20 or clay < 25:
        low, high = 150.0, 300.0
    else:
        low, high = 200.0, 400.0

    if soil_k < low:
        level = "low"
    elif soil_k > high:
        level = "high"
    else:
        level = "adequate"
    return level, low, high

def nutrient_step(feature, default):
    name = feature.lower()
    if "(%)" in feature or "%" in feature:
        return 0.01
    if "ppm" in name:
        return 1.0
    return max(abs(default) * 0.02, 0.01)

def k_status_for_panel(current_k):
    """
    Uses the advisor's existing K interpretation rather than inventing a new
    sufficiency classification for the chloride interaction panel.
    """
    if current_k < MODEL_HIGH_YIELD_K_LOW:
        return "low"
    if current_k <= MODEL_HIGH_YIELD_K_HIGH:
        return "model_high_yield"
    if current_k <= GENERAL_K_SUFF_HIGH:
        return "sufficient"
    return "high"

def potassium_chloride_panel(current_k, leaf_cl, base_k_status):
    """
    Separate chloride-aware modifier. It never overrides the normal K diagnosis.
    It answers only whether elevated Cl should change how aggressively K is pushed.
    """
    if leaf_cl < CL_BACKGROUND_HIGH:
        return {
            "css": "kcl-green",
            "title": "Potassium × Chloride: no interaction concern",
            "message": (
                "Chloride is below the usual excess range. Interpret potassium using "
                "the normal potassium guidance."
            ),
            "detail": (
                "No chloride-related restriction on potassium is suggested at this level."
            ),
        }

    # Low K gets special handling: preserve normal K correction, but don't overcorrect.
    if base_k_status == "low":
        if leaf_cl < CL_ELEVATED_HIGH:
            return {
                "css": "kcl-yellow",
                "title": "Elevated chloride with low K",
                "message": (
                    "Potassium is below the model high-yield range, so a genuine K shortage "
                    "should still be evaluated and corrected when supported by the soil/root-zone diagnosis."
                ),
                "detail": (
                    "However, do not raise K beyond the normal target range in an attempt "
                    "to compensate for chloride. Manage the chloride/salinity source in parallel."
                ),
            }
        if leaf_cl < CL_HIGH_HIGH:
            return {
                "css": "kcl-orange",
                "title": "High chloride with low K",
                "message": (
                    "Potassium is low enough that its normal deficiency diagnosis still matters, "
                    "but chloride management should now be a major priority."
                ),
                "detail": (
                    "Correct a demonstrated K shortage appropriately, but do not overcorrect K "
                    "or use extra K as a treatment for chloride stress."
                ),
            }
        return {
            "css": "kcl-red",
            "title": "Chloride toxicity concern with low K",
            "message": (
                "Leaf chloride is in a range associated with chloride injury. The normal K diagnosis "
                "should still be respected if K is truly deficient."
            ),
            "detail": (
                "Address the chloride/salinity source as the primary stress and correct K only toward "
                "its normal target—not beyond it as a presumed chloride remedy."
            ),
        }

    # K in normal/model-high-yield range
    if base_k_status == "model_high_yield":
        if leaf_cl < CL_ELEVATED_HIGH:
            return {
                "css": "kcl-yellow",
                "title": "Elevated chloride",
                "message": (
                    "Potassium is already within the model high-yield range."
                ),
                "detail": (
                    "Focus on chloride/salinity management rather than pushing K above its current target range."
                ),
            }
        if leaf_cl < CL_HIGH_HIGH:
            return {
                "css": "kcl-orange",
                "title": "High chloride — maintain normal K target",
                "message": (
                    "Potassium is already within the model high-yield range while chloride is high."
                ),
                "detail": (
                    "Maintain appropriate K nutrition, but do not increase K above the normal target "
                    "in an attempt to compensate for chloride stress."
                ),
            }
        return {
            "css": "kcl-red",
            "title": "Chloride toxicity concern",
            "message": (
                "Leaf chloride is in a range associated with chloride injury, while K is already "
                "within the model high-yield range."
            ),
            "detail": (
                "Address chloride/salinity first. Additional K is unlikely to correct this yield limitation."
            ),
        }

    # Sufficient / above model high-yield K range
    if base_k_status == "sufficient":
        if current_k < K_CL_TRANSITION_LOW:
            k_phrase = "Potassium is sufficient and only slightly above the model high-yield range."
            strength = "possible"
        elif current_k <= K_CL_TRANSITION_HIGH:
            k_phrase = "Potassium is sufficient and in the K × Cl transition range."
            strength = "transition"
        else:
            k_phrase = "Potassium is already relatively high."
            strength = "strong"

        if leaf_cl < CL_ELEVATED_HIGH:
            if strength == "possible":
                return {
                    "css": "kcl-yellow",
                    "title": "Elevated chloride",
                    "message": k_phrase,
                    "detail": (
                        "There is no reason to push K higher solely because of chloride. "
                        "Evaluate chloride/salinity management first."
                    ),
                }
            if strength == "transition":
                return {
                    "css": "kcl-yellow",
                    "title": "Possible K × Cl interaction",
                    "message": (
                        "Chloride is elevated and potassium is in a transition range where "
                        "the historical data begin to show less benefit from higher K."
                    ),
                    "detail": (
                        "Do not assume that additional potassium will improve yield. "
                        "Evaluate chloride and salinity before pushing K higher."
                    ),
                }
            return {
                "css": "kcl-orange",
                "title": "K × Cl interaction caution",
                "message": (
                    "Chloride is elevated and potassium is already relatively high."
                ),
                "detail": (
                    "In the historical avocado dataset, higher K under elevated chloride was "
                    "associated with lower yield than moderate K. Additional K may therefore "
                    "not improve yield. Evaluate chloride/salinity first."
                ),
            }

        if leaf_cl < CL_HIGH_HIGH:
            return {
                "css": "kcl-orange",
                "title": "High chloride — potassium caution",
                "message": k_phrase,
                "detail": (
                    "Chloride management should take priority. Maintain adequate K nutrition, "
                    "but do not push K higher as a presumed correction for chloride stress."
                ),
            }

        return {
            "css": "kcl-red",
            "title": "Chloride toxicity concern — do not push K higher",
            "message": (
                "Leaf chloride is in a range associated with chloride injury and potassium is already sufficient."
            ),
            "detail": (
                "Increasing K is unlikely to correct the chloride-related yield limitation. "
                "Address chloride and salinity management first."
            ),
        }

    # Very high K
    if leaf_cl < CL_ELEVATED_HIGH:
        return {
            "css": "kcl-orange",
            "title": "High K with elevated chloride",
            "message": (
                "Potassium is already above the general sufficiency range and chloride is elevated."
            ),
            "detail": (
                "Avoid additional K and review cation balance, fertilizer inputs, and chloride/salinity management."
            ),
        }

    return {
        "css": "kcl-red",
        "title": "High K plus high chloride",
        "message": (
            "Both potassium and chloride warrant attention."
        ),
        "detail": (
            "Avoid additional K. Prioritize chloride/salinity management and review Mg, Ca, "
            "soil K, and the current fertilizer program."
        ),
    }

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("1. Leaf nutrient profile")
    st.caption(
        "Enter the laboratory leaf values. Potassium will be scanned across a range "
        "while all other nutrients stay at the values entered here."
    )

    nutrient_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        d = float(defaults.get(feature, 0.0))
        with cols[i % 3]:
            nutrient_values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=d,
                step=nutrient_step(feature, d),
                format="%.4f" if "(%)" in feature or "%" in feature else "%.2f",
                key=f"nutrient_{i}",
            )

    if CL_FEATURE is None:
        st.info(
            "The loaded model metadata does not contain a chloride feature. "
            "Enter leaf chloride below so the K × Cl advisory panel can still be used."
        )
        leaf_cl_manual = st.number_input(
            "Leaf chloride, Cl (%)",
            min_value=0.0,
            value=0.25,
            step=0.01,
            format="%.3f",
            key="manual_leaf_cl",
        )
    else:
        leaf_cl_manual = None

    st.divider()
    st.subheader("2. Soil and root-zone information")

    c1, c2, c3 = st.columns(3)
    with c1:
        soil_k = st.number_input("Soil K (ppm)", min_value=0.0, value=220.0, step=5.0)
        soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.3, step=0.1)
    with c2:
        cec = st.number_input("CEC (meq/100 g)", min_value=0.0, value=15.0, step=0.5)
        ece = st.number_input("ECe (dS/m)", min_value=0.0, value=0.7, step=0.1)
    with c3:
        clay = st.number_input("Clay (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        sar = st.number_input("SAR", min_value=0.0, value=1.5, step=0.1)

    root_condition = st.selectbox(
        "Root / drainage condition",
        [
            "Good drainage / healthy roots",
            "Questionable drainage or root health",
            "Poor drainage / known root problems",
        ],
    )

    k_program = st.selectbox(
        "Current potassium fertilizer program",
        ["Low / none", "Moderate", "High"],
        index=1,
    )

    st.divider()
    st.subheader("3. Potassium model scan")

    scan1, scan2, scan3 = st.columns(3)
    current_k = float(nutrient_values[K_FEATURE])

    with scan1:
        k_min = st.number_input(
            "Scan K minimum (%)",
            min_value=0.1,
            max_value=3.0,
            value=0.40,
            step=0.05,
        )
    with scan2:
        k_max = st.number_input(
            "Scan K maximum (%)",
            min_value=0.2,
            max_value=4.0,
            value=2.00,
            step=0.05,
        )
    with scan3:
        points = st.number_input(
            "Curve points",
            min_value=25,
            max_value=250,
            value=101,
            step=10,
        )

    analyze = st.button(
        "Analyze potassium response",
        type="primary",
        use_container_width=True,
    )

with right:
    st.subheader("Potassium diagnosis")

    if not analyze:
        st.info("Enter the orchard values and select **Analyze potassium response**.")
        st.stop()

    if k_max <= k_min:
        st.error("K scan maximum must be greater than the minimum.")
        st.stop()

    current_pred = predict_yield(nutrient_values)
    current_potential = yield_potential(current_pred)

    k_grid = np.linspace(float(k_min), float(k_max), int(points))
    scan_yield = []

    for k in k_grid:
        row = dict(nutrient_values)
        row[K_FEATURE] = float(k)
        scan_yield.append(predict_yield(row))

    scan_yield = np.asarray(scan_yield, dtype=float)

    best_idx = int(np.argmax(scan_yield))
    best_k = float(k_grid[best_idx])
    best_pred = float(scan_yield[best_idx])
    best_potential = yield_potential(best_pred)

    gain_kg = best_pred - current_pred
    gain_pct_points = (
        best_potential - current_potential
        if np.isfinite(best_potential) and np.isfinite(current_potential)
        else np.nan
    )

    threshold = 0.95 * best_pred
    near = k_grid[scan_yield >= threshold]
    if len(near):
        individualized_low = float(near.min())
        individualized_high = float(near.max())
    else:
        individualized_low = best_k
        individualized_high = best_k

    sk_level, sk_low, sk_high = soil_k_class(soil_k, cec, clay)

    if current_k < MODEL_HIGH_YIELD_K_LOW:
        if sk_level == "low":
            diagnosis = "Low K — soil supply limited"
            color_class = "warningbox"
            recommendation = (
                "Leaf K is below the model high-yield profile and soil K is also low "
                "for this soil's CEC/texture. Potassium application is reasonably "
                "supported, preferably through measured incremental additions followed "
                "by repeat leaf analysis."
            )
        else:
            diagnosis = "Low K — uptake limited"
            color_class = "warningbox"
            recommendation = (
                "Leaf K is low but soil K appears adequate. Do not assume that more K "
                "fertilizer is the solution. Investigate salinity, irrigation, drainage, "
                "root health, sodium and cation competition."
            )
    elif current_k <= MODEL_HIGH_YIELD_K_HIGH:
        diagnosis = "K within model high-yield profile"
        color_class = "goodbox"
        recommendation = (
            "Maintain the current K program unless the individualized model scan shows "
            "a meaningful improvement elsewhere. Avoid pushing K higher simply because "
            "the conventional sufficiency range is broader."
        )
    elif current_k <= GENERAL_K_SUFF_HIGH:
        diagnosis = "K sufficient — above model high-yield profile"
        color_class = "bluebox"
        recommendation = (
            "K deficiency is not indicated. Additional K is unlikely to be a priority. "
            "Use the individualized response curve to judge whether the model predicts "
            "any benefit from moving K upward or downward."
        )
    else:
        diagnosis = "High K / possible imbalance"
        color_class = "redbox"
        recommendation = (
            "Avoid additional K until the fertilizer program and nutrient balance are "
            "reviewed. Pay particular attention to magnesium and calcium."
        )

    if MG_FEATURE and float(nutrient_values[MG_FEATURE]) < 0.25 and current_k > 1.0:
        recommendation += (
            " Low leaf Mg together with elevated K strengthens concern about a K–Mg imbalance."
        )

    st.markdown(
        f"""
        <div class="{color_class}">
          <b>{diagnosis}</b><br>
          <span style="font-size:.92rem">{recommendation}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # NEW: Potassium × Chloride panel
    # -----------------------------
    leaf_cl = (
        float(nutrient_values[CL_FEATURE])
        if CL_FEATURE is not None
        else float(leaf_cl_manual)
    )
    base_k_status = k_status_for_panel(current_k)
    kcl = potassium_chloride_panel(current_k, leaf_cl, base_k_status)

    st.markdown(
        f"""
        <div class="kcl-panel {kcl['css']}">
          <div class="kcl-title">{kcl['title']}</div>
          <div>{kcl['message']}</div>
          <div style="margin-top:6px">{kcl['detail']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why am I seeing the K × Cl message?"):
        st.write(
            f"Your entered leaf values are **K = {current_k:.2f}%** and "
            f"**Cl = {leaf_cl:.3f}%**."
        )
        st.write(
            "In the avocado research dataset, the relationship between potassium and yield "
            "changed as chloride increased. The K × Cl signal appeared in each of the 2011, "
            "2012 and 2013 chloride datasets and remained after accounting for the other "
            "measured nutrients."
        )
        st.write(
            "This panel does **not** redefine the normal potassium sufficiency range. "
            "Its purpose is to prevent a recommendation to push K higher simply as a response "
            "to chloride stress."
        )
        st.caption(
            "The K × Cl finding is observational and does not prove that high potassium "
            "causes chloride injury. Orchard, rootstock, irrigation water, soil salinity, "
            "crop load and management may also contribute."
        )

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Current leaf K", f"{current_k:.2f}%")
        st.metric("Leaf chloride", f"{leaf_cl:.3f}%")
        st.metric("Current predicted yield", f"{current_pred:,.1f} kg")
        if np.isfinite(current_potential):
            st.metric("Current yield potential", f"{current_potential:.0f}%")
    with m2:
        st.metric("Model-best K for this profile", f"{best_k:.2f}%")
        st.metric("Predicted yield at model-best K", f"{best_pred:,.1f} kg")
        if np.isfinite(best_potential):
            st.metric("Yield potential at model-best K", f"{best_potential:.0f}%")

    st.markdown(
        f"""
        <div class="result">
        <b>Individualized near-optimum K region:</b>
        {individualized_low:.2f}–{individualized_high:.2f}% leaf K
        <br><span class="small">
        Defined here as K values producing at least 95% of the highest predicted yield
        across this user's K scan while all other entered leaf nutrients remain fixed.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if abs(gain_kg) < 1.0:
        st.caption(
            "The model predicts little practical yield difference between current K "
            "and the highest point in the selected K scan."
        )
    else:
        if np.isfinite(gain_pct_points):
            st.write(
                f"**Modeled opportunity from K alone:** {gain_kg:+.1f} kg "
                f"({gain_pct_points:+.1f} yield-potential percentage points)."
            )
        else:
            st.write(f"**Modeled opportunity from K alone:** {gain_kg:+.1f} kg.")

    chart_data = {
        "Leaf K (%)": k_grid,
        "Predicted yield (kg)": scan_yield,
    }

    try:
        import pandas as pd
        chart_df = pd.DataFrame(chart_data).set_index("Leaf K (%)")
        st.line_chart(chart_df, height=310, use_container_width=True)
    except Exception:
        st.write("K response points:")
        st.json(
            [
                {"K": round(float(k), 3), "Predicted yield": round(float(y), 2)}
                for k, y in zip(
                    k_grid[::max(1, len(k_grid)//20)],
                    scan_yield[::max(1, len(k_grid)//20)]
                )
            ]
        )

    st.caption(
        "The curve changes when any other leaf nutrient input changes because the "
        "trained model is predicting from the complete nutrient profile."
    )

    st.divider()
    st.subheader("Soil and uptake interpretation")

    if sk_level == "low":
        st.warning(
            f"Soil K is below the practical screening threshold for this soil "
            f"(approximately <{sk_low:.0f} ppm based on entered CEC/texture)."
        )
    elif sk_level == "high":
        st.info(
            f"Soil K is high relative to the practical screening band of "
            f"{sk_low:.0f}–{sk_high:.0f} ppm for this CEC/texture."
        )
    else:
        st.success(
            f"Soil K falls within the practical screening band of "
            f"{sk_low:.0f}–{sk_high:.0f} ppm for this CEC/texture."
        )

    flags = []

    if ece >= 1.3:
        flags.append(("🔴", "ECe is a significant avocado salinity concern and may impair nutrient/water uptake."))
    elif ece >= 0.8:
        flags.append(("🟠", "ECe is elevated for a salt-sensitive avocado root system."))
    else:
        flags.append(("🟢", "ECe is in a favorable low-salinity range."))

    if sar >= 6:
        flags.append(("🔴", "SAR is high and may impair soil structure, especially where clay content is substantial."))
    elif sar >= 3:
        flags.append(("🟠", "SAR is elevated; sodium effects deserve attention."))
    else:
        flags.append(("🟢", "SAR is low and unlikely to be a major constraint by itself."))

    if soil_ph > 7.2:
        flags.append(("🟠", "Soil pH is high and may reduce availability of several nutrients."))
    elif soil_ph < 5.5:
        flags.append(("🟠", "Soil pH is below the preferred range used by this tool."))
    else:
        flags.append(("🟢", "Soil pH is within a favorable avocado range."))

    if root_condition.startswith("Poor"):
        flags.append(("🔴", "Poor drainage/root health can strongly restrict K uptake even where soil K is adequate."))
    elif root_condition.startswith("Questionable"):
        flags.append(("🟠", "Questionable root health/drainage could contribute to poor K uptake."))
    else:
        flags.append(("🟢", "No obvious root/drainage constraint was entered."))

    if MG_FEATURE:
        mg = float(nutrient_values[MG_FEATURE])
        if mg < 0.25:
            flags.append(("🔴", "Leaf Mg is low; aggressive K fertilization could aggravate K–Mg imbalance."))
        elif mg < 0.40:
            flags.append(("🟠", "Leaf Mg is on the low side; monitor Mg if K is increased."))

    if CA_FEATURE:
        ca = float(nutrient_values[CA_FEATURE])
        if ca < 1.0:
            flags.append(("🟠", "Leaf Ca is low; review root function and cation balance before aggressively increasing K."))

    for icon, message in flags:
        st.write(f"{icon} {message}")

    st.divider()
    st.subheader("How to use this result")
    st.write(
        """
        The model answers **whether changing leaf K is associated with a different
        predicted yield for this nutrient profile**. The soil decision tree then asks
        whether a shift in leaf K is agronomically plausible through K fertilization
        or whether uptake constraints should be addressed first.

        The **Potassium × Chloride** panel is a separate modifier. It does not override
        a genuine K deficiency; it helps prevent over-applying K in an attempt to compensate
        for elevated chloride.
        """
    )

    st.caption(
        "The population high-yield profile in the original model had K Q1–Q3 of "
        "approximately 0.72–1.00%. The individualized region shown above is different: "
        "it is calculated directly from this grower's complete nutrient vector."
    )
