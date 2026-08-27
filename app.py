import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Avocado Potassium Decision Support v2", page_icon="🥑", layout="wide")

MODEL_PATH = "yield_model.joblib"
METADATA_PATH = "model_metadata.json"

MODEL_HIGH_YIELD_K_LOW = 0.72
MODEL_HIGH_YIELD_K_HIGH = 1.00
GENERAL_K_SUFF_HIGH = 2.00

CL_BACKGROUND_HIGH = 0.25
CL_ELEVATED_HIGH = 0.40
CL_HIGH_HIGH = 0.50
K_CL_TRANSITION_LOW = 1.10
K_CL_TRANSITION_HIGH = 1.20

FGL_EXCH_K2O_LOW = 320.0
FGL_EXCH_K2O_HIGH = 1900.0
FGL_SOL_K2O_LOW = 170.0
FGL_SOL_K2O_HIGH = 540.0
FGL_K_BASE_LOW = 1.0
FGL_K_BASE_HIGH = 6.0

st.markdown(
    '''
    <style>
    .block-container {max-width: 1220px; padding-top: 1.25rem; padding-bottom: 3rem;}
    .hero {background:linear-gradient(135deg,#173f2a,#3f7d44);color:white;padding:28px 32px;border-radius:22px;margin-bottom:18px}
    .hero h1 {margin:0;font-size:2.25rem}
    .hero p {opacity:.94;margin:.55rem 0 0;line-height:1.55}
    .notice {background:#fff8df;border:1px solid #eadca8;color:#66531d;padding:13px 15px;border-radius:13px;margin-bottom:18px}
    .box {padding:16px;border-radius:14px;margin-top:12px}
    .green {background:#edf7ec;border:1px solid #bedbbd}
    .blue {background:#edf4fb;border:1px solid #c4d7eb}
    .yellow {background:#fff8df;border:1px solid #ead28a}
    .orange {background:#fff0e5;border:1px solid #edbd91}
    .red {background:#faecec;border:1px solid #e7bcbc}
    .title {font-weight:800;font-size:1.02rem;margin-bottom:5px}
    </style>
    <div class="hero">
      <h1>🥑 Avocado Potassium Decision Support v2</h1>
      <p>Combines the trained avocado yield model with leaf K, chloride, root-zone conditions,
      and method-specific soil potassium measurements. For FGL reports, exchangeable K₂O and
      solution K₂O are interpreted separately.</p>
    </div>
    <div class="notice"><b>Research decision-support prototype.</b> Model predictions and the K × Cl
    interaction are associations in historical avocado data. Soil-test interpretation is method-specific.</div>
    ''',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_assets()
except Exception as e:
    st.error("Could not load model files. Keep app.py, yield_model.joblib and model_metadata.json together.")
    st.code(str(e))
    st.stop()

features = list(meta["features"])
defaults = meta.get("defaults", {})

def clean_name(name):
    return str(name).lower().replace(" ","").replace("_","").replace("-","").replace("(","").replace(")","").replace("%","").replace(".","")

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

K_FEATURE = find_feature(["Potassium (%)","Potassium","K (%)","K"])
MG_FEATURE = find_feature(["Magnesium (%)","Magnesium","Mg (%)","Mg"])
CA_FEATURE = find_feature(["Calcium (%)","Calcium","Ca (%)","Ca"])
CL_FEATURE = find_feature(["Chloride (%)","Chloride","Cl (%)","Cl"])

if K_FEATURE is None:
    st.error("Could not identify potassium in model_metadata.json.")
    st.stop()

def predict_yield(values_by_feature):
    x = np.array([[float(values_by_feature[f]) for f in features]], dtype=float)
    return max(0.0, float(model.predict(x)[0]))

@st.cache_resource
def ideal_nutrient_reference():
    try:
        vals = {f: float(defaults[f]) for f in features}
        ref = predict_yield(vals)
        if ref > 0:
            return ref, "ideal/default nutrient profile"
    except Exception:
        pass
    return float(meta.get("yield_practical_max_95th_percentile", 0.0)), "95th-percentile practical maximum"

YIELD_REF, YIELD_REF_LABEL = ideal_nutrient_reference()

def yield_potential(pred):
    if YIELD_REF <= 0:
        return np.nan
    return min(100.0, 100.0 * float(pred) / float(YIELD_REF))

def nutrient_step(feature, default):
    if "(%)" in feature or "%" in feature:
        return 0.01
    return max(abs(default)*0.02, 0.01)

def k_status(current_k):
    if current_k < MODEL_HIGH_YIELD_K_LOW:
        return "low"
    if current_k <= MODEL_HIGH_YIELD_K_HIGH:
        return "model_high_yield"
    if current_k <= GENERAL_K_SUFF_HIGH:
        return "sufficient"
    return "high"

def classify(value, low, high):
    if value < low: return "low"
    if value > high: return "high"
    return "adequate"

def soil_k_diagnosis(leaf_status, exch_k, sol_k, exch_low, exch_high, sol_low, sol_high):
    ex = classify(exch_k, exch_low, exch_high)
    so = classify(sol_k, sol_low, sol_high)

    if leaf_status == "low":
        if ex == "low" and so == "low":
            return "Strong soil K supply limitation", "Leaf K is low and both exchangeable and solution K are below the laboratory range.", "A measured K application is reasonably supported, followed by repeat testing.", "red", ex, so
        if ex != "low" and so == "low":
            return "Adequate K reserve — low solution K", "Leaf K is low, but exchangeable K is not low while solution K is low.", "Review irrigation, roots, moisture, salinity and cation competition before assuming a large increase in K fertilizer is required.", "orange", ex, so
        if ex != "low" and so != "low":
            return "Likely K uptake limitation", "Leaf K is low even though both exchangeable and solution K are adequate or high.", "Investigate roots, drainage, salinity, irrigation and cation competition before adding more K.", "orange", ex, so
        return "Mixed soil K signal", "Leaf K is low, but the two soil K pools do not indicate a simple deficiency.", "Interpret the lab method and root-zone conditions together.", "yellow", ex, so

    if so == "low" and ex != "low":
        return "Adequate leaf K despite low solution K", "Solution K is below the lab range, but exchangeable K is adequate and leaf K is not deficient.", "Do not diagnose K deficiency from solution K alone.", "blue", ex, so
    if ex == "low" and so == "low":
        return "Low soil K pools, but leaf K currently adequate", "Both soil K pools test low even though leaf K is presently adequate.", "Monitor the next leaf and soil tests rather than automatically pushing K higher.", "yellow", ex, so
    if ex != "low" and so != "low":
        return "Soil and leaf K currently adequate", "Leaf K is adequate and both soil K pools are within or above the laboratory range.", "Additional K is unlikely to be a priority unless the model scan indicates a meaningful benefit.", "green", ex, so
    return "Soil K reserve and availability differ", "Exchangeable and solution K fall into different categories while leaf K is not deficient.", "Use this as a monitoring signal rather than a stand-alone fertilizer trigger.", "blue", ex, so

def kcl_panel(k, cl, leaf_status):
    if cl < 0.25:
        return "Potassium × Chloride: no interaction concern", "Chloride is below the usual excess range. Interpret K normally.", "green"

    if leaf_status == "low":
        if cl < 0.40:
            return "Elevated chloride with low K", "Correct a demonstrated K shortage appropriately, but do not push K beyond the normal target range to compensate for chloride.", "yellow"
        if cl < 0.50:
            return "High chloride with low K", "The K deficiency diagnosis still matters, but chloride management is a major priority. Do not overcorrect K.", "orange"
        return "Chloride toxicity concern with low K", "Address chloride/salinity as the primary stress and correct K only toward its normal target.", "red"

    if leaf_status == "model_high_yield":
        if cl < 0.40:
            return "Elevated chloride", "K is already within the model high-yield range. Focus on chloride/salinity rather than pushing K higher.", "yellow"
        if cl < 0.50:
            return "High chloride — maintain normal K target", "Maintain appropriate K nutrition, but do not increase K above the normal target to compensate for chloride.", "orange"
        return "Chloride toxicity concern", "K is already within the model high-yield range. Address chloride/salinity first.", "red"

    if leaf_status == "sufficient":
        if cl < 0.40:
            if k <= 1.20:
                return "Possible K × Cl interaction", "Chloride is elevated and K is already adequate. Do not assume that increasing K further will improve yield.", "yellow"
            return "K × Cl interaction caution", "Chloride is elevated and K is relatively high. Additional K may not improve yield.", "orange"
        if cl < 0.50:
            return "High chloride — potassium caution", "Maintain adequate K nutrition, but do not push K higher as a presumed correction for chloride stress.", "orange"
        return "Chloride toxicity concern — do not push K higher", "Leaf chloride is in a toxicity-concern range and K is already sufficient. Address chloride/salinity first.", "red"

    return "High K plus elevated chloride", "Avoid additional K and review cation balance, fertilizer inputs and chloride/salinity management.", "red"

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("1. Leaf nutrient profile")
    nutrient_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        d = float(defaults.get(feature, 0.0))
        with cols[i % 3]:
            nutrient_values[feature] = st.number_input(
                feature, min_value=0.0, value=d, step=nutrient_step(feature, d),
                format="%.4f" if "(%)" in feature or "%" in feature else "%.2f", key=f"nutrient_{i}"
            )

    leaf_cl_manual = None
    if CL_FEATURE is None:
        leaf_cl_manual = st.number_input("Leaf chloride, Cl (%)", min_value=0.0, value=0.25, step=0.01, format="%.3f")

    st.divider()
    st.subheader("2. Soil K and root-zone information")

    lab = st.selectbox("Soil laboratory / interpretation method", ["Fruit Growers Laboratory (FGL)", "Other laboratory"])

    if lab == "Fruit Growers Laboratory (FGL)":
        st.caption("Enter Potassium-K₂O (Exch) and Potassium-K₂O (Sol) exactly as shown on the FGL report, in lb/acre-foot.")
        exch_low, exch_high = FGL_EXCH_K2O_LOW, FGL_EXCH_K2O_HIGH
        sol_low, sol_high = FGL_SOL_K2O_LOW, FGL_SOL_K2O_HIGH
    else:
        st.caption("Enter your laboratory's values and its own reference limits.")
        c1,c2,c3,c4 = st.columns(4)
        with c1: exch_low = st.number_input("Exchangeable K lower reference", min_value=0.0, value=320.0)
        with c2: exch_high = st.number_input("Exchangeable K upper reference", min_value=0.0, value=1900.0)
        with c3: sol_low = st.number_input("Solution K lower reference", min_value=0.0, value=170.0)
        with c4: sol_high = st.number_input("Solution K upper reference", min_value=0.0, value=540.0)

    s1,s2,s3 = st.columns(3)
    with s1:
        exchangeable_k = st.number_input("Exchangeable K₂O (lb/acre-foot)" if lab.startswith("Fruit") else "Exchangeable K (lab units)", min_value=0.0, value=635.0, step=5.0)
        solution_k = st.number_input("Solution K₂O (lb/acre-foot)" if lab.startswith("Fruit") else "Solution K (lab units)", min_value=0.0, value=85.3, step=1.0)
    with s2:
        cec = st.number_input("CEC (meq/100 g)", min_value=0.0, value=17.1, step=0.1)
        k_base_sat = st.number_input("K base saturation (%)", min_value=0.0, value=1.97, step=0.05)
    with s3:
        soil_salinity = st.number_input("Soil salinity / EC (dS/m)", min_value=0.0, value=0.94, step=0.05)
        sar = st.number_input("SAR", min_value=0.0, value=2.6, step=0.1)

    p1,p2 = st.columns(2)
    with p1:
        soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
    with p2:
        root_condition = st.selectbox("Root / drainage condition", ["Good drainage / healthy roots","Questionable drainage or root health","Poor drainage / known root problems"])

    st.divider()
    st.subheader("3. Potassium model scan")
    q1,q2,q3 = st.columns(3)
    with q1: k_min = st.number_input("Scan K minimum (%)", min_value=0.1, max_value=3.0, value=0.40, step=0.05)
    with q2: k_max = st.number_input("Scan K maximum (%)", min_value=0.2, max_value=4.0, value=2.00, step=0.05)
    with q3: points = st.number_input("Curve points", min_value=25, max_value=250, value=101, step=10)

    analyze = st.button("Analyze potassium response", type="primary", use_container_width=True)

with right:
    st.subheader("Potassium diagnosis")
    if not analyze:
        st.info("Enter the orchard values and select **Analyze potassium response**.")
        st.stop()

    current_k = float(nutrient_values[K_FEATURE])
    current_pred = predict_yield(nutrient_values)
    current_potential = yield_potential(current_pred)

    k_grid = np.linspace(float(k_min), float(k_max), int(points))
    scan_yield = []
    for kval in k_grid:
        row = dict(nutrient_values)
        row[K_FEATURE] = float(kval)
        scan_yield.append(predict_yield(row))
    scan_yield = np.asarray(scan_yield)

    best_idx = int(np.argmax(scan_yield))
    best_k = float(k_grid[best_idx])
    best_pred = float(scan_yield[best_idx])
    best_potential = yield_potential(best_pred)

    gain_kg = best_pred - current_pred
    gain_pct_points = best_potential - current_potential if np.isfinite(best_potential) and np.isfinite(current_potential) else np.nan

    threshold = 0.95 * best_pred
    near = k_grid[scan_yield >= threshold]
    near_low = float(near.min()) if len(near) else best_k
    near_high = float(near.max()) if len(near) else best_k

    leaf_status = k_status(current_k)

    if leaf_status == "low":
        title = "Low K relative to model high-yield profile"; cls = "orange"
        msg = "Use the soil-pool diagnosis below to determine whether this is true supply limitation or an uptake/availability problem."
    elif leaf_status == "model_high_yield":
        title = "K within model high-yield profile"; cls = "green"
        msg = "Maintain the current K program unless the individualized model scan shows a meaningful improvement elsewhere."
    elif leaf_status == "sufficient":
        title = "K sufficient — above model high-yield profile"; cls = "blue"
        msg = "K deficiency is not indicated. Additional K is unlikely to be a priority."
    else:
        title = "High K / possible imbalance"; cls = "red"
        msg = "Avoid additional K until the fertilizer program and nutrient balance are reviewed."

    st.markdown(f'<div class="box {cls}"><div class="title">{title}</div>{msg}</div>', unsafe_allow_html=True)

    soil_title, soil_msg, soil_action, soil_cls, ex_stat, sol_stat = soil_k_diagnosis(
        leaf_status, exchangeable_k, solution_k, exch_low, exch_high, sol_low, sol_high
    )
    st.markdown(f'<div class="box {soil_cls}"><div class="title">{soil_title}</div>{soil_msg}<div style="margin-top:6px">{soil_action}</div></div>', unsafe_allow_html=True)

    with st.expander("Soil K details"):
        st.write(f"**Exchangeable K:** {exchangeable_k:.1f} — {ex_stat}; lab range {exch_low:.1f}–{exch_high:.1f}.")
        st.write(f"**Solution K:** {solution_k:.1f} — {sol_stat}; lab range {sol_low:.1f}–{sol_high:.1f}.")
        if lab.startswith("Fruit"):
            base_stat = classify(k_base_sat, FGL_K_BASE_LOW, FGL_K_BASE_HIGH)
            st.write(f"**K base saturation:** {k_base_sat:.2f}% — {base_stat}; FGL range 1–6%.")
        st.caption("Low solution K alone does not establish K deficiency when leaf K and exchangeable reserve are adequate.")

    leaf_cl = float(nutrient_values[CL_FEATURE]) if CL_FEATURE is not None else float(leaf_cl_manual)
    kcl_title, kcl_msg, kcl_cls = kcl_panel(current_k, leaf_cl, leaf_status)
    st.markdown(f'<div class="box {kcl_cls}"><div class="title">{kcl_title}</div>{kcl_msg}</div>', unsafe_allow_html=True)

    st.markdown("### Yield model summary")
    y1,y2 = st.columns(2)
    with y1:
        if np.isfinite(current_potential): st.metric("Current nutrient-based yield potential", f"{current_potential:.0f}%")
        st.metric("Current model-predicted yield", f"{current_pred:,.1f} kg")
    with y2:
        if np.isfinite(best_potential): st.metric("K-only scenario potential", f"{best_potential:.0f}%")
        st.metric("Model-preferred K", f"{best_k:.2f}%")

    if np.isfinite(gain_pct_points):
        st.info(f"**K-only modeled opportunity:** {gain_pct_points:+.1f} percentage points ({gain_kg:+.1f} kg) if K alone were moved to the model-preferred region.")
    else:
        st.info(f"**K-only modeled opportunity:** {gain_kg:+.1f} kg if K alone were moved to the model-preferred region.")

    st.caption(f"Yield-potential reference: {YIELD_REF_LABEL}. The K-only scenario is not a second estimate of current yield.")

    st.success(f"**Individualized near-optimum K region:** {near_low:.2f}–{near_high:.2f}% leaf K")

    try:
        import pandas as pd
        chart_df = pd.DataFrame({"Leaf K (%)":k_grid,"Predicted yield (kg)":scan_yield}).set_index("Leaf K (%)")
        st.line_chart(chart_df, height=300, use_container_width=True)
    except Exception:
        pass

    st.divider()
    st.subheader("Root-zone context")
    if soil_salinity >= 1.3: st.error("Soil salinity is a significant avocado concern and may impair water and nutrient uptake.")
    elif soil_salinity >= 0.8: st.warning("Soil salinity is elevated for a salt-sensitive avocado root system.")
    else: st.success("Soil salinity is in a favorable low range.")

    if sar >= 6: st.error("SAR is high and may impair soil structure and root function.")
    elif sar >= 3: st.warning("SAR is elevated; sodium effects deserve attention.")
    else: st.success("SAR is low enough that sodium is unlikely to be the main constraint by itself.")

    if soil_ph > 7.2: st.warning("Soil pH is high and may reduce nutrient availability.")
    elif soil_ph < 5.5: st.warning("Soil pH is below the preferred range used by this tool.")
    else: st.success("Soil pH is within a favorable avocado range.")

    if root_condition.startswith("Poor"): st.error("Poor drainage/root health can restrict K uptake even when soil K is present.")
    elif root_condition.startswith("Questionable"): st.warning("Questionable root health/drainage could contribute to poor K uptake.")
    else: st.success("No obvious root/drainage constraint was entered.")

    st.divider()
    st.subheader("How to use this result")
    st.write("**Leaf K** tells whether the tree is expressing a K shortage. **Solution K** is the immediate soil-solution pool. **Exchangeable K** is the readily available reserve that can replenish solution K. The advisor does not treat a low solution-K value as a fertilizer recommendation by itself.")
