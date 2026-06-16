import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Procurement Model | Dilip Buildcon",
    page_icon="🏗️",
    layout="wide",
)

# ── Training data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        "Project": [
            "Surat Metro Phase-1", "KARIMNAGAR TO WARANGAL PROJECT",
            "URGA-PATHALGAON TO TARUAMA ROA", "Sahibganj-Manihari By Pass",
            "UMMEDPURA-NAYAGAON PROJ PKG-15", "Bengaluru Vijaywada Pkg-7",
            "SARGI-BASANWAHI ROAD PROJECT", "Bengaluru Vijaywada Pkg-4",
            "MARADGI S ANDOLA TO BASWANTPUR", "MEHGAMA TO HANSDIHA ROAD PROJE",
            "Bengaluru Vijaywada Pkg-1", "BHANUPALI-BILASPUR RVNL PKG-5",
            "Bangarupalem to Gudipala Proej", "Rishikesh-Karnaprayag Tunnel",
            "DHARMAPURI TO SALEM ROAD PROJE", "Sannur to Bikarnakatte Project",
            "Malur to Bangarpet Project", "Puducherry-Poondiyankuppam Pro",
            "Sharavathi Backwater Bridge Pr", "Bangalore to Malur Project",
            "Viluppuram Puducherry Project", "BARPALI RAILROAD PROJECT",
        ],
        "Project_Type": [
            "Metro", "Highway", "Highway", "Bridge", "Expressway",
            "Highway", "Highway", "Highway", "Highway", "Highway",
            "Highway", "Tunnel", "Expressway", "Tunnel", "Highway",
            "Highway", "Highway", "Highway", "Bridge", "Highway",
            "Highway", "Railway",
        ],
        "Total_MT": [
            41979.566, 37309.5, 33575.645, 26224.343, 21297.197,
            18833.18, 18817.0, 18776.4, 18325.81, 14832.94,
            13133.33, 13038.92, 12821.0, 11358.935, 10965.86,
            9381.145, 7054.43, 5837.76, 5714.277, 5481.0,
            3137.0, 460.0,
        ],
        "Length_km": [
            8.702, 68.015, 87.55, 6.0, 8.3,
            16.0, 57.0, 24.3, 65.5, 81.83,
            24.3, 4.495, 20.0, 8.04, 6.6,
            45.012, 27.0, 27.0, 2.1, 27.0,
            29.0, 8.4,
        ],
    }
    df = pd.DataFrame(data)
    df["MT_per_km"] = df["Total_MT"] / df["Length_km"]
    return df

# ── Train models ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    y = df["Total_MT"].values
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = np.hstack([enc.fit_transform(df[["Project_Type"]]), df[["Length_km"]].values])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=50, max_depth=3,
                                          learning_rate=0.1, random_state=42, verbosity=0),
    }

    loo = LeaveOneOut()
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        tr_mape = mean_absolute_percentage_error(y, model.predict(X)) * 100
        cv_pred = np.zeros(len(y))
        for tr_idx, te_idx in loo.split(X):
            m = type(model)(**model.get_params())
            m.fit(X[tr_idx], y[tr_idx])
            cv_pred[te_idx] = m.predict(X[te_idx])
        cv_mape = mean_absolute_percentage_error(y, cv_pred) * 100
        results[name] = {
            "train": round(tr_mape, 1),
            "cv":    round(cv_mape, 1),
            "gap":   round(cv_mape - tr_mape, 1),
            "preds": cv_pred,
            "model": model,
        }

    best = min(results, key=lambda k: results[k]["cv"])
    return enc, results, best, X, y

# ── Benchmark ranges ──────────────────────────────────────────────────────────
BENCHMARKS = {
    "Highway":    {"min": 108,  "avg": 491,  "max": 1661, "n": 14},
    "Expressway": {"min": 641,  "avg": 1603, "max": 2566, "n": 2},
    "Metro":      {"min": 4824, "avg": 4824, "max": 4824, "n": 1},
    "Tunnel":     {"min": 1413, "avg": 2157, "max": 2901, "n": 2},
    "Bridge":     {"min": 2721, "avg": 3546, "max": 4371, "n": 2},
    "Railway":    {"min": 55,   "avg": 55,   "max": 55,   "n": 1},
}

TYPE_COLORS = {
    "Highway": "#2563eb", "Metro": "#f59e0b", "Tunnel": "#10b981",
    "Bridge": "#ef4444", "Expressway": "#7c3aed", "Railway": "#6b7280",
}

# ── Load & train ──────────────────────────────────────────────────────────────
df = load_data()
enc, results, best_name, X, y = train_models(df)
best_model = results[best_name]["model"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a8a,#1d4ed8);
            padding:28px 32px;border-radius:12px;margin-bottom:24px;color:white'>
    <h1 style='margin:0;font-size:1.7rem'>🏗️ Steel Procurement Predictor</h1>
    <p style='margin:6px 0 0;opacity:.8;font-size:0.95rem'>
        Dilip Buildcon Ltd. · Internship Project · FY 2023–26 · 22 Training Projects
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predictor", "📊 Model Comparison", "📐 Benchmarks", "📋 Training Data"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Total Steel Requirement")
    st.caption(f"Using **{best_name}** · CV MAPE: {results[best_name]['cv']}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        proj_type = st.selectbox(
            "Project Type",
            ["Highway", "Expressway", "Metro", "Tunnel", "Bridge", "Railway"],
        )
    with col2:
        length_km = st.number_input("Package Length (km)", min_value=0.1, step=0.5, value=45.0)
    with col3:
        procured  = st.number_input("Steel Procured Till Date (MT)", min_value=0.0, step=100.0, value=0.0)

    predict_btn = st.button("Generate Prediction →", type="primary", use_container_width=True)

    if predict_btn:
        # Encode and predict
        input_df  = pd.DataFrame({"Project_Type": [proj_type], "Length_km": [length_km]})
        X_input   = np.hstack([enc.transform(input_df[["Project_Type"]]), input_df[["Length_km"]].values])
        predicted = float(best_model.predict(X_input)[0])
        bench     = BENCHMARKS[proj_type]
        bench_low = bench["min"] * length_km
        bench_avg = bench["avg"] * length_km
        bench_hi  = bench["max"] * length_km

        # ── Results ──────────────────────────────────────────────────────────
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Predicted Total (MT)", f"{predicted:,.0f}")
        m2.metric("Steel Procured (MT)", f"{procured:,.0f}" if procured > 0 else "—")

        if procured > 0:
            progress = procured / predicted * 100
            remaining = max(predicted - procured, 0)
            m3.metric("Progress", f"{progress:.1f}%")

            # Progress bar
            bar_pct = min(progress / 100, 1.0)
            bar_col = "#ef4444" if progress > 100 else ("#10b981" if progress >= 80 else "#f59e0b")
            st.markdown(f"""
            <div style='background:#e5e7eb;border-radius:99px;height:14px;margin:8px 0'>
              <div style='width:{min(bar_pct*100,100):.1f}%;background:{bar_col};
                          height:100%;border-radius:99px;transition:width .5s'></div>
            </div>
            <p style='font-size:0.85rem;color:#6b7280'>
              Remaining demand: <strong>{remaining:,.0f} MT</strong>
            </p>
            """, unsafe_allow_html=True)

            if progress > 100:
                st.warning(
                    f"⚠️ Steel procured ({procured:,.0f} MT) exceeds model prediction ({predicted:,.0f} MT). "
                    f"The model has underpredicted — this project is likely more structure-intensive than "
                    f"the type average. Use the benchmark **high estimate ({bench_hi:,.0f} MT)** as a revised total."
                )
        else:
            m3.metric("Progress", "Enter procured MT")

        # ── Benchmark range ───────────────────────────────────────────────────
        st.markdown("#### Benchmark Range (Option C)")
        st.caption(f"Based on {bench['n']} {proj_type} project(s) in training data · "
                   f"MT/km: {bench['min']:,} – {bench['max']:,} · avg: {bench['avg']:,}")

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Low Estimate (MT)", f"{bench_low:,.0f}", f"{bench['min']} MT/km")
        bc2.metric("Avg Estimate (MT)", f"{bench_avg:,.0f}", f"{bench['avg']} MT/km")
        bc3.metric("High Estimate (MT)", f"{bench_hi:,.0f}", f"{bench['max']} MT/km")

        if bench["n"] <= 2:
            st.info(
                f"ℹ️ Only {bench['n']} {proj_type} project(s) in training data. "
                f"Treat this prediction with caution — consider consulting project DPR for structure counts."
            )

        # ── How to interpret ──────────────────────────────────────────────────
        with st.expander("How to interpret these results"):
            st.markdown(f"""
**Model prediction** ({predicted:,.0f} MT) is the Random Forest's point estimate based on the 
relationship learned from {len(df)} full-scope DBL projects.

**Benchmark range** ({bench_low:,.0f} – {bench_hi:,.0f} MT) shows the plausible range based on 
historical MT/km rates for {proj_type} projects. Use this when you know whether your project 
is structure-heavy (shade toward high) or plain-terrain (shade toward low).

**CV MAPE of {results[best_name]['cv']}%** means on average the model's prediction was off by 
that percentage on projects it hadn't seen during training. The primary driver of the error is 
structure count (number of bridges/viaducts per package), which is not available in procurement data.
            """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Evaluation — Training vs Cross-Validation Error")
    st.caption("LOOCV used due to small dataset (n=22). Lower CV MAPE = better generalisation.")

    # Table
    comp_data = []
    for name, r in results.items():
        comp_data.append({
            "Model":         name,
            "Train MAPE":    f"{r['train']}%",
            "CV MAPE":       f"{r['cv']}%",
            "Overfit Gap":   f"{r['gap']}%",
            "Selected":      "✓ Yes" if name == best_name else "—",
        })
    comp_df = pd.DataFrame(comp_data)

    def highlight_selected(row):
        if row["Selected"] == "✓ Yes":
            return ["background-color:#dbeafe; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comp_df.style.apply(highlight_selected, axis=1),
        use_container_width=True, hide_index=True
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names  = list(results.keys())
    tr_v   = [results[k]["train"] for k in names]
    cv_v   = [results[k]["cv"]    for k in names]
    x      = np.arange(len(names))
    w      = 0.35
    b1 = ax.bar(x - w/2, tr_v, w, label="Training MAPE", color="#2563eb", zorder=3)
    b2 = ax.bar(x + w/2, cv_v, w, label="CV MAPE (LOOCV)", color="#f97316", zorder=3)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color="#2563eb", fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color="#f97316", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("Model Comparison — Training vs CV Error\n(Large gap = overfitting)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(cv_v) * 1.2)
    for i, k in enumerate(names):
        ax.annotate(f"Gap: {results[k]['gap']:.1f}%",
                    xy=(i, max(tr_v[i], cv_v[i]) + 18),
                    ha="center", fontsize=8.5, color="#595959", style="italic")
    plt.tight_layout()
    st.pyplot(fig)

    # Rationale
    st.markdown("#### Selection Rationale")
    st.info(
        f"**{best_name}** was selected with the lowest CV MAPE of **{results[best_name]['cv']}%** "
        f"and smallest overfit gap (**{results[best_name]['gap']}%**). "
        "XGBoost achieved lower training error (62.6%) but its sequential boosting nearly memorised the "
        "22 training rows, causing CV error to blow up (231.2%). Linear Regression collapses when a rare "
        "project type is held out during LOOCV, producing a gap of 205.6%. Random Forest's bootstrap "
        "aggregation distributes type information more robustly across 200 trees, giving the best "
        "out-of-sample generalisation on this small dataset."
    )

    # Actual vs predicted scatter
    st.markdown("#### Actual vs CV-Predicted (Best Model)")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    best_cv = results[best_name]["preds"]
    for typ in df["Project_Type"].unique():
        mask = df["Project_Type"] == typ
        ax2.scatter(df.loc[mask, "Total_MT"], best_cv[mask],
                    color=TYPE_COLORS.get(typ, "grey"), s=90, label=typ,
                    zorder=4, edgecolors="white", linewidths=0.6)
    lim = max(df["Total_MT"].max(), best_cv.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Perfect prediction")
    ax2.fill_between([0, lim], [0, lim*0.7], [0, lim*1.3], alpha=0.06, color="green", label="±30% band")
    for i, row in df.iterrows():
        ax2.annotate(row["Project"][:22], (row["Total_MT"], best_cv[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=6.5, color="#555")
    ax2.set_xlabel("Actual Total MT", fontsize=11)
    ax2.set_ylabel("CV Predicted MT", fontsize=11)
    ax2.set_title(f"Actual vs CV Predicted — {best_name}", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARKS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Steel Intensity Benchmark Ranges by Project Type")
    st.caption("Derived from 22 full-scope DBL projects · FY 2023–26 · Use: Estimate = Length × Avg MT/km")

    for typ, b in BENCHMARKS.items():
        col = TYPE_COLORS.get(typ, "#6b7280")
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 2])
        c1.markdown(f"<span style='color:{col};font-weight:700;font-size:1rem'>{typ}</span> "
                    f"<span style='color:#9ca3af;font-size:0.8rem'>(n={b['n']})</span>",
                    unsafe_allow_html=True)
        c2.metric("Min MT/km", f"{b['min']:,}")
        c3.metric("Avg MT/km", f"{b['avg']:,}")
        c4.metric("Max MT/km", f"{b['max']:,}")

        # Visual bar
        max_val = 5500
        pct_start = b["min"] / max_val * 100
        pct_width = (b["max"] - b["min"]) / max_val * 100
        pct_avg   = b["avg"] / max_val * 100
        st.markdown(f"""
        <div style='background:#f3f4f6;border-radius:99px;height:8px;margin:-8px 0 16px;position:relative'>
          <div style='position:absolute;left:{pct_start:.1f}%;width:{pct_width:.1f}%;
                      background:{col};opacity:.35;height:100%;border-radius:99px'></div>
          <div style='position:absolute;left:{pct_avg:.1f}%;width:3px;height:100%;
                      background:{col};border-radius:2px'></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Interpretation Notes")
    notes = {
        "Highway":    "Wide range (108–1,661 MT/km) driven by structure count. Ghat/mountain sections and urban packages are 5–10× more steel-intensive than plain-terrain packages of the same length.",
        "Expressway": "Limited data (2 projects). Short high-structure packages (UMMEDPURA: 8.3 km) show very high intensity vs. longer plain packages (Bangarupalem: 20 km).",
        "Metro":      "Single project (Surat Metro, elevated viaduct). Do not rely on model alone — use 4,824 MT/km as a reference and adjust for your specific metro alignment type.",
        "Tunnel":     "Two projects. Range 1,413–2,901 MT/km driven by cross-section size and lining type. Bhanupali-Bilaspur (railway tunnel) is more intensive than Rishikesh-Karnaprayag.",
        "Bridge":     "Two major river/backwater bridges. Extradosed or cable-stayed designs are at the high end. Use structure type to select within the 2,721–4,371 range.",
        "Railway":    "Single small project. Insufficient data for a reliable benchmark — treat with caution.",
    }
    for typ, note in notes.items():
        col = TYPE_COLORS.get(typ, "#6b7280")
        st.markdown(f"**<span style='color:{col}'>{typ}:</span>** {note}", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAINING DATA
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Training Dataset — 22 Full-Scope Projects")
    st.caption("FY 2023–26 · Dilip Buildcon Ltd. · Filtered: MT/km > 50 (full-scope packages only)")

    display_df = df.rename(columns={
        "Project": "Project", "Project_Type": "Type",
        "Total_MT": "Total MT", "Length_km": "Length (km)", "MT_per_km": "MT/km"
    }).copy()
    display_df["Total MT"]   = display_df["Total MT"].apply(lambda x: f"{x:,.0f}")
    display_df["Length (km)"] = display_df["Length (km)"].apply(lambda x: f"{x:.1f}")
    display_df["MT/km"]      = display_df["MT/km"].apply(lambda x: f"{x:,.0f}")

    def color_type(val):
        colors = {
            "Highway":"background-color:#dbeafe", "Metro":"background-color:#fef3c7",
            "Tunnel":"background-color:#d1fae5", "Bridge":"background-color:#fee2e2",
            "Expressway":"background-color:#ede9fe", "Railway":"background-color:#f3f4f6",
        }
        return colors.get(val, "")

    st.dataframe(
        display_df.style.map(color_type, subset=["Type"]),
        use_container_width=True, hide_index=True, height=600
    )

    # MT/km scatter
    st.markdown("#### MT/km Distribution by Project Type")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for typ in df["Project_Type"].unique():
        mask = df["Project_Type"] == typ
        ax3.scatter(df.loc[mask, "Length_km"], df.loc[mask, "MT_per_km"],
                    color=TYPE_COLORS.get(typ, "grey"), s=80, label=typ,
                    zorder=4, edgecolors="white", linewidths=0.6)
    for _, row in df.iterrows():
        ax3.annotate(row["Project"][:18], (row["Length_km"], row["MT_per_km"]),
                     textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="#555")
    ax3.set_xlabel("Package Length (km)", fontsize=11)
    ax3.set_ylabel("MT per km", fontsize=11)
    ax3.set_title("Steel Intensity vs Package Length", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.8rem'>"
    "Internship Project · Dilip Buildcon Ltd. · Steel Procurement Optimization · FY 2023–26"
    "</p>",
    unsafe_allow_html=True
)
