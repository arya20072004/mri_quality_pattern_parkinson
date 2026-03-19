import streamlit as st
import numpy as np
import nibabel as nib
import pandas as pd
import json
import os
import joblib
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.signal import detrend as sp_detrend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# ── CONFIG ──────────────────────────────────────────────────────
BASE      = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
MODEL_DIR = os.path.join(BASE, "models")

st.set_page_config(
    page_title="PD MRI Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODELS ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf   = joblib.load(os.path.join(MODEL_DIR, "production_classifier.pkl"))
    parc  = joblib.load(os.path.join(MODEL_DIR, "production_parcellator.pkl"))
    with open(os.path.join(MODEL_DIR, "production_config.json")) as f:
        cfg = json.load(f)
    return clf, parc, cfg

clf, parc, cfg = load_models()
TR        = cfg["tr"]
LOW_F     = cfg["low_f"]
HIGH_F    = cfg["high_f"]
N_PARCELS = cfg["n_parcels"]


# ════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT FUNCTIONS
# ════════════════════════════════════════════════════════════════

def compute_snr(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0: return 0.0
    signal_r = flat[flat > np.percentile(flat, 80)]
    noise_r  = flat[flat < np.percentile(flat, 10)]
    if len(noise_r) == 0 or np.std(noise_r) == 0: return 0.0
    return float(np.mean(signal_r) / (np.std(noise_r) + 1e-8))

def compute_cnr(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0: return 0.0
    p25, p50, p75 = np.percentile(flat, [25, 50, 75])
    gm  = flat[(flat >= p25) & (flat < p50)]
    wm  = flat[flat >= p75]
    bg  = flat[flat < np.percentile(flat, 5)]
    if len(gm) == 0 or len(wm) == 0 or len(bg) == 0: return 0.0
    return float(abs(np.mean(wm) - np.mean(gm)) / (np.std(bg) + 1e-8))

def compute_sharpness(arr):
    gx = np.diff(arr, axis=0)
    gy = np.diff(arr, axis=1)
    gz = np.diff(arr, axis=2)
    grad = (np.mean(np.abs(gx)) + np.mean(np.abs(gy)) +
            np.mean(np.abs(gz))) / 3.0
    return float(min(100.0, grad * 50))

def compute_ghosting(arr):
    mid        = arr[:, :, arr.shape[2]//2]
    brain_mask = mid > np.percentile(mid, 60)
    outside    = mid[~brain_mask]
    inside     = mid[brain_mask]
    if len(inside) == 0: return 0.0
    return float(np.std(outside) / (np.mean(inside) + 1e-8))

def compute_fov(arr, voxel_size):
    brain    = arr > np.percentile(arr, 20)
    edge     = (brain[0].any() or brain[-1].any() or
                brain[:,0,:].any() or brain[:,-1,:].any() or
                brain[:,:,0].any() or brain[:,:,-1].any())
    vol      = brain.sum() * np.prod(voxel_size)
    coverage = min(100.0, vol / 12000.0)
    if edge: coverage *= 0.8
    return float(coverage)

def compute_uniformity(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0: return 0.0
    brain = flat[flat > np.percentile(flat, 30)]
    cv    = np.std(brain) / (np.mean(brain) + 1e-8)
    return float(max(0.0, 100.0 - cv * 100))

def quality_grade(snr, cnr, sharp, ghost, fov, unif):
    score  = min(30, snr / 3)
    score += min(20, cnr / 2)
    score += sharp * 0.25
    score += fov   * 0.15
    score += unif  * 0.10
    score -= ghost * 20
    score  = max(0, min(100, score))
    if score >= 75:   grade = "A"
    elif score >= 55: grade = "B"
    elif score >= 35: grade = "C"
    else:             grade = "FAIL"
    return round(score, 1), grade

def assess_quality(arr, voxel_size):
    snr   = compute_snr(arr)
    cnr   = compute_cnr(arr)
    sharp = compute_sharpness(arr)
    ghost = compute_ghosting(arr)
    fov   = compute_fov(arr, voxel_size)
    unif  = compute_uniformity(arr)
    score, grade = quality_grade(snr, cnr, sharp, ghost, fov, unif)
    return {
        "SNR":          round(snr, 2),
        "CNR":          round(cnr, 2),
        "Sharpness":    round(sharp, 1),
        "Ghosting":     round(ghost, 4),
        "FOV Coverage": round(fov, 1),
        "Uniformity":   round(unif, 1),
        "Score":        score,
        "Grade":        grade,
    }


# ════════════════════════════════════════════════════════════════
# fMRI PREDICTION FUNCTIONS
# ════════════════════════════════════════════════════════════════

def bandpass(ts, tr, lo, hi):
    nyq = 0.5 / tr
    l, h = lo/nyq, hi/nyq
    l = max(0.001, min(l, 0.49))
    h = max(0.001, min(h, 0.49))
    if l >= h: return ts
    b, a = signal.butter(4, [l, h], btype='band')
    return signal.filtfilt(b, a, ts)

def predict_pd(fmri_arr):
    data = fmri_arr.astype(np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D fMRI volume (X, Y, Z, timepoints) "
            f"but got {data.ndim}D shape {data.shape}. "
            f"Please upload an fMRI scan, not a T1 structural scan."
        )
    X, Y, Z, T = data.shape
    if T < 50:
        raise ValueError(
            f"Too few timepoints ({T}). "
            f"Expected at least 50 for resting-state fMRI."
        )
    st.write(f"Debug — fMRI shape: {data.shape}, TR={TR}")

    mean_vol   = data.mean(axis=3)
    thresh     = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > thresh
    n_vox      = brain_mask.sum()
    st.write(f"Debug — Brain voxels: {n_vox}")

    vts = data[brain_mask]
    vts = sp_detrend(vts, axis=1)
    vts = vts / (vts.std(axis=1, keepdims=True) + 1e-8)
    filtered = np.array([bandpass(vts[i], TR, LOW_F, HIGH_F)
                         for i in range(n_vox)])

    coords  = np.array(np.where(brain_mask)).T.astype(float)
    coords /= coords.max(axis=0) + 1e-8
    labels  = parc.predict(coords)
    st.write(f"Debug — Parcellation done, unique labels: {np.unique(labels)}")

    parcel_ts = np.zeros((N_PARCELS, T))
    for p in range(N_PARCELS):
        idx = labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    fc      = np.corrcoef(parcel_ts)
    fc      = np.arctanh(np.clip(fc, -0.999, 0.999))
    fc_feats= fc[np.triu_indices(N_PARCELS, k=1)]
    st.write(f"Debug — FC features: {fc_feats.shape}")

    alff = np.zeros(N_PARCELS)
    for p in range(N_PARCELS):
        freqs = np.fft.rfftfreq(T, d=TR)
        power = np.abs(np.fft.rfft(parcel_ts[p]))**2
        rs    = (freqs >= LOW_F) & (freqs <= HIGH_F)
        if power.sum() > 0:
            alff[p] = power[rs].sum() / power.sum()

    reho = np.zeros(N_PARCELS)
    for p in range(N_PARCELS):
        idx = labels == p
        if idx.sum() > 1:
            c       = np.corrcoef(filtered[idx])
            reho[p] = c[np.triu_indices(len(c), k=1)].mean()

    feats = np.concatenate([fc_feats, alff, reho])
    st.write(f"Debug — Total features: {feats.shape}, expected: {cfg['n_features']}")

    prob  = clf.predict_proba(feats.reshape(1, -1))[0]
    pred  = int(clf.predict(feats.reshape(1, -1))[0])
    return pred, float(prob[1]), float(prob[0]), fc, parcel_ts


# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Brain_human_normal_inferior_view_with_labels_en.svg/320px-Brain_human_normal_inferior_view_with_labels_en.svg.png",
             width=200)
    st.title("🧠 PD MRI System")
    st.markdown("**Parkinson's Disease**\nMRI Quality Assessment\n& Pattern Analysis")
    st.divider()
    st.markdown("**Models loaded:**")
    st.success("✅ Quality Assessor")
    st.success("✅ fMRI Classifier")
    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("1. Upload T1 MRI\n2. Quality Check\n3. Upload fMRI\n4. PD Prediction")

# Main
st.title("🧠 MRI Quality Assessment & Parkinson's Disease Analysis")
st.markdown("Upload MRI scans to assess quality and predict Parkinson's Disease using fMRI functional connectivity.")
st.divider()

# ── TABS ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📤 Upload & Analyze",
    "📊 Dataset Results",
    "ℹ️ About"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD & ANALYZE
# ════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 1 — Upload T1 MRI Scan")
        t1_file = st.file_uploader(
            "Upload T1 structural MRI (.nii.gz)",
            type=["gz", "nii"],
            key="t1"
        )

    with col2:
        st.subheader("Step 2 — Upload fMRI Scan (optional)")
        fmri_file = st.file_uploader(
            "Upload resting-state fMRI (.nii.gz) for PD prediction",
            type=["gz", "nii"],
            key="fmri"
        )

    if t1_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz",
                                         delete=False) as tmp:
            tmp.write(t1_file.read())
            t1_tmp = tmp.name

        with st.spinner("Loading T1 scan..."):
            img        = nib.load(t1_tmp)
            arr        = img.get_fdata().astype(np.float32)
            voxel_size = img.header.get_zooms()[:3]

        st.success(f"✅ T1 loaded — Shape: {arr.shape} | "
                   f"Voxel: {[round(v,2) for v in voxel_size]} mm")

        # ── Brain Slice Viewer ────────────────────────────────────
        st.subheader("🔍 Scan Preview")
        mid = [s//2 for s in arr.shape]
        slice_col1, slice_col2, slice_col3 = st.columns(3)

        def show_slice(sl, title, col):
            sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            fig = px.imshow(sl_norm.T, color_continuous_scale="gray",
                            origin="lower", title=title,
                            aspect="equal")
            fig.update_layout(coloraxis_showscale=False,
                              margin=dict(l=0,r=0,t=30,b=0),
                              height=250)
            col.plotly_chart(fig, use_container_width=True)

        show_slice(arr[:, :, mid[2]], "Axial",    slice_col1)
        show_slice(arr[:, mid[1], :], "Coronal",  slice_col2)
        show_slice(arr[mid[0], :, :], "Sagittal", slice_col3)

        # ── Quality Assessment ────────────────────────────────────
        st.subheader("📋 Quality Assessment")
        with st.spinner("Running quality assessment..."):
            qa = assess_quality(arr, voxel_size)

        grade        = qa["Grade"]
        score        = qa["Score"]
        grade_color  = {"A":"#00cc66","B":"#88cc00",
                        "C":"#ffaa00","FAIL":"#ff3333"}
        grade_emoji  = {"A":"✅","B":"✅","C":"⚠️","FAIL":"❌"}
        color        = grade_color.get(grade, "#888")

        # Grade display
        g1, g2, g3 = st.columns(3)
        g1.metric("Quality Grade", f"{grade_emoji[grade]} {grade}")
        g2.metric("Quality Score", f"{score}/100")
        g3.metric("Scan Shape",
                  f"{arr.shape[0]}×{arr.shape[1]}×{arr.shape[2]}")

        # Metrics bar chart
        metrics_display = {
            "SNR":          min(100, qa["SNR"] * 3),
            "CNR":          min(100, qa["CNR"] * 5),
            "Sharpness":    qa["Sharpness"],
            "FOV Coverage": qa["FOV Coverage"],
            "Uniformity":   qa["Uniformity"],
        }
        bar_colors = ["#00cc66" if v >= 60 else
                      "#ffaa00" if v >= 35 else
                      "#ff3333" for v in metrics_display.values()]

        fig_bar = go.Figure(go.Bar(
            x=list(metrics_display.values()),
            y=list(metrics_display.keys()),
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in metrics_display.values()],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="Quality Metrics (normalized to 100)",
            xaxis=dict(range=[0, 110]),
            height=280,
            margin=dict(l=0, r=40, t=40, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Raw metric values
        with st.expander("Raw metric values"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("SNR",        f"{qa['SNR']:.2f}")
            mc1.metric("CNR",        f"{qa['CNR']:.2f}")
            mc2.metric("Sharpness",  f"{qa['Sharpness']:.1f}")
            mc2.metric("Ghosting",   f"{qa['Ghosting']:.4f}")
            mc3.metric("FOV Coverage",f"{qa['FOV Coverage']:.1f}%")
            mc3.metric("Uniformity", f"{qa['Uniformity']:.1f}%")

        # ── Quality Gate ─────────────────────────────────────────
        if grade == "FAIL":
            st.error("❌ **Scan REJECTED** — Quality too low for reliable analysis. "
                     "Please re-acquire the MRI scan.")
        else:
            st.success(f"✅ **Scan PASSED** quality check — Grade {grade}")

            # ── fMRI Prediction ───────────────────────────────────
            if fmri_file:
                st.divider()
                st.subheader("🔬 Parkinson's Disease Prediction")

                with tempfile.NamedTemporaryFile(suffix=".nii.gz",
                                                  delete=False) as tmp2:
                    tmp2.write(fmri_file.read())
                    fmri_tmp = tmp2.name

                with st.spinner("Analyzing fMRI connectivity... "
                                "(this takes ~2 minutes)"):
                    try:
                        fmri_img  = nib.load(fmri_tmp)
                        fmri_arr  = fmri_img.get_fdata()
                        pred, pd_prob, hc_prob, fc_mat, parcel_ts = \
                            predict_pd(fmri_arr)

                        # Result display
                        res_col1, res_col2 = st.columns([1, 1])

                        with res_col1:
                            if pred == 1:
                                st.error(
                                    f"### 🔴 Parkinson's Disease Detected\n"
                                    f"**PD Probability: {pd_prob:.1%}**")
                            else:
                                st.success(
                                    f"### 🟢 Healthy Control\n"
                                    f"**HC Probability: {hc_prob:.1%}**")

                            conf = ("High" if max(pd_prob, hc_prob) > 0.8
                                    else "Medium" if max(pd_prob, hc_prob) > 0.6
                                    else "Low")
                            st.metric("Confidence", conf)

                            # Probability gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=pd_prob * 100,
                                title={"text": "PD Risk (%)"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar":  {"color": "#ff4444"},
                                    "steps": [
                                        {"range":[0,40],
                                         "color":"#00cc66"},
                                        {"range":[40,65],
                                         "color":"#ffaa00"},
                                        {"range":[65,100],
                                         "color":"#ffdddd"},
                                    ],
                                    "threshold": {
                                        "line":{"color":"red","width":4},
                                        "thickness":0.75,
                                        "value": 50
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=280,
                                                    margin=dict(t=40,b=0))
                            st.plotly_chart(fig_gauge,
                                            use_container_width=True)

                        with res_col2:
                            # FC matrix heatmap
                            fig_fc = px.imshow(
                                fc_mat,
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                                title="Functional Connectivity Matrix",
                                labels={"color": "Fisher Z"}
                            )
                            fig_fc.update_layout(height=350,
                                                 margin=dict(t=40,b=0))
                            st.plotly_chart(fig_fc,
                                            use_container_width=True)

                        # Parcel timeseries
                        with st.expander("View parcel timeseries"):
                            fig_ts = go.Figure()
                            for p in range(min(5, N_PARCELS)):
                                fig_ts.add_trace(go.Scatter(
                                    y=parcel_ts[p],
                                    name=f"Parcel {p+1}",
                                    mode="lines",
                                    line=dict(width=1)
                                ))
                            fig_ts.update_layout(
                                title="Resting-State fMRI Timeseries "
                                      "(first 5 parcels)",
                                xaxis_title="Timepoints",
                                yaxis_title="BOLD Signal (normalized)",
                                height=300,
                                legend=dict(orientation="h")
                            )
                            st.plotly_chart(fig_ts,
                                            use_container_width=True)

                        st.caption(
                            "⚠️ This system is for research purposes only "
                            "and should not be used for clinical diagnosis.")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info("📎 Upload an fMRI scan above to get "
                        "PD prediction.")


# ════════════════════════════════════════════════════════════════
# TAB 2: DATASET RESULTS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Pipeline Results — All 83 Subjects")

    results_path = os.path.join(BASE, "outputs",
                                "pipeline_results", "pipeline_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        # Build summary dataframe
        rows = []
        for r in results:
            qa   = r.get("quality", {})
            pred = r.get("prediction")
            rows.append({
                "Subject":      r["subject"],
                "True Label":   "Patient" if r.get("true_label")==1
                                else "Control",
                "QA Grade":     qa.get("grade","?"),
                "QA Score":     qa.get("quality_score", 0),
                "Prediction":   pred["prediction"] if pred else "N/A",
                "PD Prob":      f"{pred['pd_probability']:.1%}"
                                if pred else "N/A",
                "Confidence":   pred["confidence"] if pred else "N/A",
                "Correct":      "✓" if pred and (
                    (pred["label"]==1) == (r.get("true_label")==1)
                ) else "✗" if pred else "—",
            })

        df = pd.DataFrame(rows)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        total     = len(df)
        passed    = (df["QA Grade"] != "FAIL").sum()
        predicted = df[df["Prediction"] != "N/A"]
        correct   = (predicted["Correct"] == "✓").sum()

        m1.metric("Total Subjects",   total)
        m2.metric("Quality Passed",   f"{passed}/{total}")
        m3.metric("PD Predictions",   len(predicted))
        m4.metric("Pipeline Accuracy",
                  f"{correct/len(predicted):.1%}"
                  if len(predicted) > 0 else "N/A")

        # Grade distribution
        st.divider()
        gc1, gc2 = st.columns(2)

        with gc1:
            grade_counts = df["QA Grade"].value_counts()
            fig_grade = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                title="Quality Grade Distribution",
                color_discrete_map={
                    "A":"#00cc66","B":"#88cc00",
                    "C":"#ffaa00","FAIL":"#ff3333"
                }
            )
            st.plotly_chart(fig_grade, use_container_width=True)

        with gc2:
            pred_df = predicted.copy()
            pred_df["Result"] = pred_df.apply(
                lambda r: f"{'PD' if 'Parkinson' in r['Prediction'] else 'HC'} "
                          f"({'✓' if r['Correct']=='✓' else '✗'})", axis=1)
            result_counts = pred_df["Result"].value_counts()
            fig_pred = px.bar(
                x=result_counts.index,
                y=result_counts.values,
                title="Prediction Results",
                color=result_counts.index,
                color_discrete_map={
                    "PD (✓)":"#00cc66", "PD (✗)":"#ff6666",
                    "HC (✓)":"#3399ff", "HC (✗)":"#ffaa00"
                }
            )
            fig_pred.update_layout(showlegend=False)
            st.plotly_chart(fig_pred, use_container_width=True)

        # Full table
        st.subheader("Per-Subject Results")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "QA Score": st.column_config.ProgressColumn(
                    "QA Score", min_value=0, max_value=100),
                "Correct":  st.column_config.TextColumn("✓/✗"),
            }
        )
    else:
        st.warning("No pipeline results found. "
                   "Run `python src/pipeline.py` first.")


# ════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This System")
    st.markdown("""
    ### MRI Quality Assessment & Pattern Analysis System for Parkinson's Disease

    This system implements a two-module pipeline for automated MRI analysis:

    **Module 1 — Quality Assessment**
    - Computes SNR, CNR, Sharpness, FOV Coverage, Uniformity, Ghosting
    - Assigns quality grade: A (excellent) / B (good) / C (acceptable) / FAIL
    - Acts as a quality gate before pattern analysis

    **Module 2 — Pattern Analysis (PD Detection)**
    - Extracts resting-state fMRI functional connectivity features
    - Data-driven brain parcellation (15 regions, k-means on voxel coordinates)
    - Features: FC matrix + mALFF + ReHo (135 features total)
    - Gradient Boosting classifier trained on NEUROCON dataset
    - 5-fold CV Accuracy: 64.7% | Sensitivity: 74.1%

    **Datasets Used**
    - NEUROCON: 43 subjects (16 HC + 27 PD), TR=3.48s
    - TaoWu: 40 subjects (20 HC + 20 PD), TR=2.00s
    - NTUA: 77 subjects (23 HC + 54 PD), PNG format

    **Pipeline Accuracy (83 subjects): 77.1%**

    ---
    ⚠️ *This system is for research purposes only and is not intended
    for clinical diagnosis.*

    **Developed by:** Arya | MIT-WPU | Computer Engineering (AI & Data Science)
    """)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets",        "3")
    c2.metric("Total Subjects",  "83")
    c3.metric("Pipeline Accuracy","77.1%")