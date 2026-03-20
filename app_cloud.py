import streamlit as st
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
import tempfile
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

st.set_page_config(
    page_title="PD MRI Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD PRECOMPUTED DEMO DATA ────────────────────────────────────
@st.cache_data
def load_demo_data():
    path = os.path.join(os.path.dirname(__file__), "demo_data.json")
    with open(path) as f:
        return json.load(f)

demo_data = load_demo_data()
demo_df   = pd.DataFrame([{
    "Subject":    d["subject"],
    "True Label": "Patient" if d["true_label"]==1 else "Control",
    "Dataset":    d.get("dataset","neurocon"),
    "QA Grade":   d["qa"]["grade"],
    "QA Score":   d["qa"]["quality_score"],
    "Prediction": d["prediction"]["prediction"]
                  if d["prediction"] else "N/A",
    "PD Prob":    f"{d['prediction']['pd_probability']:.1%}"
                  if d["prediction"] else "N/A",
    "Confidence": d["prediction"]["confidence"]
                  if d["prediction"] else "N/A",
    "Correct":    "✓" if d["prediction"] and
                  d["prediction"]["label"] == d["true_label"]
                  else "✗" if d["prediction"] else "—",
} for d in demo_data])


# ════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT
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
    if len(gm)==0 or len(wm)==0 or len(bg)==0: return 0.0
    return float(abs(np.mean(wm)-np.mean(gm)) / (np.std(bg)+1e-8))

def compute_sharpness(arr):
    gx = np.diff(arr, axis=0)
    gy = np.diff(arr, axis=1)
    gz = np.diff(arr, axis=2)
    return float(min(100.0,
        (np.mean(np.abs(gx))+np.mean(np.abs(gy))+
         np.mean(np.abs(gz)))/3.0 * 50))

def compute_ghosting(arr):
    mid  = arr[:,:,arr.shape[2]//2]
    mask = mid > np.percentile(mid, 60)
    if mask.sum()==0: return 0.0
    return float(np.std(mid[~mask])/(np.mean(mid[mask])+1e-8))

def compute_fov(arr, voxel_size):
    brain    = arr > np.percentile(arr, 20)
    edge     = (brain[0].any() or brain[-1].any() or
                brain[:,0,:].any() or brain[:,-1,:].any() or
                brain[:,:,0].any() or brain[:,:,-1].any())
    vol      = brain.sum() * np.prod(voxel_size)
    coverage = min(100.0, vol/12000.0)
    if edge: coverage *= 0.8
    return float(coverage)

def compute_uniformity(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat)==0: return 0.0
    brain = flat[flat > np.percentile(flat, 30)]
    cv    = np.std(brain)/(np.mean(brain)+1e-8)
    return float(max(0.0, 100.0 - cv*100))

def quality_grade(snr, cnr, sharp, ghost, fov, unif):
    score  = min(30, snr/3)
    score += min(20, cnr/2)
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
    return {"SNR":snr,"CNR":cnr,"Sharpness":sharp,
            "Ghosting":ghost,"FOV Coverage":fov,
            "Uniformity":unif,"Score":score,"Grade":grade}

def make_slice_fig(arr, titles):
    """Show axial/coronal/sagittal slices side by side."""
    mid = [s//2 for s in arr.shape]
    slices = [arr[:,:,mid[2]], arr[:,mid[1],:], arr[mid[0],:,:]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax, sl, title in zip(axes, slices, titles):
        sl_norm = (sl-sl.min())/(sl.max()-sl.min()+1e-8)
        ax.imshow(sl_norm.T, cmap="gray", origin="lower")
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor("black")
    plt.tight_layout(pad=0.5)
    return fig

def make_bar_fig(metrics_display, bar_colors):
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(list(metrics_display.keys()),
                   list(metrics_display.values()),
                   color=bar_colors)
    ax.set_xlim(0, 115)
    ax.set_title("Quality Metrics (normalized to 100)",
                 color="white", fontsize=11)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    for bar, val in zip(bars, metrics_display.values()):
        ax.text(bar.get_width()+1,
                bar.get_y()+bar.get_height()/2,
                f"{val:.1f}", va="center",
                color="white", fontsize=9)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 PD MRI System")
    st.markdown("**Parkinson's Disease**\nMRI Quality Assessment\n& Pattern Analysis")
    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("1. Upload T1 MRI\n2. Quality Check\n3. PD Prediction")
    st.divider()
    st.markdown("**Dataset:**")
    st.info("83 subjects\n36 HC | 47 PD\nAccuracy: 77.1%")
    st.divider()
    st.caption("⚠️ Research use only.\nNot for clinical diagnosis.")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

st.title("🧠 MRI Quality Assessment & Parkinson's Disease Analysis")
st.markdown("Upload a T1 MRI scan for quality assessment, "
            "or explore precomputed results on 83 subjects.")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "📤 Upload & Analyze",
    "📊 Dataset Results",
    "ℹ️ About"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload T1 MRI Scan for Quality Assessment")
    t1_file = st.file_uploader(
        "Upload T1 structural MRI (.nii or .nii.gz)",
        type=["gz","nii"]
    )

    if t1_file:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz",
                                         delete=False) as tmp:
            tmp.write(t1_file.read())
            t1_tmp = tmp.name

        with st.spinner("Loading scan..."):
            img        = nib.load(t1_tmp)
            arr        = img.get_fdata().astype(np.float32)
            voxel_size = img.header.get_zooms()[:3]

        st.success(f"✅ Loaded — Shape: {arr.shape} | "
                   f"Voxel: {[round(v,2) for v in voxel_size]} mm")

        # Scan preview
        st.subheader("🔍 Scan Preview")
        fig_slices = make_slice_fig(arr,
                                    ["Axial","Coronal","Sagittal"])
        st.pyplot(fig_slices, use_container_width=True)
        plt.close(fig_slices)

        # Quality assessment
        st.subheader("📋 Quality Assessment")
        with st.spinner("Assessing quality..."):
            qa = assess_quality(arr, voxel_size)

        grade       = qa["Grade"]
        score       = qa["Score"]
        grade_emoji = {"A":"✅","B":"✅","C":"⚠️","FAIL":"❌"}

        g1, g2, g3 = st.columns(3)
        g1.metric("Quality Grade",
                  f"{grade_emoji.get(grade,'?')} {grade}")
        g2.metric("Quality Score", f"{score}/100")
        g3.metric("Shape",
                  f"{arr.shape[0]}×{arr.shape[1]}×{arr.shape[2]}")

        metrics_display = {
            "SNR":          min(100, qa["SNR"]*3),
            "CNR":          min(100, qa["CNR"]*5),
            "Sharpness":    qa["Sharpness"],
            "FOV Coverage": qa["FOV Coverage"],
            "Uniformity":   qa["Uniformity"],
        }
        bar_colors = ["#00cc66" if v>=60 else
                      "#ffaa00" if v>=35 else "#ff3333"
                      for v in metrics_display.values()]

        fig_bar = make_bar_fig(metrics_display, bar_colors)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

        with st.expander("Raw values"):
            rc1,rc2,rc3 = st.columns(3)
            rc1.metric("SNR",          f"{qa['SNR']:.2f}")
            rc1.metric("CNR",          f"{qa['CNR']:.2f}")
            rc2.metric("Sharpness",    f"{qa['Sharpness']:.1f}")
            rc2.metric("Ghosting",     f"{qa['Ghosting']:.4f}")
            rc3.metric("FOV Coverage", f"{qa['FOV Coverage']:.1f}%")
            rc3.metric("Uniformity",   f"{qa['Uniformity']:.1f}%")

        if grade == "FAIL":
            st.error("❌ Scan REJECTED — quality too low. "
                     "Please re-acquire the MRI.")
        else:
            st.success(f"✅ Scan PASSED — Grade {grade}")
            st.info("💡 For PD prediction, explore the "
                    "**Dataset Results** tab to see model "
                    "performance on 83 subjects.")
    else:
        st.info("👆 Upload a T1 MRI (.nii.gz) to get started.")
        st.markdown("**Don't have a scan?** Check the "
                    "**Dataset Results** tab for precomputed results.")


# ════════════════════════════════════════════════════════════════
# TAB 2: DATASET RESULTS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Pipeline Results — All 83 Subjects")

    total     = len(demo_df)
    passed    = (demo_df["QA Grade"] != "FAIL").sum()
    predicted = demo_df[demo_df["Prediction"] != "N/A"]
    correct   = (predicted["Correct"] == "✓").sum()

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Subjects",   total)
    m2.metric("Quality Passed",   f"{passed}/{total}")
    m3.metric("With Prediction",  len(predicted))
    m4.metric("Pipeline Accuracy",
              f"{correct/len(predicted):.1%}"
              if len(predicted)>0 else "N/A")

    st.divider()

    # Grade pie chart
    gc1, gc2 = st.columns(2)

    with gc1:
        grade_counts = demo_df["QA Grade"].value_counts()
        grade_color_map = {"A":"#00cc66","B":"#88cc00",
                           "C":"#ffaa00","FAIL":"#ff3333"}
        colors_pie = [grade_color_map.get(g,"#888")
                      for g in grade_counts.index]

        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(grade_counts.values,
                   labels=grade_counts.index,
                   colors=colors_pie,
                   autopct='%1.1f%%',
                   textprops={'color':'white'})
        ax_pie.set_title("Quality Grade Distribution",
                          color="white", fontsize=11)
        fig_pie.patch.set_facecolor("#0e1117")
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    with gc2:
        counts = demo_df[demo_df["Prediction"]!="N/A"][
            "True Label"].value_counts()
        dist_colors = ["#3399ff" if x=="Control" else "#ff6666"
                       for x in counts.index]

        fig_dist, ax_dist = plt.subplots(figsize=(5, 4))
        ax_dist.bar(counts.index, counts.values,
                    color=dist_colors)
        ax_dist.set_title("True Label Distribution",
                          color="white", fontsize=11)
        ax_dist.set_facecolor("#1a1a2e")
        fig_dist.patch.set_facecolor("#0e1117")
        ax_dist.tick_params(colors="white")
        for spine in ax_dist.spines.values():
            spine.set_color("#444")
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

    # QA score histogram
    fig_qa, ax_qa = plt.subplots(figsize=(10, 3))
    controls_qa = demo_df[demo_df["True Label"]=="Control"]["QA Score"]
    patients_qa = demo_df[demo_df["True Label"]=="Patient"]["QA Score"]
    ax_qa.hist(controls_qa.values, bins=20, alpha=0.7,
               color="#3399ff", label="Control")
    ax_qa.hist(patients_qa.values, bins=20, alpha=0.7,
               color="#ff6666", label="Patient")
    ax_qa.set_title("Quality Score Distribution by Group",
                    color="white", fontsize=11)
    ax_qa.set_xlabel("QA Score", color="white")
    ax_qa.set_ylabel("Count", color="white")
    ax_qa.set_facecolor("#1a1a2e")
    fig_qa.patch.set_facecolor("#0e1117")
    ax_qa.tick_params(colors="white")
    for spine in ax_qa.spines.values():
        spine.set_color("#444")
    ax_qa.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig_qa, use_container_width=True)
    plt.close(fig_qa)

    # PD prediction results
    st.subheader("🔬 PD Prediction Results")

    pred_df = demo_df[demo_df["Prediction"] != "N/A"].copy()

    # Confusion breakdown
    tp = ((pred_df["True Label"]=="Patient") &
          (pred_df["Correct"]=="✓")).sum()
    tn = ((pred_df["True Label"]=="Control") &
          (pred_df["Correct"]=="✓")).sum()
    fp = ((pred_df["True Label"]=="Control") &
          (pred_df["Correct"]=="✗")).sum()
    fn = ((pred_df["True Label"]=="Patient") &
          (pred_df["Correct"]=="✗")).sum()

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("True Positives (PD✓)",  tp)
    r2.metric("True Negatives (HC✓)",  tn)
    r3.metric("False Positives (HC✗)", fp)
    r4.metric("False Negatives (PD✗)", fn)

    sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0
    s1, s2 = st.columns(2)
    s1.metric("Sensitivity (PD recall)", f"{sensitivity:.1%}")
    s2.metric("Specificity (HC recall)", f"{specificity:.1%}")

    # Full table
    st.subheader("Per-Subject Results")
    st.dataframe(
        demo_df,
        use_container_width=True,
        column_config={
            "QA Score": st.column_config.ProgressColumn(
                "QA Score", min_value=0, max_value=100),
        }
    )

    csv = demo_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        csv, "pd_pipeline_results.csv", "text/csv"
    )


# ════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This System")
    st.markdown("""
    ### MRI Quality Assessment & Pattern Analysis for Parkinson's Disease

    **Module 1 — Quality Assessment**
    - Computes SNR, CNR, Sharpness, FOV Coverage, Uniformity, Ghosting
    - Assigns quality grade: A (excellent) / B (good) / C (acceptable) / FAIL
    - Runs live on any uploaded T1 MRI scan

    **Module 2 — Pattern Analysis (PD Detection)**
    - Resting-state fMRI functional connectivity features
    - Data-driven brain parcellation (15 regions, k-means)
    - Features: FC matrix + mALFF + ReHo (135 features total)
    - Gradient Boosting classifier trained on NEUROCON dataset
    - 5-fold CV Accuracy: 64.7% | Sensitivity: 74.1%
    - **Pipeline Accuracy: 77.1%** (83 subjects)

    **Datasets Used**
    | Dataset | Subjects | HC | PD |
    |---------|----------|----|----|
    | NEUROCON | 43 | 16 | 27 |
    | TaoWu | 40 | 20 | 20 |
    | NTUA | 77 | 23 | 54 |

    ---
    ⚠️ *Research purposes only — not for clinical diagnosis.*

    **Author:** Arya | MIT-WPU Pune | Computer Engineering (AI & DS)
    """)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Subjects",   "83")
    c2.metric("Features",         "135")
    c3.metric("CV Accuracy",      "64.7%")
    c4.metric("Pipeline Accuracy","77.1%")