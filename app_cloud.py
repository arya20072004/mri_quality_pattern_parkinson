import streamlit as st
import numpy as np
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile
import warnings
from scipy import signal as sp_signal
from scipy.signal import detrend as sp_detrend
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

st.set_page_config(
    page_title="PD MRI Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# LOAD DEMO DATA
# ════════════════════════════════════════════════════════════════
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
    gm = flat[(flat >= p25) & (flat < p50)]
    wm = flat[flat >= p75]
    bg = flat[flat < np.percentile(flat, 5)]
    if len(gm)==0 or len(wm)==0 or len(bg)==0: return 0.0
    return float(abs(np.mean(wm)-np.mean(gm))/(np.std(bg)+1e-8))

def compute_sharpness(arr):
    gx = np.diff(arr, axis=0)
    gy = np.diff(arr, axis=1)
    gz = np.diff(arr, axis=2)
    return float(min(100.0,
        (np.mean(np.abs(gx))+np.mean(np.abs(gy))+
         np.mean(np.abs(gz)))/3.0*50))

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
    return float(max(0.0, 100.0-cv*100))

def quality_grade(snr, cnr, sharp, ghost, fov, unif):
    score  = min(30, snr/3)
    score += min(20, cnr/2)
    score += sharp*0.25
    score += fov*0.15
    score += unif*0.10
    score -= ghost*20
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


# ════════════════════════════════════════════════════════════════
# fMRI PREDICTION FUNCTIONS
# ════════════════════════════════════════════════════════════════
def bandpass_cloud(ts, tr, lo=0.01, hi=0.10):
    nyq = 0.5/tr
    l, h = lo/nyq, hi/nyq
    l = max(0.001, min(l, 0.49))
    h = max(0.001, min(h, 0.49))
    if l >= h: return ts
    b, a = sp_signal.butter(4, [l, h], btype='band')
    return sp_signal.filtfilt(b, a, ts)

@st.cache_resource
def load_cloud_model():
    path = os.path.join(os.path.dirname(__file__),
                        "models", "model_cloud.json")
    with open(path) as f:
        return json.load(f)

def predict_with_json_model(feats, model_data):
    """Run inference using JSON model — no pickle needed."""
    # Scale
    mean = np.array(model_data["scaler_mean"])
    std  = np.array(model_data["scaler_std"])
    x    = (feats - mean) / (std + 1e-8)

    # Select features
    mask = np.array(model_data["selected_mask"])
    x    = x[mask]

    # GBM prediction
    lr        = model_data["learning_rate"]
    log_odds  = np.log(model_data["init_pred"] /
                       (1 - model_data["init_pred"] + 1e-8))
    pred_sum  = log_odds

    for tree in model_data["trees"]:
        cl  = tree["children_left"]
        cr  = tree["children_right"]
        ft  = tree["feature"]
        thr = tree["threshold"]
        val = tree["value"]

        node = 0
        while cl[node] != -1:
            if x[ft[node]] <= thr[node]:
                node = cl[node]
            else:
                node = cr[node]
        pred_sum += lr * val[node][0][0]

    pd_prob = 1 / (1 + np.exp(-pred_sum))
    pd_prob = float(np.clip(pd_prob, 0.01, 0.99))
    pred    = 1 if pd_prob > 0.5 else 0
    return pred, pd_prob, 1 - pd_prob

def predict_pd_cloud(fmri_arr, tr=3.48, n_parcels=15):
    """Memory-efficient fMRI prediction using JSON model."""
    model_data = load_cloud_model()
    km_centers = np.array(model_data["km_centers"])

    data = fmri_arr[::2, ::2, ::2, :].astype(np.float32)
    T    = data.shape[3]

    mean_vol   = data.mean(axis=3)
    thresh     = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > thresh
    n_vox      = brain_mask.sum()

    vts = data[brain_mask]
    vts = sp_detrend(vts, axis=1)
    vts = vts / (vts.std(axis=1, keepdims=True) + 1e-8)
    filtered = np.array([bandpass_cloud(vts[i], tr)
                         for i in range(n_vox)])

    coords  = np.array(np.where(brain_mask)).T.astype(float)
    coords /= coords.max(axis=0) + 1e-8

    # Use saved cluster centers — no random state needed
    dists  = np.linalg.norm(
        coords[:,None,:] - km_centers[None,:,:], axis=2)
    labels = dists.argmin(axis=1)

    parcel_ts = np.zeros((n_parcels, T))
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    fc      = np.corrcoef(parcel_ts)
    fc      = np.arctanh(np.clip(fc, -0.999, 0.999))
    fc_feats= fc[np.triu_indices(n_parcels, k=1)]

    alff = np.zeros(n_parcels)
    for p in range(n_parcels):
        freqs = np.fft.rfftfreq(T, d=tr)
        power = np.abs(np.fft.rfft(parcel_ts[p]))**2
        rs    = (freqs >= 0.01) & (freqs <= 0.10)
        if power.sum() > 0:
            alff[p] = power[rs].sum()/power.sum()

    reho = np.zeros(n_parcels)
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 1:
            c       = np.corrcoef(filtered[idx])
            reho[p] = c[np.triu_indices(len(c), k=1)].mean()

    feats = np.concatenate([fc_feats, alff, reho])
    pred, pd_prob, hc_prob = predict_with_json_model(
        feats, model_data)
    return pred, pd_prob, hc_prob, fc


# ════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════
def make_slice_fig(arr, titles):
    mid = [s//2 for s in arr.shape]
    slices = [arr[:,:,mid[2]], arr[:,mid[1],:], arr[mid[0],:,:]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax, sl, title in zip(axes, slices, titles):
        sl_n = (sl-sl.min())/(sl.max()-sl.min()+1e-8)
        ax.imshow(sl_n.T, cmap="gray", origin="lower")
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
    st.markdown("1. Upload T1 MRI\n2. Quality Check\n3. Upload fMRI\n4. PD Prediction")
    st.divider()
    st.markdown("**Dataset:**")
    st.info("83 subjects\n36 HC | 47 PD\nAccuracy: 77.1%")
    st.divider()
    st.caption("⚠️ Research use only.\nNot for clinical diagnosis.")


# ════════════════════════════════════════════════════════════════
# MAIN TITLE
# ════════════════════════════════════════════════════════════════
st.title("🧠 MRI Quality Assessment & Parkinson's Disease Analysis")
st.markdown("Upload MRI scans for quality assessment and "
            "Parkinson's Disease prediction.")
st.divider()

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
            "Upload T1 structural MRI (.nii or .nii.gz)",
            type=["gz","nii"], key="t1"
        )
    with col2:
        st.subheader("Step 2 — Upload fMRI Scan")
        st.caption("Upload resting-state fMRI (.nii.gz) "
                   "for PD prediction")
        fmri_file = st.file_uploader(
            "Upload resting-state fMRI (.nii.gz)",
            type=["gz","nii"], key="fmri"
        )

    if t1_file:
        with tempfile.NamedTemporaryFile(
                suffix=".nii.gz", delete=False) as tmp:
            tmp.write(t1_file.read())
            t1_tmp = tmp.name

        with st.spinner("Loading T1 scan..."):
            img        = nib.load(t1_tmp)
            arr        = img.get_fdata().astype(np.float32)
            voxel_size = img.header.get_zooms()[:3]

        st.success(f"✅ T1 loaded — Shape: {arr.shape} | "
                   f"Voxel: {[round(v,2) for v in voxel_size]} mm")

        # Scan preview
        st.subheader("🔍 Scan Preview")
        fig_slices = make_slice_fig(
            arr, ["Axial","Coronal","Sagittal"])
        st.pyplot(fig_slices, use_container_width=True)
        plt.close(fig_slices)

        # Quality assessment
        st.subheader("📋 Quality Assessment")
        with st.spinner("Assessing scan quality..."):
            qa = assess_quality(arr, voxel_size)

        grade       = qa["Grade"]
        score       = qa["Score"]
        grade_emoji = {"A":"✅","B":"✅","C":"⚠️","FAIL":"❌"}

        g1, g2, g3 = st.columns(3)
        g1.metric("Quality Grade",
                  f"{grade_emoji.get(grade,'?')} {grade}")
        g2.metric("Quality Score", f"{score}/100")
        g3.metric("Shape",
                  f"{arr.shape[0]}×"
                  f"{arr.shape[1]}×{arr.shape[2]}")

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

        with st.expander("Raw metric values"):
            rc1,rc2,rc3 = st.columns(3)
            rc1.metric("SNR",          f"{qa['SNR']:.2f}")
            rc1.metric("CNR",          f"{qa['CNR']:.2f}")
            rc2.metric("Sharpness",    f"{qa['Sharpness']:.1f}")
            rc2.metric("Ghosting",     f"{qa['Ghosting']:.4f}")
            rc3.metric("FOV Coverage",
                       f"{qa['FOV Coverage']:.1f}%")
            rc3.metric("Uniformity",   f"{qa['Uniformity']:.1f}%")

        if grade == "FAIL":
            st.error("❌ Scan REJECTED — quality too low. "
                     "Please re-acquire the MRI.")
        else:
            st.success(f"✅ Scan PASSED quality check — Grade {grade}")

            # fMRI Prediction
            st.divider()
            st.subheader("🔬 Parkinson's Disease Prediction")

            if fmri_file:
                with tempfile.NamedTemporaryFile(
                        suffix=".nii.gz", delete=False) as tmp2:
                    tmp2.write(fmri_file.read())
                    fmri_tmp = tmp2.name

                with st.spinner(
                        "Analyzing fMRI connectivity... "
                        "(2–3 minutes)"):
                    try:
                        fmri_img = nib.load(fmri_tmp)
                        fmri_arr = fmri_img.get_fdata().astype(
                            np.float32)

                        if fmri_arr.ndim != 4:
                            st.error(
                                f"Expected 4D fMRI but got "
                                f"{fmri_arr.ndim}D shape "
                                f"{fmri_arr.shape}. Please upload "
                                f"the resting-state fMRI bold file.")
                        elif fmri_arr.shape[3] < 50:
                            st.error(
                                f"Too few timepoints "
                                f"({fmri_arr.shape[3]}). "
                                f"Need resting-state fMRI "
                                f"(≥50 timepoints).")
                        else:
                            pred, pd_prob, hc_prob, fc_mat = \
                                predict_pd_cloud(fmri_arr)

                            res1, res2 = st.columns([1,1])

                            with res1:
                                if pred == 1:
                                    st.error(
                                        "### 🔴 Parkinson's Disease\n"
                                        f"**PD Probability: "
                                        f"{pd_prob:.1%}**")
                                else:
                                    st.success(
                                        "### 🟢 Healthy Control\n"
                                        f"**HC Probability: "
                                        f"{hc_prob:.1%}**")

                                conf = (
                                    "High"
                                    if max(pd_prob,hc_prob) > 0.8
                                    else "Medium"
                                    if max(pd_prob,hc_prob) > 0.6
                                    else "Low")
                                st.metric("Confidence", conf)
                                st.metric("PD Risk",
                                          f"{pd_prob:.1%}")
                                st.metric("HC Probability",
                                          f"{hc_prob:.1%}")

                            with res2:
                                fig_fc, ax_fc = plt.subplots(
                                    figsize=(5,4))
                                im = ax_fc.imshow(
                                    fc_mat, cmap="RdBu_r",
                                    vmin=-1, vmax=1)
                                ax_fc.set_title(
                                    "Functional Connectivity Matrix",
                                    color="white", fontsize=10)
                                ax_fc.set_xlabel("Parcel",
                                                  color="white")
                                ax_fc.set_ylabel("Parcel",
                                                  color="white")
                                ax_fc.tick_params(colors="white")
                                fig_fc.patch.set_facecolor(
                                    "#0e1117")
                                ax_fc.set_facecolor("#0e1117")
                                plt.colorbar(im, ax=ax_fc)
                                plt.tight_layout()
                                st.pyplot(fig_fc,
                                          use_container_width=True)
                                plt.close(fig_fc)

                            st.caption(
                                "⚠️ For research only. "
                                "Not for clinical diagnosis.")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info(
                    "📎 Upload a resting-state fMRI scan "
                    "(.nii.gz) above to get PD prediction.\n\n"
                    "**Example file:**\n"
                    "`sub-patient032030_task-resting_"
                    "run-1_bold.nii.gz`")
    else:
        st.info("👆 Upload a T1 MRI scan (.nii.gz) to begin.")


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
    gc1, gc2 = st.columns(2)

    with gc1:
        grade_counts    = demo_df["QA Grade"].value_counts()
        grade_color_map = {"A":"#00cc66","B":"#88cc00",
                           "C":"#ffaa00","FAIL":"#ff3333"}
        colors_pie = [grade_color_map.get(g,"#888")
                      for g in grade_counts.index]
        fig_pie, ax_pie = plt.subplots(figsize=(5,4))
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
        counts = demo_df[
            demo_df["Prediction"]!="N/A"]["True Label"].value_counts()
        dist_colors = ["#3399ff" if x=="Control"
                       else "#ff6666" for x in counts.index]
        fig_dist, ax_dist = plt.subplots(figsize=(5,4))
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

    # QA histogram
    fig_qa, ax_qa = plt.subplots(figsize=(10,3))
    ctrl_qa = demo_df[
        demo_df["True Label"]=="Control"]["QA Score"]
    pat_qa  = demo_df[
        demo_df["True Label"]=="Patient"]["QA Score"]
    ax_qa.hist(ctrl_qa.values, bins=20, alpha=0.7,
               color="#3399ff", label="Control")
    ax_qa.hist(pat_qa.values,  bins=20, alpha=0.7,
               color="#ff6666", label="Patient")
    ax_qa.set_title("Quality Score Distribution by Group",
                    color="white", fontsize=11)
    ax_qa.set_xlabel("QA Score", color="white")
    ax_qa.set_ylabel("Count",    color="white")
    ax_qa.set_facecolor("#1a1a2e")
    fig_qa.patch.set_facecolor("#0e1117")
    ax_qa.tick_params(colors="white")
    for spine in ax_qa.spines.values():
        spine.set_color("#444")
    ax_qa.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig_qa, use_container_width=True)
    plt.close(fig_qa)

    # Confusion metrics
    st.subheader("🔬 PD Prediction Results")
    pred_df = demo_df[demo_df["Prediction"] != "N/A"].copy()
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

    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    s1,s2 = st.columns(2)
    s1.metric("Sensitivity (PD recall)", f"{sens:.1%}")
    s2.metric("Specificity (HC recall)", f"{spec:.1%}")

    # Full table
    st.subheader("Per-Subject Results")
    st.dataframe(
        demo_df, use_container_width=True,
        column_config={
            "QA Score": st.column_config.ProgressColumn(
                "QA Score", min_value=0, max_value=100),
        }
    )
    csv = demo_df.to_csv(index=False)
    st.download_button("⬇️ Download Results CSV",
                       csv, "pd_results.csv", "text/csv")


# ════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This System")
    st.markdown("""
    ### MRI Quality Assessment & Pattern Analysis for Parkinson's Disease

    **Module 1 — Quality Assessment**
    - SNR, CNR, Sharpness, FOV Coverage, Uniformity, Ghosting
    - Grade: A / B / C / FAIL
    - Runs live on any uploaded T1 MRI

    **Module 2 — Pattern Analysis (PD Detection)**
    - Resting-state fMRI functional connectivity
    - 15-parcel data-driven brain parcellation (k-means)
    - Features: FC matrix + mALFF + ReHo (135 features)
    - Gradient Boosting classifier
    - CV Accuracy: 64.7% | Sensitivity: 74.1%
    - **Pipeline Accuracy: 77.1%** (83 subjects)

    **Datasets**
    | Dataset | Subjects | HC | PD |
    |---------|----------|----|----|
    | NEUROCON | 43 | 16 | 27 |
    | TaoWu | 40 | 20 | 20 |
    | NTUA | 77 | 23 | 54 |

    ---
    ⚠️ *Research purposes only — not for clinical diagnosis.*

    **Author:** Arya | MIT-WPU Pune | CE (AI & Data Science)
    """)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Subjects",   "83")
    c2.metric("Features",         "135")
    c3.metric("CV Accuracy",      "64.7%")
    c4.metric("Pipeline Accuracy","77.1%")