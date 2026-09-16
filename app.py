"""
CAMELYON17 pN-Stage Classification Dashboard
Evaluation interface for Branch 1 (Ensemble DL) and Branch 2 (SNA Graph Model).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Streamlit Page Config
st.set_page_config(
    page_title="CAMELYON17 pN-Staging Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent


def load_all_data():
    b1_path = PROJECT_ROOT / "output" / "branch1_predictions" / "branch1_patient_predictions.json"
    b2_path = PROJECT_ROOT / "output" / "branch2_predictions" / "branch2_patient_predictions.json"
    meta_path = PROJECT_ROOT / "output" / "slide_visualizations" / "slide_meta.json"
    stage_csv = PROJECT_ROOT.parent / "dataset" / "stage_labels.csv"

    b1_data = {}
    if b1_path.exists():
        with open(b1_path, "r") as f:
            raw_b1 = json.load(f)
            b1_data = {item["patient_id"]: item for item in raw_b1}

    b2_data = {}
    if b2_path.exists():
        with open(b2_path, "r") as f:
            b2_data = json.load(f)

    slide_meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            slide_meta = json.load(f)

    gt_patient = {}
    gt_slide = {}
    if stage_csv.exists():
        df_csv = pd.read_csv(stage_csv)
        for _, row in df_csv.iterrows():
            item = str(row["patient"]).strip()
            stage = str(row["stage"]).strip()
            if item.endswith(".zip"):
                gt_patient[item.replace(".zip", "")] = stage
            elif item.endswith(".tif"):
                gt_slide[item.replace(".tif", "")] = stage

    return b1_data, b2_data, slide_meta, gt_patient, gt_slide


b1_data, b2_data, slide_meta, gt_patient, gt_slide = load_all_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.subheader("Navigation & Parameters")

patients = sorted(list(set(list(b1_data.keys()) + list(b2_data.keys()))))
if not patients:
    st.error("No prediction data found in output directory.")
    st.stop()

selected_patient = st.sidebar.selectbox("Patient", patients, index=0)

patient_slides = sorted([s for s in slide_meta.keys() if s.startswith(selected_patient)])
if not patient_slides and selected_patient in b2_data:
    patient_slides = [s["slide_id"] for s in b2_data[selected_patient].get("slide_breakdown", [])]

selected_slide = st.sidebar.selectbox(
    "Slide (Lymph Node)",
    patient_slides,
    format_func=lambda s: f"Node {s.split('_')[-1]} ({s})",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small><b>Stage Definitions</b><br>"
    "pN0: Negative<br>"
    "pN0(i+): ITC only (area &le; 0.2mm&sup2;)<br>"
    "pN1mi: Micro (0.2 &lt; area &le; 2.0mm&sup2;)<br>"
    "pN1: 1-3 macro (&gt; 2.0mm&sup2;)<br>"
    "pN2: 4-9 macro</small>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Page Header & Summary Row
# ---------------------------------------------------------
st.title("CAMELYON17 pN-Stage Classification")
st.caption(f"Active Patient: **{selected_patient}** | Selected Slide: **{selected_slide}**")

p_gt = gt_patient.get(selected_patient, "N/A")
p_b1 = b1_data.get(selected_patient, {})
p_b2 = b2_data.get(selected_patient, {})

b1_stage = p_b1.get("predicted_stage", "N/A")
b1_conf = p_b1.get("confidence", 0.0)
b2_stage = p_b2.get("predicted_pn_stage", "N/A")
pos_nodes = p_b2.get("positive_nodes", 0)
total_nodes = p_b2.get("num_nodes_examined", 5)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Ground Truth", p_gt)
with m2:
    delta_b1 = "Match" if b1_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("Branch 1 (Ensemble)", b1_stage, delta=delta_b1, delta_color="normal")
with m3:
    delta_b2 = "Match" if b2_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("Branch 2 (SNA)", b2_stage, delta=delta_b2, delta_color="normal")
with m4:
    st.metric("Positive Nodes", f"{pos_nodes}/{total_nodes}")
with m5:
    total_lesions = p_b2.get("num_macro", 0) + p_b2.get("num_micro", 0) + p_b2.get("num_itc", 0)
    st.metric("Detected Lesions", total_lesions, help=f"Macro: {p_b2.get('num_macro', 0)}, Micro: {p_b2.get('num_micro', 0)}, ITC: {p_b2.get('num_itc', 0)}")

st.divider()

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_slides, tab_b1, tab_b2, tab_cohort = st.tabs([
    "Slide Inspection",
    "Branch 1 (Ensemble)",
    "Branch 2 (SNA & Staging)",
    "Cohort Summary",
])

# ---------------------------------------------------------
# Tab 1: Slide Inspection
# ---------------------------------------------------------
with tab_slides:
    s_meta = slide_meta.get(selected_slide, {})
    s_gt = gt_slide.get(selected_slide, "Unknown")
    
    # Slide metadata bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Slide:** `{selected_slide}`")
        st.write(f"**Ground Truth:** `{s_gt}`")
    with c2:
        st.write(f"**Branch 2 Cat:** `{s_meta.get('highest_category', 'N/A')}`")
        st.write(f"**Total Patches:** `{s_meta.get('num_patches', 0):,}`")
    with c3:
        st.write(f"**Max Prob:** `{s_meta.get('max_prob', 0.0):.4f}`")
        st.write(f"**Mean Prob:** `{s_meta.get('mean_prob', 0.0):.4f}`")
    with c4:
        st.write(f"**Macro Count:** `{s_meta.get('num_macro', 0)}`")
        st.write(f"**Micro / ITC:** `{s_meta.get('num_micro', 0)} / {s_meta.get('num_itc', 0)}`")

    st.write("")

    vis_dir = PROJECT_ROOT / "output" / "slide_visualizations"
    mask_dir = PROJECT_ROOT / "output" / "tissue_masks"

    overlay_path = vis_dir / f"{selected_slide}_overlay.png"
    heatmap_path = vis_dir / f"{selected_slide}_heatmap.png"
    mask_path = mask_dir / f"{selected_slide}_mask.png"
    binary_path = vis_dir / f"{selected_slide}_binary.png"

    view_mode = st.radio(
        "Display Mode",
        ["Overlay (Heatmap + Tissue)", "Probability Heatmap", "Tissue Mask", "Binary Lesion Map", "Side-by-Side Comparison"],
        horizontal=True,
    )

    if view_mode == "Overlay (Heatmap + Tissue)":
        if overlay_path.exists():
            st.image(str(overlay_path), caption=f"{selected_slide} - Overlay", width="stretch")
        else:
            st.warning("Overlay image not found.")

    elif view_mode == "Probability Heatmap":
        if heatmap_path.exists():
            st.image(str(heatmap_path), caption=f"{selected_slide} - Continuous Heatmap (Jet)", width="stretch")
        else:
            st.warning("Heatmap image not found.")

    elif view_mode == "Tissue Mask":
        if mask_path.exists():
            st.image(str(mask_path), caption=f"{selected_slide} - Segmented Tissue Foreground", width="stretch")
        else:
            st.warning("Tissue mask not found.")

    elif view_mode == "Binary Lesion Map":
        if binary_path.exists():
            st.image(str(binary_path), caption=f"{selected_slide} - Connected Components (Binary)", width="stretch")
        else:
            st.warning("Binary map not found.")

    elif view_mode == "Side-by-Side Comparison":
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if mask_path.exists():
                st.image(str(mask_path), caption="Tissue Mask", width="stretch")
        with col_img2:
            if overlay_path.exists():
                st.image(str(overlay_path), caption="Heatmap Overlay", width="stretch")

# ---------------------------------------------------------
# Tab 2: Branch 1 (Ensemble)
# ---------------------------------------------------------
with tab_b1:
    st.subheader(f"Branch 1 Predictions: {selected_patient}")

    if p_b1:
        stages = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
        
        w_res = globals().get("resnet_weight", 0.5)
        w_dense = 1.0 - w_res

        res_raw = [p_b1.get("resnet_probabilities", {}).get(s, 0.2) for s in stages]
        dense_raw = [p_b1.get("densenet_probabilities", {}).get(s, 0.2) for s in stages]
        
        # Recalculate based on current sidebar weight
        fused = [w_res * r + w_dense * d for r, d in zip(res_raw, dense_raw)]
        total = sum(fused) or 1.0
        fused = [p / total for p in fused]
        pred_stage = stages[int(np.argmax(fused))]

        c_plot, c_tbl = st.columns([3, 2])

        with c_plot:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=stages,
                y=fused,
                marker_color="#2b5c8f",
                text=[f"{p:.3f}" for p in fused],
                textposition="auto",
            ))
            fig.update_layout(
                title="Ensemble Stage Probability Distribution",
                xaxis_title="Stage",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1.0]),
                margin=dict(l=30, r=20, t=40, b=30),
                height=350,
            )
            st.plotly_chart(fig, width="stretch")

        with c_tbl:
            st.write("**Model Probabilities by Stage**")
            df_probs = pd.DataFrame({
                "Stage": stages,
                f"ResNet-50 (w={w_res:.2f})": [f"{p:.4f}" for p in res_raw],
                f"DenseNet-121 (w={w_dense:.2f})": [f"{p:.4f}" for p in dense_raw],
                "Ensemble": [f"{p:.4f}" for p in fused],
            })
            st.dataframe(df_probs, hide_index=True, width="stretch")

            st.caption(f"Current ensemble predicted stage: **{pred_stage}** (Confidence: **{max(fused):.4f}**)")
    else:
        st.warning(f"No Branch 1 data available for {selected_patient}.")

# ---------------------------------------------------------
# Tab 3: Branch 2 (SNA & Staging)
# ---------------------------------------------------------
with tab_b2:
    st.subheader(f"Branch 2 Evaluation: {selected_patient}")

    if p_b2:
        breakdown = p_b2.get("slide_breakdown", [])
        
        c_tbl2, c_chart2 = st.columns([3, 2])

        with c_tbl2:
            st.write("**Lymph Node Series Breakdown**")
            if breakdown:
                df_b2 = pd.DataFrame(breakdown)
                df_b2 = df_b2.rename(columns={
                    "slide_id": "Slide",
                    "highest_category": "Branch 2 Category",
                    "num_macro": "Macro",
                    "num_micro": "Micro",
                    "num_itc": "ITC",
                    "ground_truth": "Ground Truth",
                })
                st.dataframe(df_b2, hide_index=True, width="stretch")

        with c_chart2:
            st.write("**Lesion Summary**")
            macro_c = p_b2.get("num_macro", 0)
            micro_c = p_b2.get("num_micro", 0)
            itc_c = p_b2.get("num_itc", 0)

            df_lesions = pd.DataFrame({
                "Category": ["Macro (>2.0mm²)", "Micro (0.2-2.0mm²)", "ITC (≤0.2mm²)"],
                "Count": [macro_c, micro_c, itc_c],
            })
            fig_bar = px.bar(
                df_lesions,
                x="Category",
                y="Count",
                text="Count",
                color_discrete_sequence=["#4a5568"],
            )
            fig_bar.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Count",
                xaxis_title="",
            )
            st.plotly_chart(fig_bar, width="stretch")

        st.write("")
        st.write(
            f"**Staging Rule Trace:** "
            f"Examined {total_nodes} nodes &rarr; Found {sum(1 for s in breakdown if s.get('highest_category') == 'macro')} Macro, "
            f"{sum(1 for s in breakdown if s.get('highest_category') == 'micro')} Micro, "
            f"{sum(1 for s in breakdown if s.get('highest_category') == 'itc')} ITC. "
            f"Final Stage Assigned: **`{b2_stage}`** (Ground Truth: **`{p_gt}`**)."
        )
    else:
        st.warning(f"No Branch 2 data available for {selected_patient}.")

# ---------------------------------------------------------
# Tab 4: Cohort Summary
# ---------------------------------------------------------
with tab_cohort:
    st.subheader("Cohort-Level Evaluation")

    rows = []
    for pid in patients:
        p1 = b1_data.get(pid, {})
        p2 = b2_data.get(pid, {})
        gt_s = gt_patient.get(pid, "Unknown")
        b1_s = p1.get("predicted_stage", "N/A")
        b2_s = p2.get("predicted_pn_stage", "N/A")

        rows.append({
            "Patient": pid,
            "Ground Truth": gt_s,
            "Branch 1": b1_s,
            "B1 Match": "Yes" if b1_s.lower() == gt_s.lower() else "No",
            "Branch 2": b2_s,
            "B2 Match": "Yes" if b2_s.lower() == gt_s.lower() else "No",
            "Pos Nodes": f"{p2.get('positive_nodes', 0)}/5",
            "Macro": p2.get("num_macro", 0),
            "Micro": p2.get("num_micro", 0),
            "ITC": p2.get("num_itc", 0),
        })

    df_cohort = pd.DataFrame(rows)
    st.dataframe(df_cohort, hide_index=True, width="stretch")

    b1_acc = sum(1 for r in rows if r["B1 Match"] == "Yes") / len(rows) * 100 if rows else 0.0
    b2_acc = sum(1 for r in rows if r["B2 Match"] == "Yes") / len(rows) * 100 if rows else 0.0
    agreement = sum(1 for r in rows if r["Branch 1"] == r["Branch 2"]) / len(rows) * 100 if rows else 0.0

    c_acc1, c_acc2, c_acc3 = st.columns(3)
    with c_acc1:
        st.metric("Branch 1 Accuracy", f"{b1_acc:.1f}%")
    with c_acc2:
        st.metric("Branch 2 Accuracy", f"{b2_acc:.1f}%")
    with c_acc3:
        st.metric("Branch Agreement", f"{agreement:.1f}%")
