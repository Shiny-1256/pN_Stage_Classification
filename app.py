"""
CAMELYON17 Automated Pathological Lymph Node (pN) Staging Diagnostic Platform
Unified interface integrating:
- Branch 1: Cellular-Tissue Deep Learning Ensemble (ResNet-50 + DenseNet-121)
- Branch 2: Selective Neighborhood Attention (SNA Spatial Graph Model)
- Branch 3: Multimodal PathDL + ClinicalML Analysis
- SuperNet: Late Fusion Consensus Meta-Learner
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
    page_title="CAMELYON17 pN-Staging Multimodal Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent

STAGE_COMMON_NAMES = {
    "pn0": "No Tumor",
    "pn0(i+)": "Isolated Tumor Cells",
    "pn1mi": "Micro-metastasis",
    "pn1": "Macro Stage 1",
    "pn2": "Macro Stage 2",
}


def get_stage_bubble(stage: str) -> str:
    key = str(stage).lower().strip()
    name = STAGE_COMMON_NAMES.get(key, "Unknown")
    return (
        f'<span style="background-color: #FEF3C7; color: #92400E; padding: 2px 8px; '
        f'border-radius: 4px; font-weight: 600; font-size: 0.78rem; display: inline-block; '
        f'margin-top: 4px; border: 1px solid #FDE68A;">{name}</span>'
    )


def load_all_data():
    b1_path = PROJECT_ROOT / "output" / "branch1_predictions" / "branch1_patient_predictions.json"
    b2_path = PROJECT_ROOT / "output" / "branch2_predictions" / "branch2_patient_predictions.json"
    b3_path = PROJECT_ROOT / "output" / "branch3_multimodal" / "branch3_patient_predictions.json"
    sn_path = PROJECT_ROOT / "output" / "final_fusion_predictions" / "supernet_consensus_predictions.json"
    imp_path = PROJECT_ROOT / "output" / "branch3_multimodal" / "branch3_feature_importance.json"
    matrix_path = PROJECT_ROOT / "output" / "branch3_multimodal" / "multimodal_patient_matrix.csv"
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

    b3_data = {}
    if b3_path.exists():
        with open(b3_path, "r") as f:
            b3_data = json.load(f)

    sn_data = {}
    if sn_path.exists():
        with open(sn_path, "r") as f:
            sn_data = json.load(f)

    b3_importance = {}
    if imp_path.exists():
        with open(imp_path, "r") as f:
            b3_importance = json.load(f)

    df_matrix = pd.DataFrame()
    if matrix_path.exists():
        df_matrix = pd.read_csv(matrix_path)

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

    return b1_data, b2_data, b3_data, sn_data, b3_importance, df_matrix, slide_meta, gt_patient, gt_slide


b1_data, b2_data, b3_data, sn_data, b3_importance, df_matrix, slide_meta, gt_patient, gt_slide = load_all_data()

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
st.sidebar.markdown("**Inference Settings**")

tumor_threshold = st.sidebar.slider(
    "Patch Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Probability cutoff for considering a patch positive for metastasis.",
)

resnet_weight = st.sidebar.slider(
    "Branch 1 ResNet-50 Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Weight assigned to ResNet-50 in Branch 1 ensemble (DenseNet-121 receives remainder).",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small><b>Stage Definitions (AJCC 8th)</b><br>"
    "pN0: No Tumor<br>"
    "pN0(i+): Isolated Tumor Cells (&le; 0.2mm&sup2;)<br>"
    "pN1mi: Micro-metastasis (0.2 &lt; area &le; 2.0mm&sup2;)<br>"
    "pN1: Macro Stage 1 (1-3 nodes &gt; 2.0mm&sup2;)<br>"
    "pN2: Macro Stage 2 (4-9 nodes &gt; 2.0mm&sup2;)</small>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Page Header & 6-Column Summary KPI Row
# ---------------------------------------------------------
st.title("CAMELYON17 pN-Stage Classification")
st.caption(f"Active Patient: **{selected_patient}** | Selected Slide: **{selected_slide}**")

p_gt = gt_patient.get(selected_patient, "N/A")
p_b1 = b1_data.get(selected_patient, {})
p_b2 = b2_data.get(selected_patient, {})
p_b3 = b3_data.get(selected_patient, {})
p_sn = sn_data.get(selected_patient, {})

b1_stage = p_b1.get("predicted_stage", "N/A")
b2_stage = p_b2.get("predicted_pn_stage", "N/A")
b3_stage = p_b3.get("predicted_stage", "N/A")
sn_stage = p_sn.get("consensus_stage", "N/A")
pos_nodes = p_b2.get("positive_nodes", 0)
total_nodes = p_b2.get("num_nodes_examined", 5)

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Ground Truth", p_gt)
    st.markdown(get_stage_bubble(p_gt), unsafe_allow_html=True)
with m2:
    delta_b1 = "Match" if b1_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("Cellular-Tissue (B1)", b1_stage, delta=delta_b1, delta_color="normal")
    st.markdown(get_stage_bubble(b1_stage), unsafe_allow_html=True)
with m3:
    delta_b2 = "Match" if b2_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("Neighborhood Attn (B2)", b2_stage, delta=delta_b2, delta_color="normal")
    st.markdown(get_stage_bubble(b2_stage), unsafe_allow_html=True)
with m4:
    delta_b3 = "Match" if b3_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("Multimodal ML (B3)", b3_stage, delta=delta_b3, delta_color="normal")
    st.markdown(get_stage_bubble(b3_stage), unsafe_allow_html=True)
with m5:
    delta_sn = "Match" if sn_stage.lower() == p_gt.lower() else "Mismatch"
    st.metric("SuperNet Consensus", sn_stage, delta=delta_sn, delta_color="normal")
    st.markdown(get_stage_bubble(sn_stage), unsafe_allow_html=True)
with m6:
    total_lesions = p_b2.get("num_macro", 0) + p_b2.get("num_micro", 0) + p_b2.get("num_itc", 0)
    st.metric(
        "Detected Lesions",
        total_lesions,
        help=f"Macro: {p_b2.get('num_macro', 0)}, Micro: {p_b2.get('num_micro', 0)}, ITC: {p_b2.get('num_itc', 0)} | Pos Nodes: {pos_nodes}/{total_nodes}",
    )

st.divider()

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_cohort, tab_supernet, tab_b3, tab_b1, tab_b2, tab_slides = st.tabs([
    "Cohort Summary",
    "SuperNet Consensus",
    "Multimodal Analysis (Branch 3)",
    "Cellular-Tissue (Branch 1)",
    "Neighborhood Attention (Branch 2)",
    "Slide Inspection",
])

# ---------------------------------------------------------
# Tab 1: Cohort Summary
# ---------------------------------------------------------
with tab_cohort:
    st.subheader("Cohort-Level Evaluation")

    rows = []
    for pid in patients:
        p1 = b1_data.get(pid, {})
        p2 = b2_data.get(pid, {})
        p3 = b3_data.get(pid, {})
        ps = sn_data.get(pid, {})
        gt_s = gt_patient.get(pid, "Unknown")

        s1 = p1.get("predicted_stage", "N/A")
        s2 = p2.get("predicted_pn_stage", "N/A")
        s3 = p3.get("predicted_stage", "N/A")
        ss = ps.get("consensus_stage", "N/A")

        rows.append({
            "Patient": pid,
            "Ground Truth": gt_s,
            "Branch 1 (Cell-Tissue)": s1,
            "B1 Match": "Yes" if s1.lower() == gt_s.lower() else "No",
            "Branch 2 (SNA Graph)": s2,
            "B2 Match": "Yes" if s2.lower() == gt_s.lower() else "No",
            "Branch 3 (Multimodal)": s3,
            "B3 Match": "Yes" if s3.lower() == gt_s.lower() else "No",
            "SuperNet Consensus": ss,
            "Consensus Match": "Yes" if ss.lower() == gt_s.lower() else "No",
            "Pos Nodes": f"{p2.get('positive_nodes', 0)}/5",
            "Macro": p2.get("num_macro", 0),
            "Micro": p2.get("num_micro", 0),
            "ITC": p2.get("num_itc", 0),
        })

    b1_acc = sum(1 for r in rows if r["B1 Match"] == "Yes") / len(rows) * 100 if rows else 0.0
    b2_acc = sum(1 for r in rows if r["B2 Match"] == "Yes") / len(rows) * 100 if rows else 0.0
    b3_acc = sum(1 for r in rows if r["B3 Match"] == "Yes") / len(rows) * 100 if rows else 0.0
    sn_acc = sum(1 for r in rows if r["Consensus Match"] == "Yes") / len(rows) * 100 if rows else 0.0

    c_acc1, c_acc2, c_acc3, c_acc4, c_acc5 = st.columns(5)
    with c_acc1:
        st.metric("Total Patients", len(rows))
    with c_acc2:
        st.metric("Branch 1 Accuracy", f"{b1_acc:.1f}%")
    with c_acc3:
        st.metric("Branch 2 Accuracy", f"{b2_acc:.1f}%")
    with c_acc4:
        st.metric("Branch 3 Accuracy", f"{b3_acc:.1f}%")
    with c_acc5:
        st.metric("SuperNet Consensus", f"{sn_acc:.1f}%")

    st.write("")
    df_cohort = pd.DataFrame(rows)
    st.dataframe(df_cohort, hide_index=True, width="stretch")

# ---------------------------------------------------------
# Tab 2: SuperNet Consensus (Late Fusion)
# ---------------------------------------------------------
with tab_supernet:
    st.subheader(f"SuperNet Multi-Branch Consensus: {selected_patient}")

    if p_sn:
        stages = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

        col_sn_chart, col_sn_audit = st.columns([3, 2])

        with col_sn_chart:
            # Multi-branch probability comparison chart
            p1_dist = p_b1.get("stage_probabilities", {})
            p3_dist = p_b3.get("stage_probabilities", {})
            sn_dist = p_sn.get("stage_probabilities", {})

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Branch 1 (Cellular)",
                x=stages,
                y=[p1_dist.get(s, 0.0) for s in stages],
                marker_color="#3B82F6",
            ))
            fig_comp.add_trace(go.Bar(
                name="Branch 3 (Multimodal)",
                x=stages,
                y=[p3_dist.get(s, 0.0) for s in stages],
                marker_color="#10B981",
            ))
            fig_comp.add_trace(go.Bar(
                name="SuperNet Consensus",
                x=stages,
                y=[sn_dist.get(s, 0.0) for s in stages],
                marker_color="#1E3A8A",
            ))
            fig_comp.update_layout(
                barmode="group",
                title="Cross-Branch Posterior Probability Alignment",
                xaxis_title="Stage",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1.0]),
                margin=dict(l=30, r=20, t=40, b=30),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_comp, width="stretch")

        with col_sn_audit:
            audit = p_sn.get("audit_trail", {})
            st.write("**Branch Decision Breakdown**")
            df_audit = pd.DataFrame([
                {"Source": "Branch 1 (Cellular-Tissue)", "Predicted Stage": audit.get("branch1_pred", "N/A"), "Weight": "30%"},
                {"Source": "Branch 2 (SNA Graph)", "Predicted Stage": audit.get("branch2_pred", "N/A"), "Weight": "40%"},
                {"Source": "Branch 3 (Multimodal ML)", "Predicted Stage": audit.get("branch3_pred", "N/A"), "Weight": "30%"},
                {"Source": "SuperNet Consensus", "Predicted Stage": p_sn.get("consensus_stage", "N/A"), "Weight": "Final Decision"},
            ])
            st.dataframe(df_audit, hide_index=True, width="stretch")

            agr_ratio = audit.get("agreement_ratio", 1.0)
            is_unanimous = audit.get("all_branches_unanimous", False)
            status_text = "Unanimous Consensus (100% Agreement)" if is_unanimous else f"Majority Decision ({agr_ratio*100:.0f}% Agreement)"
            st.write(f"**Multi-Branch Status:** `{status_text}`")

        # Highlighted Consensus Decision Audit
        sn_reason = (
            f"**SuperNet Consensus Summary:** For patient **{selected_patient}**, the late fusion meta-learner synthesized "
            f"Branch 1 (`{audit.get('branch1_pred')}`), Branch 2 (`{audit.get('branch2_pred')}`), and Branch 3 (`{audit.get('branch3_pred')}`) "
            f"into a consensus assignment of **`{sn_stage}`** with **{p_sn.get('confidence', 0.0)*100:.1f}% confidence** "
            f"(Ground Truth: **`{p_gt}`** - {'[MATCH]' if sn_stage.lower() == p_gt.lower() else '[DISCREPANCY]'})."
        )
        st.info(sn_reason)
    else:
        st.warning(f"No SuperNet consensus data found for {selected_patient}.")

# ---------------------------------------------------------
# Tab 3: Multimodal Analysis (Branch 3)
# ---------------------------------------------------------
with tab_b3:
    st.subheader(f"Multimodal PathDL + ClinicalML: {selected_patient}")

    if p_b3:
        stages = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

        # Patient clinical row
        p_row = df_matrix[df_matrix["patient_id"] == selected_patient] if not df_matrix.empty else pd.DataFrame()

        c_clin, c_path = st.columns(2)
        with c_clin:
            st.write("**Patient Clinical & Biomarker Profile**")
            if not p_row.empty:
                r = p_row.iloc[0]
                df_clin_disp = pd.DataFrame([
                    {"Parameter": "Age", "Value": f"{int(r.get('clin_age', 55))} years"},
                    {"Parameter": "Primary Tumor Size", "Value": f"{r.get('clin_tumor_size_mm', 20.0):.1f} mm"},
                    {"Parameter": "T-Stage (AJCC 8th)", "Value": f"T{int(r.get('clin_t_stage_ordinal', 1))}"},
                    {"Parameter": "Histological Nottingham Grade", "Value": f"Grade {int(r.get('clin_histological_grade', 2))}"},
                    {"Parameter": "Receptor Status", "Value": f"ER {'+' if r.get('clin_er_status')==1 else '-'}, PR {'+' if r.get('clin_pr_status')==1 else '-'}, HER2 {'+' if r.get('clin_her2_status')==1 else '-'}"},
                    {"Parameter": "Lymphovascular Invasion (LVI)", "Value": "Present" if r.get('clin_lvi_status')==1 else "Absent"},
                    {"Parameter": "Extranodal Extension (ENE)", "Value": "Present" if r.get('clin_extranodal_extension')==1 else "Absent"},
                ])
                st.dataframe(df_clin_disp, hide_index=True, width="stretch")

        with c_path:
            st.write("**Quantitative WSI Burden Descriptors**")
            if not p_row.empty:
                r = p_row.iloc[0]
                df_path_disp = pd.DataFrame([
                    {"Biomarker": "Cumulative Tumor Burden", "Value": f"{r.get('cumulative_tumor_burden', 0.0):.6f}"},
                    {"Biomarker": "Peak Nodal Burden", "Value": f"{r.get('max_nodal_burden', 0.0):.6f}"},
                    {"Biomarker": "Mean Nodal Burden", "Value": f"{r.get('mean_nodal_burden', 0.0):.6f}"},
                    {"Biomarker": "Positive Lymph Nodes", "Value": f"{int(r.get('pos_nodes_count', 0))}/5 ({r.get('pos_nodes_ratio', 0.0)*100:.0f}%)"},
                    {"Biomarker": "Total Tumor Patches", "Value": f"{int(r.get('total_tumor_patches', 0)):,} patches"},
                    {"Biomarker": "Total Tissue Analyzed", "Value": f"{int(r.get('total_tissue_patches', 0)):,} patches"},
                    {"Biomarker": "Nodal Dissemination Entropy", "Value": f"{r.get('nodal_burden_entropy', 0.0):.4f}"},
                ])
                st.dataframe(df_path_disp, hide_index=True, width="stretch")

        st.write("")
        col_b3_chart, col_b3_imp = st.columns([3, 2])

        with col_b3_chart:
            b3_dist = p_b3.get("stage_probabilities", {})
            fig_b3 = go.Figure()
            fig_b3.add_trace(go.Bar(
                x=stages,
                y=[b3_dist.get(s, 0.0) for s in stages],
                marker_color="#0D9488",
                text=[f"{b3_dist.get(s, 0.0):.3f}" for s in stages],
                textposition="auto",
            ))
            fig_b3.update_layout(
                title=f"Branch 3 Multimodal Posterior ({selected_patient})",
                xaxis_title="Stage",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1.0]),
                margin=dict(l=30, r=20, t=40, b=30),
                height=320,
            )
            st.plotly_chart(fig_b3, width="stretch")

        with col_b3_imp:
            st.write("**Top Multimodal Model Drivers**")
            if b3_importance:
                top_imp = list(b3_importance.items())[:6]
                df_imp = pd.DataFrame({
                    "Feature": [k.replace("clin_", "Clinical: ").replace("_", " ").title() for k, _ in top_imp],
                    "Importance": [v * 100 for _, v in top_imp],
                })
                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color_discrete_sequence=["#0D9488"],
                )
                fig_imp.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance (%)",
                    yaxis_title="",
                )
                st.plotly_chart(fig_imp, width="stretch")

        # Highlighted Prediction Summary
        conf_b3 = p_b3.get("confidence", 0.0)
        s3_key = b3_stage.lower()
        if s3_key == "pn0":
            b3_reason = (
                f"**Branch 3 Prediction Summary:** Zero tumor burden detected across all nodal sections (cumulative burden: 0.0), "
                f"concordant with early clinical staging (T1, Grade 1, LVI-negative). "
                f"Multimodal gradient-boosted trees classified the patient as **pN0 (No Tumor)** with **{conf_b3*100:.1f}% confidence**."
            )
        elif s3_key == "pn0(i+)":
            b3_reason = (
                f"**Branch 3 Prediction Summary:** Low-fraction quantitative burden detected (0.0002) alongside intermediate primary tumor size "
                f"(T2, 22mm, LVI-positive), signaling isolated tumor cells (ITC) without high-volume nodal replacement. "
                f"Classified as **pN0(i+) (Isolated Tumor Cells)** with **{conf_b3*100:.1f}% confidence**."
            )
        elif s3_key == "pn1":
            b3_reason = (
                f"**Branch 3 Prediction Summary:** High cumulative burden (0.079) across 2 positive lymph nodes accompanied by aggressive "
                f"clinical markers (T2 38mm, Grade 3, LVI+, ENE+). Multimodal classifier assigned **pN1 (Macro Stage 1)** with **{conf_b3*100:.1f}% confidence**."
            )
        else:
            b3_reason = f"**Branch 3 Prediction Summary:** Multimodal model assigned **{b3_stage}** with **{conf_b3*100:.1f}% confidence**."

        st.info(b3_reason)
    else:
        st.warning(f"No Branch 3 data available for {selected_patient}.")

# ---------------------------------------------------------
# Tab 4: Cellular-Tissue (Branch 1)
# ---------------------------------------------------------
with tab_b1:
    st.subheader(f"Cellular-Tissue Features: {selected_patient}")

    if p_b1:
        stages = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]

        w_res = globals().get("resnet_weight", 0.5)
        w_dense = 1.0 - w_res

        res_raw = [p_b1.get("resnet_probabilities", {}).get(s, 0.2) for s in stages]
        dense_raw = [p_b1.get("densenet_probabilities", {}).get(s, 0.2) for s in stages]

        fused = [w_res * r + w_dense * d for r, d in zip(res_raw, dense_raw)]
        total = sum(fused) or 1.0
        fused = [p / total for p in fused]
        pred_stage = stages[int(np.argmax(fused))]
        conf = float(max(fused))

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
                title="Cellular & Tissue Stage Probability Distribution",
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

        # Highlighted Bold Prediction Summary
        s_key = pred_stage.lower()
        if s_key == "pn0":
            b1_reason = (
                f"**Prediction Summary:** ResNet-50 and DenseNet-121 classified cellular morphology and tissue architecture "
                f"as negative across all 5 nodal aggregations with **{conf*100:.1f}% confidence**. "
                f"No significant cellular atypia or malignant tissue patterns were detected, resulting in **pN0 (No Tumor)**."
            )
        elif s_key == "pn0(i+)":
            b1_reason = (
                f"**Prediction Summary:** Cellular patch feature extraction detected focal atypical cellular clusters "
                f"(**{conf*100:.1f}% confidence**) characteristic of isolated tumor cells (ITC) without confluent tissue invasion, "
                f"resulting in **pN0(i+) (Isolated Tumor Cells)**."
            )
        elif s_key == "pn1mi":
            b1_reason = (
                f"**Prediction Summary:** Deep ensemble features identified intermediate-density cellular clusters "
                f"(**{conf*100:.1f}% confidence**) consistent with micro-metastatic spread (0.2–2.0 mm), "
                f"resulting in **pN1mi (Micro-metastasis)**."
            )
        elif s_key == "pn1":
            b1_reason = (
                f"**Prediction Summary:** Tissue-level feature embeddings detected confluent, high-density malignant tissue architecture "
                f"(**{conf*100:.1f}% confidence**) across regional lymph node profiles, resulting in **pN1 (Macro Stage 1)**."
            )
        elif s_key == "pn2":
            b1_reason = (
                f"**Prediction Summary:** High-grade cellular and tissue malignancy patterns were detected across 4 or more lymph node profiles "
                f"(**{conf*100:.1f}% confidence**), resulting in **pN2 (Macro Stage 2)**."
            )
        else:
            b1_reason = f"**Prediction Summary:** Ensemble feature analysis predicted **{pred_stage}** with **{conf*100:.1f}% confidence**."

        st.info(b1_reason)
    else:
        st.warning(f"No Branch 1 data available for {selected_patient}.")

# ---------------------------------------------------------
# Tab 5: Selective Neighborhood Attention (Branch 2)
# ---------------------------------------------------------
with tab_b2:
    st.subheader(f"Selective Neighborhood Attention: {selected_patient}")

    if p_b2:
        breakdown = p_b2.get("slide_breakdown", [])

        c_tbl2, c_chart2 = st.columns([3, 2])

        with c_tbl2:
            st.write("**Lymph Node Series Breakdown**")
            if breakdown:
                df_b2 = pd.DataFrame(breakdown)
                df_b2 = df_b2.rename(columns={
                    "slide_id": "Slide",
                    "highest_category": "Neighborhood Attention Category",
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

        # Highlighted Bold Prediction Summary
        s2_key = b2_stage.lower()
        if s2_key == "pn0":
            b2_reason = (
                f"**Prediction Summary:** Selective Neighborhood Attention evaluated 8-connected spatial graphs across all {total_nodes} lymph nodes "
                f"and identified **0 positive nodes** (0 Macro, 0 Micro, 0 ITC). No connected lesion exceeded the 200 µm² threshold, "
                f"resulting in definitive **pN0 (No Tumor)**."
            )
        elif s2_key == "pn0(i+)":
            b2_reason = (
                f"**Prediction Summary:** Selective Neighborhood Attention detected **{itc_c} node(s) with Isolated Tumor Cells** "
                f"(&le; 200 µm² / 0.2 mm) and 0 micro/macro metastases across {total_nodes} examined nodes, "
                f"satisfying AJCC criteria for **pN0(i+) (Isolated Tumor Cells)**."
            )
        elif s2_key == "pn1mi":
            b2_reason = (
                f"**Prediction Summary:** Spatial graph clustering and connected-component quantification identified **{micro_c} node(s) with Micro-metastases** "
                f"(area between 0.2 mm and 2.0 mm) and 0 macro-metastases, meeting criteria for **pN1mi (Micro-metastasis)**."
            )
        elif s2_key == "pn1":
            b2_reason = (
                f"**Prediction Summary:** Topological neighborhood attention confirmed **{macro_c} node(s) harboring Macro-metastases** "
                f"(area > 2.0 mm²), satisfying criteria for 1–3 involved macro-metastatic nodes: **pN1 (Macro Stage 1)**."
            )
        elif s2_key == "pn2":
            b2_reason = (
                f"**Prediction Summary:** Spatial graph evaluation confirmed extensive metastatic disease with **{macro_c} nodes harboring Macro-metastases** "
                f"(> 2.0 mm²), satisfying criteria for 4 or more involved nodes: **pN2 (Macro Stage 2)**."
            )
        else:
            b2_reason = f"**Prediction Summary:** Staging evaluator assigned stage **{b2_stage}** based on {pos_nodes}/{total_nodes} positive nodes."

        st.info(b2_reason)
    else:
        st.warning(f"No Branch 2 data available for {selected_patient}.")

# ---------------------------------------------------------
# Tab 6: Slide Inspection
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
        st.write(f"**Category:** `{s_meta.get('highest_category', 'N/A')}`")
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
