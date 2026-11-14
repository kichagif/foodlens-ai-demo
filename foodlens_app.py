from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# FOODLENS AI — PROFESSIONAL DASHBOARD
# ============================================================

st.set_page_config(
    page_title="FoodLens AI | Food Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

FOOD_CLASSES = [
    "apple", "banana", "orange", "pizza", "cake", "donut",
    "mango", "grapes", "watermelon", "strawberry", "pineapple",
    "papaya", "carrot", "broccoli", "sandwich",
]

NUTRITION = {
    "apple": {"calories": 52, "protein": 0.3, "carbs": 13.8, "fat": 0.2, "fibre": 2.4},
    "banana": {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3, "fibre": 2.6},
    "orange": {"calories": 47, "protein": 0.9, "carbs": 11.8, "fat": 0.1, "fibre": 2.4},
    "mango": {"calories": 60, "protein": 0.8, "carbs": 15.0, "fat": 0.4, "fibre": 1.6},
    "grapes": {"calories": 69, "protein": 0.7, "carbs": 18.1, "fat": 0.2, "fibre": 0.9},
    "watermelon": {"calories": 30, "protein": 0.6, "carbs": 7.6, "fat": 0.2, "fibre": 0.4},
    "strawberry": {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3, "fibre": 2.0},
    "pineapple": {"calories": 50, "protein": 0.5, "carbs": 13.1, "fat": 0.1, "fibre": 1.4},
    "papaya": {"calories": 43, "protein": 0.5, "carbs": 10.8, "fat": 0.3, "fibre": 1.7},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 6.6, "fat": 0.4, "fibre": 2.6},
    "carrot": {"calories": 41, "protein": 0.9, "carbs": 9.6, "fat": 0.2, "fibre": 2.8},
    "sandwich": {"calories": 250, "protein": 10.0, "carbs": 30.0, "fat": 10.0, "fibre": 2.0},
    "pizza": {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0, "fibre": 2.3},
    "donut": {"calories": 452, "protein": 4.9, "carbs": 51.0, "fat": 25.0, "fibre": 2.0},
    "cake": {"calories": 257, "protein": 4.0, "carbs": 53.0, "fat": 3.0, "fibre": 0.5},
}

EMOJI = {
    "apple": "🍎", "banana": "🍌", "orange": "🍊", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "strawberry": "🍓",
    "pineapple": "🍍", "papaya": "🧡", "broccoli": "🥦",
    "carrot": "🥕", "sandwich": "🥪", "pizza": "🍕", "donut": "🍩",
    "cake": "🍰",
}

@st.cache_resource(show_spinner="Loading FoodLens vision model…")
def load_model():
    detector = YOLO("yolov8s-worldv2.pt")
    detector.set_classes(FOOD_CLASSES)
    return detector

def analyze_image(image: Image.Image, confidence: float, iou: float, max_det: int):
    model = load_model()
    predictions = model(image, conf=confidence, iou=iou, max_det=max_det, verbose=False)
    result = predictions[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(model.names[class_id]).lower()
            score = float(box.conf[0])
            if name in FOOD_CLASSES and score >= confidence:
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({"food": name, "confidence": score, "box": xyxy})
    return annotated, detections

def summarize_detections(detections):
    grouped = {}
    for item in detections:
        grouped.setdefault(item["food"], []).append(item["confidence"])
    rows = []
    for food, scores in sorted(grouped.items()):
        rows.append({
            "Food": f"{EMOJI.get(food, '🍽️')} {food.title()}",
            "Count": len(scores),
            "Average confidence": round(sum(scores) / len(scores) * 100, 1),
            "Highest confidence": round(max(scores) * 100, 1),
        })
    return pd.DataFrame(rows), grouped

def nutrition_totals(grouped):
    totals = {"Calories (kcal)": 0.0, "Protein (g)": 0.0, "Carbs (g)": 0.0, "Fat (g)": 0.0, "Fibre (g)": 0.0}
    for food, scores in grouped.items():
        data = NUTRITION.get(food)
        if not data:
            continue
        count = len(scores)
        totals["Calories (kcal)"] += data["calories"] * count
        totals["Protein (g)"] += data["protein"] * count
        totals["Carbs (g)"] += data["carbs"] * count
        totals["Fat (g)"] += data["fat"] * count
        totals["Fibre (g)"] += data["fibre"] * count
    return totals

def make_history_row(record):
    return {
        "Date & time": record["timestamp"],
        "Image": record["filename"],
        "Detected items": record["count"],
        "Unique foods": len(record["grouped"]),
        "Top confidence (%)": round(max((d["confidence"] for d in record["detections"]), default=0) * 100, 1),
    }

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🍎 FoodLens AI")
    st.caption("FOOD INTELLIGENCE PLATFORM")
    st.badge("YOLO-WORLD ENGINE") if hasattr(st, "badge") else st.info("🟢 YOLO-World Engine Active")
    st.divider()
    
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Analyze food", "Detection history", "Model & project"],
    )
    
    st.divider()
    st.markdown("**SYSTEM INFO**")
    st.text("Model: YOLOv8s-World")
    st.text(f"Categories: {len(FOOD_CLASSES)} classes")
    st.divider()
    st.caption("FoodLens AI · Computer Vision Suite")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("### AI-POWERED FOOD RECOGNITION")
st.title("Food intelligence, at a glance.")
st.write("Detect supported food items from images, inspect model confidence, explore detection trends, and generate educational nutrition summaries seamlessly.")
st.divider()

# ============================================================
# 1. OVERVIEW PAGE
# ============================================================
if page == "Overview":
    st.subheader("Workspace overview")
    records = st.session_state.history
    total_detections = sum(r["count"] for r in records)
    unique_foods = sorted({d["food"] for r in records for d in r["detections"]})
    avg_conf = [d["confidence"] for r in records for d in r["detections"]]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images analyzed", len(records))
    c2.metric("Food instances", total_detections)
    c3.metric("Unique categories", len(unique_foods))
    c4.metric("Average confidence", f"{sum(avg_conf)/len(avg_conf)*100:.1f}%" if avg_conf else "—")
    
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Recent analysis session")
        if records:
            recent = pd.DataFrame([make_history_row(r) for r in records[-6:][::-1]])
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            with st.container(border=True):
                st.markdown("##### Ready for your first scan")
                st.caption("Upload food images under the **Analyze food** tab to populate your activity and analytics charts.")
    with right:
        st.markdown("#### Detection distribution")
        counts = {}
        for record in records:
            for detection in record["detections"]:
                counts[detection["food"].title()] = counts.get(detection["food"].title(), 0) + 1
        if counts:
            st.bar_chart(pd.Series(counts, name="Detections").sort_values(ascending=False))
        else:
            with st.container(border=True):
                st.caption("Your food-category chart will appear here after your first analysis run.")

# ============================================================
# 2. ANALYZE FOOD PAGE
# ============================================================
elif page == "Analyze food":
    st.subheader("Image analysis workspace")
    uploaded_files = st.file_uploader(
        "Choose food images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select one or multiple images for batch computer-vision analysis.",
    )
    
    with st.expander("⚙️ Advanced Detection Settings", expanded=False):
        s1, s2, s3 = st.columns(3)
        confidence = s1.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
        iou = s2.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
        max_det = s3.slider("Maximum detections", 10, 200, 100, 10)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} image(s) selected.")
        preview_cols = st.columns(min(3, len(uploaded_files)))
        for index, file in enumerate(uploaded_files):
            with preview_cols[index % len(preview_cols)]:
                preview = Image.open(BytesIO(file.getvalue())).convert("RGB")
                st.image(preview, caption=file.name, use_container_width=True)
                
        if st.button("Run AI Vision Analysis", type="primary", use_container_width=True):
            batch_results = []
            progress = st.progress(0, text="Preparing model inference…")
            for index, file in enumerate(uploaded_files):
                try:
                    source = Image.open(BytesIO(file.getvalue())).convert("RGB")
                    annotated, detections = analyze_image(source, confidence, iou, max_det)
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": file.name,
                        "count": len(detections),
                        "detections": detections,
                        "grouped": summarize_detections(detections)[1],
                        "annotated": annotated,
                        "source": source,
                    }
                    batch_results.append(record)
                    st.session_state.history.append(record)
                except Exception as exc:
                    st.error(f"Error processing {file.name}: {exc}")
                progress.progress((index + 1) / len(uploaded_files), text=f"Processed {index + 1} of {len(uploaded_files)} image(s)")
            st.session_state.latest_results = batch_results
            progress.empty()
            if batch_results:
                st.success("Batch analysis completed successfully!")
    elif not st.session_state.latest_results:
        with st.container(border=True):
            st.markdown("#### Get started by uploading an image")
            st.caption("Supported formats: JPG, JPEG, PNG. Clear lighting and visible food items yield the best detection rates.")

    if st.session_state.latest_results:
        st.divider()
        st.markdown("### Analysis Results")
        all_detections = [d for r in st.session_state.latest_results for d in r["detections"]]
        unique_count = len(set(d["food"] for d in all_detections))
        avg = sum(d["confidence"] for d in all_detections) / len(all_detections) if all_detections else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images processed", len(st.session_state.latest_results))
        m2.metric("Food instances", len(all_detections))
        m3.metric("Categories found", unique_count)
        m4.metric("Mean confidence", f"{avg*100:.1f}%" if all_detections else "—")
        
        for idx, record in enumerate(st.session_state.latest_results, start=1):
            with st.expander(f"Result {idx}: {record['filename']} — {record['count']} item(s) detected", expanded=(idx == 1)):
                img_col, data_col = st.columns([1.15, 1])
                with img_col:
                    st.image(record["annotated"], caption="YOLO-World Annotated Output", use_container_width=True)
                with data_col:
                    if record["detections"]:
                        detection_df, grouped = summarize_detections(record["detections"])
                        st.dataframe(detection_df, use_container_width=True, hide_index=True)
                        totals = nutrition_totals(grouped)
                        st.markdown("**Estimated Nutrition Totals**")
                        n1, n2 = st.columns(2)
                        n1.metric("Calories", f"{totals['Calories (kcal)']:.0f} kcal")
                        n2.metric("Protein", f"{totals['Protein (g)']: .1f} g")
                    else:
                        st.warning("No supported items recognized in this image.")

# ============================================================
# 3. DETECTION HISTORY PAGE
# ============================================================
elif page == "Detection history":
    st.subheader("Session Detection History")
    records = st.session_state.history
    if records:
        history_df = pd.DataFrame([make_history_row(r) for r in records[::-1]])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.latest_results = []
            st.rerun()
    else:
        with st.container(border=True):
            st.caption("No history recorded in this session yet.")

# ============================================================
# 4. MODEL & PROJECT INFO PAGE
# ============================================================
else:
    st.subheader("Model & Architecture")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### YOLOv8s-World")
            st.caption("Open-vocabulary object detection tuned for flexible prompt-based food classification.")
    with col2:
        with st.container(border=True):
            st.markdown(f"#### {len(FOOD_CLASSES)} Active Classes")
            st.caption(", ".join(x.title() for x in FOOD_CLASSES))
            
    st.info("ℹ️ Note: Nutrition metrics are educational estimates calculated using standard 100g reference values per detected instance.")

st.divider()
st.markdown("<p style='text-align:center; color: gray; font-size: 0.8rem;'>FOODLENS AI · COMPUTER VISION PLATFORM</p>", unsafe_allow_html=True)
