from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# FRUITLENS AI — PROFESSIONAL DASHBOARD
# Existing YOLO-World fruit detection model + expanded analysis workflow
# ============================================================

st.set_page_config(
    page_title="FruitLens AI | Fruit Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRUIT_CLASSES = [
    'apple',
    'banana',
    'orange',
    'mango',
    'grapes',
    'watermelon',
    'strawberry',
    'pineapple',
    'papaya',
    'pear',
    'peach',
    'plum',
    'cherry',
    'kiwi',
    'pomegranate',
    'guava',
    'coconut',
    'lemon',
    'lime',
    'avocado',
    'blueberry',
    'raspberry',
    'blackberry',
    'cranberry',
    'dragon fruit',
    'passion fruit',
    'lychee',
    'rambutan',
    'jackfruit',
    'durian',
    'fig',
    'date',
    'apricot',
    'nectarine',
    'grapefruit',
    'tangerine',
    'mandarin',
    'custard apple',
    'sapodilla',
    'star fruit',
    'mulberry',
    'indian gooseberry',
    'jamun',
    'longan',
    'persimmon',
]

# Approximate reference values per 100 g. These are educational estimates,
# not portion measurements or medical/dietary advice.
FRUIT_NUTRITION = {
    'apple': {"calories": 52, "protein": 0.3, "carbs": 13.8, "fat": 0.2, "fibre": 2.4},
    'banana': {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3, "fibre": 2.6},
    'orange': {"calories": 47, "protein": 0.9, "carbs": 11.8, "fat": 0.1, "fibre": 2.4},
    'mango': {"calories": 60, "protein": 0.8, "carbs": 15, "fat": 0.4, "fibre": 1.6},
    'grapes': {"calories": 69, "protein": 0.7, "carbs": 18.1, "fat": 0.2, "fibre": 0.9},
    'watermelon': {"calories": 30, "protein": 0.6, "carbs": 7.6, "fat": 0.2, "fibre": 0.4},
    'strawberry': {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3, "fibre": 2},
    'pineapple': {"calories": 50, "protein": 0.5, "carbs": 13.1, "fat": 0.1, "fibre": 1.4},
    'papaya': {"calories": 43, "protein": 0.5, "carbs": 10.8, "fat": 0.3, "fibre": 1.7},
    'pear': {"calories": 57, "protein": 0.4, "carbs": 15.2, "fat": 0.1, "fibre": 3.1},
    'peach': {"calories": 39, "protein": 0.9, "carbs": 9.5, "fat": 0.3, "fibre": 1.5},
    'plum': {"calories": 46, "protein": 0.7, "carbs": 11.4, "fat": 0.3, "fibre": 1.4},
    'cherry': {"calories": 63, "protein": 1.1, "carbs": 16, "fat": 0.2, "fibre": 2.1},
    'kiwi': {"calories": 61, "protein": 1.1, "carbs": 14.7, "fat": 0.5, "fibre": 3},
    'pomegranate': {"calories": 83, "protein": 1.7, "carbs": 18.7, "fat": 1.2, "fibre": 4},
    'guava': {"calories": 68, "protein": 2.6, "carbs": 14.3, "fat": 1, "fibre": 5.4},
    'coconut': {"calories": 354, "protein": 3.3, "carbs": 15.2, "fat": 33.5, "fibre": 9},
    'lemon': {"calories": 29, "protein": 1.1, "carbs": 9.3, "fat": 0.3, "fibre": 2.8},
    'lime': {"calories": 30, "protein": 0.7, "carbs": 10.5, "fat": 0.2, "fibre": 2.8},
    'avocado': {"calories": 160, "protein": 2, "carbs": 8.5, "fat": 14.7, "fibre": 6.7},
    'blueberry': {"calories": 57, "protein": 0.7, "carbs": 14.5, "fat": 0.3, "fibre": 2.4},
    'raspberry': {"calories": 52, "protein": 1.2, "carbs": 11.9, "fat": 0.7, "fibre": 6.5},
    'blackberry': {"calories": 43, "protein": 1.4, "carbs": 9.6, "fat": 0.5, "fibre": 5.3},
    'cranberry': {"calories": 46, "protein": 0.4, "carbs": 12, "fat": 0.1, "fibre": 4.6},
    'dragon fruit': {"calories": 57, "protein": 1.2, "carbs": 13, "fat": 0.1, "fibre": 3},
    'passion fruit': {"calories": 97, "protein": 2.2, "carbs": 23.4, "fat": 0.7, "fibre": 10.4},
    'lychee': {"calories": 66, "protein": 0.8, "carbs": 16.5, "fat": 0.4, "fibre": 1.3},
    'rambutan': {"calories": 68, "protein": 0.9, "carbs": 16.5, "fat": 0.2, "fibre": 0.9},
    'jackfruit': {"calories": 95, "protein": 1.7, "carbs": 23, "fat": 0.6, "fibre": 1.5},
    'durian': {"calories": 147, "protein": 1.5, "carbs": 27.1, "fat": 5.3, "fibre": 3.8},
    'fig': {"calories": 74, "protein": 0.8, "carbs": 19.2, "fat": 0.3, "fibre": 2.9},
    'date': {"calories": 282, "protein": 2.5, "carbs": 75, "fat": 0.4, "fibre": 8},
    'apricot': {"calories": 48, "protein": 1.4, "carbs": 11.1, "fat": 0.4, "fibre": 2},
    'nectarine': {"calories": 44, "protein": 1.1, "carbs": 10.6, "fat": 0.3, "fibre": 1.7},
    'grapefruit': {"calories": 42, "protein": 0.8, "carbs": 10.7, "fat": 0.1, "fibre": 1.6},
    'tangerine': {"calories": 53, "protein": 0.8, "carbs": 13.3, "fat": 0.3, "fibre": 1.8},
    'mandarin': {"calories": 53, "protein": 0.8, "carbs": 13.3, "fat": 0.3, "fibre": 1.8},
    'custard apple': {"calories": 101, "protein": 1.7, "carbs": 25.2, "fat": 0.6, "fibre": 2.4},
    'sapodilla': {"calories": 83, "protein": 0.4, "carbs": 19.9, "fat": 1.1, "fibre": 5.3},
    'star fruit': {"calories": 31, "protein": 1, "carbs": 6.7, "fat": 0.3, "fibre": 2.8},
    'mulberry': {"calories": 43, "protein": 1.4, "carbs": 9.8, "fat": 0.4, "fibre": 1.7},
    'indian gooseberry': {"calories": 44, "protein": 0.9, "carbs": 10.2, "fat": 0.6, "fibre": 4.3},
    'jamun': {"calories": 60, "protein": 0.7, "carbs": 15.6, "fat": 0.2, "fibre": 0.6},
    'longan': {"calories": 60, "protein": 1.3, "carbs": 15.1, "fat": 0.1, "fibre": 1.1},
    'persimmon': {"calories": 70, "protein": 0.6, "carbs": 18.6, "fat": 0.2, "fibre": 3.6},
}

FRUIT_EMOJI = {
    'apple': '🍎',
    'banana': '🍌',
    'orange': '🍊',
    'mango': '🥭',
    'grapes': '🍇',
    'watermelon': '🍉',
    'strawberry': '🍓',
    'pineapple': '🍍',
    'papaya': '🧡',
    'pear': '🍐',
    'peach': '🍑',
    'plum': '🟣',
    'cherry': '🍒',
    'kiwi': '🥝',
    'pomegranate': '🔴',
    'guava': '🍈',
    'coconut': '🥥',
    'lemon': '🍋',
    'lime': '🍋',
    'avocado': '🥑',
    'blueberry': '🫐',
    'raspberry': '🔴',
    'blackberry': '🫐',
    'cranberry': '🔴',
    'dragon fruit': '🐉',
    'passion fruit': '🟣',
    'lychee': '🍒',
    'rambutan': '🍒',
    'jackfruit': '🍈',
    'durian': '🍈',
    'fig': '🟣',
    'date': '🌴',
    'apricot': '🍑',
    'nectarine': '🍑',
    'grapefruit': '🍊',
    'tangerine': '🍊',
    'mandarin': '🍊',
    'custard apple': '🍈',
    'sapodilla': '🤎',
    'star fruit': '⭐',
    'mulberry': '🫐',
    'indian gooseberry': '🟢',
    'jamun': '🟣',
    'longan': '🟤',
    'persimmon': '🟠',
}

# ------------------------------ Styling ------------------------------
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp {
    background: #edf2f0;
    color: #26383a;
}
[data-testid="stHeader"] {
    background: rgba(237,242,240,.94);
}
[data-testid="stSidebar"] {
    background: #e2ebe7;
    border-right: 1px solid #d0ddd7;
}
[data-testid="stSidebar"] * {
    color: #304442;
}
.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3, h4 {
    color: #243b38 !important;
    letter-spacing: -.02em;
}
p, label, .stCaption {
    color: #586b67;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #f5f9f7);
    border: 1px solid #d8e4de;
    padding: 17px 19px;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(39,72,61,.045);
}
[data-testid="stMetricLabel"] {
    color: #687b75 !important;
}
[data-testid="stMetricValue"] {
    color: #294b43 !important;
}
[data-testid="stFileUploader"] {
    background: #f8fbf9;
    border: 1px dashed #9eb9ac;
    border-radius: 16px;
    padding: 10px;
}
[data-testid="stFileUploader"] section {
    background: transparent;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid #c7d8cf;
    min-height: 2.7rem;
    font-weight: 650;
    color: #31574b;
    background: #f8fbf9;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #6e9b87;
    color: #234d40;
    background: #edf5f0;
}
.stButton > button[kind="primary"] {
    background: #548b72;
    color: #ffffff;
    border: 1px solid #548b72;
}
.stButton > button[kind="primary"]:hover {
    background: #43785f;
    border-color: #43785f;
    color: #ffffff;
}
.stTabs [data-baseweb="tab"] {
    color: #657871;
}
.stTabs [aria-selected="true"] {
    color: #39775c !important;
}
hr {
    border-color: #d5e0da;
}
.hero {
    background:
        radial-gradient(circle at 86% 18%, rgba(107,158,128,.18), transparent 31%),
        linear-gradient(120deg, #dce9e1, #f5f8f5);
    border: 1px solid #ccddd2;
    border-radius: 22px;
    padding: 28px 30px;
    margin: 8px 0 22px;
}
.eyebrow {
    color: #477c61;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 2.35rem;
    margin: .45rem 0 .35rem;
    color: #243b38 !important;
}
.hero p {
    color: #5b7068;
    max-width: 720px;
    margin: 0;
    font-size: 1rem;
}
.panel {
    background: #f9fbfa;
    border: 1px solid #d9e4de;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 3px 12px rgba(39,72,61,.035);
}
.muted {
    color: #6b7d76;
    font-size: .86rem;
}
.pill {
    display: inline-block;
    background: #e0eee5;
    color: #397354;
    border: 1px solid #c3dacb;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 750;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #d8e3dd;
    border-radius: 12px;
    overflow: hidden;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
    background: #fbfdfc;
    border-color: #cbdad2;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading FruitLens vision model…")
def load_model():
    detector = YOLO("yolov8s-worldv2.pt")
    detector.set_classes(FRUIT_CLASSES)
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
            if name in FRUIT_CLASSES and score >= confidence:
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({"fruit": name, "confidence": score, "box": xyxy})
    return annotated, detections


def summarize_detections(detections):
    grouped = {}
    for item in detections:
        grouped.setdefault(item["fruit"], []).append(item["confidence"])
    rows = []
    for fruit, scores in sorted(grouped.items()):
        rows.append({
            "Fruit": f"{FRUIT_EMOJI.get(fruit, '🍎')} {fruit.title()}",
            "Count": len(scores),
            "Average confidence": round(sum(scores) / len(scores) * 100, 1),
            "Highest confidence": round(max(scores) * 100, 1),
        })
    return pd.DataFrame(rows), grouped


def nutrition_totals(grouped):
    totals = {"Calories (kcal)": 0.0, "Protein (g)": 0.0, "Carbs (g)": 0.0, "Fat (g)": 0.0, "Fibre (g)": 0.0}
    for fruit, scores in grouped.items():
        data = FRUIT_NUTRITION.get(fruit)
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
        "Unique fruits": len(record["grouped"]),
        "Top confidence (%)": round(max((d["confidence"] for d in record["detections"]), default=0) * 100, 1),
    }


if "history" not in st.session_state:
    st.session_state.history = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = []

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("# 🍎 FruitLens AI")
    st.caption("FRUIT INTELLIGENCE PLATFORM")
    st.markdown('<span class="pill">● YOLO-WORLD ENGINE</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Analyze fruits", "Detection history", "Model & project"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SYSTEM CONFIGURATION**")
    st.caption("Vision model")
    st.code("YOLOv8s-World", language=None)
    st.caption("Configured fruit classes")
    st.write(f"**{len(FRUIT_CLASSES)} categories**")
    st.caption("Analysis mode")
    st.write("Fruit recognition + nutrition estimates")
    st.divider()
    st.caption("FruitLens AI · Computer Vision Project")

# ------------------------------ Header ------------------------------
st.markdown('<div class="hero"><div class="eyebrow">AI-POWERED FRUIT RECOGNITION</div><h1>Fruit intelligence, at a glance.</h1><p>Detect supported fruits from images, inspect model confidence, explore detection trends and generate an educational nutrition summary.</p></div>', unsafe_allow_html=True)

# ------------------------------ Overview ------------------------------
if page == "Overview":
    st.subheader("Workspace overview")
    st.caption("Your live session activity and FruitLens capabilities.")
    records = st.session_state.history
    total_detections = sum(r["count"] for r in records)
    unique_fruits = sorted({d["fruit"] for r in records for d in r["detections"]})
    avg_conf = [d["confidence"] for r in records for d in r["detections"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images analyzed", len(records))
    c2.metric("Fruit instances", total_detections)
    c3.metric("Unique categories", len(unique_fruits))
    c4.metric("Average confidence", f"{sum(avg_conf)/len(avg_conf)*100:.1f}%" if avg_conf else "—")
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Recent analysis")
        if records:
            recent = pd.DataFrame([make_history_row(r) for r in records[-6:][::-1]])
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="panel"><h3>Ready for your first scan</h3><p class="muted">Upload one or more fruit images in Analyze fruits to populate your activity, charts and history.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Detection distribution")
        counts = {}
        for record in records:
            for detection in record["detections"]:
                counts[detection["fruit"].title()] = counts.get(detection["fruit"].title(), 0) + 1
        if counts:
            st.bar_chart(pd.Series(counts, name="Detections").sort_values(ascending=False))
        else:
            st.markdown('<div class="panel"><p class="muted">Your fruit-category chart will appear here after your first analysis.</p></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Platform modules")
    cards = st.columns(3)
    modules = [
        ("01", "Visual detection", "Annotated image previews, object counts and confidence scores."),
        ("02", "Nutrition insights", "Reference nutrition estimates linked to each detected category."),
        ("03", "Analysis history", "Session activity, category trends and CSV export."),
    ]
    for col, (number, title, desc) in zip(cards, modules):
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">MODULE {number}</div><h3>{title}</h3><p class="muted">{desc}</p></div>', unsafe_allow_html=True)

# ------------------------------ Analyze ------------------------------
elif page == "Analyze fruits":
    st.subheader("Image analysis")
    st.caption("Upload fruit images, adjust detection settings and run the YOLO-World fruit detection model.")
    uploaded_files = st.file_uploader(
        "Choose fruit images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can select multiple images for batch analysis.",
    )
    with st.expander("Detection settings", expanded=False):
        s1, s2, s3 = st.columns(3)
        confidence = s1.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
        iou = s2.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
        max_det = s3.slider("Maximum detections / image", 10, 200, 100, 10)
    if uploaded_files:
        st.info(f"{len(uploaded_files)} image(s) selected. Review the previews, then start analysis.")
        preview_cols = st.columns(min(3, len(uploaded_files)))
        for index, file in enumerate(uploaded_files):
            with preview_cols[index % len(preview_cols)]:
                preview = Image.open(BytesIO(file.getvalue())).convert("RGB")
                st.image(preview, caption=file.name, use_container_width=True)
        if st.button("Run AI analysis", type="primary", use_container_width=True):
            batch_results = []
            progress = st.progress(0, text="Preparing images…")
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
                    st.error(f"Could not analyze {file.name}: {exc}")
                progress.progress((index + 1) / len(uploaded_files), text=f"Processed {index + 1} of {len(uploaded_files)} image(s)")
            st.session_state.latest_results = batch_results
            progress.empty()
            if batch_results:
                st.success(f"Analysis complete. {len(batch_results)} image(s) processed.")
    elif not st.session_state.latest_results:
        st.markdown('<div class="panel"><h3>Start with an image</h3><p class="muted">Supported formats: JPG, JPEG and PNG. For best results, use clear images with visible fruit items.</p></div>', unsafe_allow_html=True)

    if st.session_state.latest_results:
        st.divider()
        st.markdown("### Latest analysis results")
        all_detections = [d for r in st.session_state.latest_results for d in r["detections"]]
        unique_count = len(set(d["fruit"] for d in all_detections))
        avg = sum(d["confidence"] for d in all_detections) / len(all_detections) if all_detections else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images processed", len(st.session_state.latest_results))
        m2.metric("Fruit instances", len(all_detections))
        m3.metric("Fruit categories", unique_count)
        m4.metric("Mean confidence", f"{avg*100:.1f}%" if all_detections else "—")
        for idx, record in enumerate(st.session_state.latest_results, start=1):
            with st.expander(f"{idx}. {record['filename']} · {record['count']} detection(s)", expanded=(idx == 1)):
                img_col, data_col = st.columns([1.15, 1])
                with img_col:
                    st.image(record["annotated"], caption="YOLO-World annotated preview", use_container_width=True)
                with data_col:
                    if record["detections"]:
                        detection_df, grouped = summarize_detections(record["detections"])
                        st.dataframe(detection_df, use_container_width=True, hide_index=True)
                        totals = nutrition_totals(grouped)
                        st.markdown("**Estimated nutrition summary**")
                        n1, n2 = st.columns(2)
                        n1.metric("Calories", f"{totals['Calories (kcal)']:.0f} kcal")
                        n2.metric("Protein", f"{totals['Protein (g)']:.1f} g")
                        n3, n4 = st.columns(2)
                        n3.metric("Carbs", f"{totals['Carbs (g)']:.1f} g")
                        n4.metric("Fibre", f"{totals['Fibre (g)']:.1f} g")
                    else:
                        st.warning("No supported fruit varieties were detected in this image. Try a clearer image or adjust the confidence threshold.")
        if all_detections:
            st.divider()
            st.markdown("### Batch detection breakdown")
            batch_counts = {}
            for detection in all_detections:
                batch_counts[detection["fruit"].title()] = batch_counts.get(detection["fruit"].title(), 0) + 1
            st.bar_chart(pd.Series(batch_counts, name="Detected instances").sort_values(ascending=False))
            export_rows = [{"Image": r["filename"], "Fruit": d["fruit"], "Confidence (%)": round(d["confidence"] * 100, 2), "X1": d["box"][0], "Y1": d["box"][1], "X2": d["box"][2], "Y2": d["box"][3]} for r in st.session_state.latest_results for d in r["detections"]]
            st.download_button("Download detection results (CSV)", pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"), file_name="fruitlens_detection_results.csv", mime="text/csv")
            st.caption("Nutrition estimates use reference values per 100 g and assume 100 g per detected instance. They are illustrative only; actual portion sizes and recipes vary.")

# ------------------------------ History ------------------------------
elif page == "Detection history":
    st.subheader("Detection history")
    st.caption("Review the analyses performed during this browser session.")
    records = st.session_state.history
    if records:
        history_df = pd.DataFrame([make_history_row(r) for r in records[::-1]])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        fruit_counts = {}
        for record in records:
            for detection in record["detections"]:
                fruit_counts[detection["fruit"].title()] = fruit_counts.get(detection["fruit"].title(), 0) + 1
        left, right = st.columns(2)
        with left:
            st.markdown("### Most detected fruits")
            st.bar_chart(pd.Series(fruit_counts, name="Instances").sort_values(ascending=False))
        with right:
            st.markdown("### Session summary")
            st.metric("Images analyzed", len(records))
            st.metric("Total detected instances", sum(r["count"] for r in records))
            st.metric("Distinct categories", len(fruit_counts))
        export_history = []
        for record in records:
            for detection in record["detections"]:
                export_history.append({"Date & time": record["timestamp"], "Image": record["filename"], "Fruit": detection["fruit"], "Confidence (%)": round(detection["confidence"] * 100, 2)})
        if export_history:
            st.download_button("Export full history (CSV)", pd.DataFrame(export_history).to_csv(index=False).encode("utf-8"), file_name="fruitlens_history.csv", mime="text/csv")
        if st.button("Clear session history", type="secondary"):
            st.session_state.history = []
            st.session_state.latest_results = []
            st.rerun()
    else:
        st.markdown('<div class="panel"><h3>No history yet</h3><p class="muted">Your completed scans will appear here. Run an analysis to build your session history.</p></div>', unsafe_allow_html=True)

# ------------------------------ Model / About ------------------------------
else:
    st.subheader("Model & project information")
    st.caption("Technical details about the current FruitLens AI build.")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="eyebrow">COMPUTER VISION</div><h2>YOLOv8s-World</h2><p class="muted">An open-vocabulary object detection model configured with a fixed list of fruit prompts for this project.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="panel"><div class="eyebrow">SUPPORTED LABELS</div><h2>{len(FRUIT_CLASSES)} classes</h2><p class="muted">{", ".join(x.title() for x in FRUIT_CLASSES)}</p></div>', unsafe_allow_html=True)
    st.markdown("### Analysis pipeline")
    steps = st.columns(3)
    for col, number, title, desc in [
        (steps[0], "01", "Image input", "Images are converted to RGB before inference."),
        (steps[1], "02", "Object detection", "YOLO-World returns class labels, confidence scores and bounding boxes."),
        (steps[2], "03", "Insight generation", "Detected labels are summarized and matched to approximate nutrition references."),
    ]:
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">STEP {number}</div><h3>{title}</h3><p class="muted">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("### Configured fruit categories")
    st.write(" · ".join(f"{FRUIT_EMOJI.get(fruit, '🍎')} {fruit.title()}" for fruit in FRUIT_CLASSES))
    st.info("Nutrition values are general reference estimates per 100 g. The application does not measure fruit weight, ingredient quantities or preparation method, so nutrition totals should not be treated as exact dietary information.")

st.divider()
st.markdown('<p style="text-align:center;color:#71819a;font-size:.8rem;">FRUITLENS AI · COMPUTER VISION + FRUIT NUTRITION INSIGHTS · EDUCATIONAL PROJECT</p>', unsafe_allow_html=True)
