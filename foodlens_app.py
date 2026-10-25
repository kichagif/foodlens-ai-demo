import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2

# ==========================================================
# FOODLENS AI
# AI FOOD RECOGNITION & NUTRITION ANALYZER
# ==========================================================

st.set_page_config(
    page_title="FoodLens AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM DESIGN
# ==========================================================

st.markdown("""
<style>

.main-title {
    font-size: 52px;
    font-weight: 850;
    letter-spacing: -1.5px;
    margin-bottom: 2px;
    line-height: 1.05;
}

.main-subtitle {
    font-size: 19px;
    color: #6b7280;
    margin-top: 4px;
    line-height: 1.5;
}

.hero-card {
    padding: 26px 28px;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,0.18);
    background: linear-gradient(
        135deg,
        rgba(128,128,128,0.08),
        rgba(128,128,128,0.025)
    );
    margin: 18px 0 22px 0;
}

.dashboard-card {
    padding: 22px;
    min-height: 128px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.055);
    box-shadow: 0 5px 18px rgba(0,0,0,0.035);
    margin-bottom: 15px;
}

.dashboard-card h3 {
    margin: 5px 0 3px 0;
    font-size: 23px;
}

.dashboard-card p {
    margin: 0;
    color: #6b7280;
}

.section-title {
    font-size: 27px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 5px;
}

.small-label {
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 700;
}

.analysis-badge {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.06);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.result-card {
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.035);
    margin: 8px 0;
}

.food-breakdown-note {
    padding: 14px 17px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.04);
    margin: 10px 0 18px 0;
}

.footer {
    text-align: center;
    color: #777777;
    padding: 28px;
    font-size: 13px;
    border-top: 1px solid rgba(128,128,128,0.15);
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# NUTRITION DATABASE
# Approximate values per 100g
# ==========================================================

nutrition = {

    "apple": {
        "calories": 52,
        "protein": 0.3,
        "carbs": 13.8,
        "fat": 0.2,
        "fibre": 2.4
    },

    "banana": {
        "calories": 89,
        "protein": 1.1,
        "carbs": 22.8,
        "fat": 0.3,
        "fibre": 2.6
    },

    "orange": {
        "calories": 47,
        "protein": 0.9,
        "carbs": 11.8,
        "fat": 0.1,
        "fibre": 2.4
    },

    "mango": {
        "calories": 60,
        "protein": 0.8,
        "carbs": 15.0,
        "fat": 0.4,
        "fibre": 1.6
    },

    "grapes": {
        "calories": 69,
        "protein": 0.7,
        "carbs": 18.1,
        "fat": 0.2,
        "fibre": 0.9
    },

    "watermelon": {
        "calories": 30,
        "protein": 0.6,
        "carbs": 7.6,
        "fat": 0.2,
        "fibre": 0.4
    },

    "strawberry": {
        "calories": 32,
        "protein": 0.7,
        "carbs": 7.7,
        "fat": 0.3,
        "fibre": 2.0
    },

    "pineapple": {
        "calories": 50,
        "protein": 0.5,
        "carbs": 13.1,
        "fat": 0.1,
        "fibre": 1.4
    },

    "papaya": {
        "calories": 43,
        "protein": 0.5,
        "carbs": 10.8,
        "fat": 0.3,
        "fibre": 1.7
    },

    "broccoli": {
        "calories": 34,
        "protein": 2.8,
        "carbs": 6.6,
        "fat": 0.4,
        "fibre": 2.6
    },

    "carrot": {
        "calories": 41,
        "protein": 0.9,
        "carbs": 9.6,
        "fat": 0.2,
        "fibre": 2.8
    },

    "sandwich": {
        "calories": 250,
        "protein": 10.0,
        "carbs": 30.0,
        "fat": 10.0,
        "fibre": 2.0
    },

    "pizza": {
        "calories": 266,
        "protein": 11.0,
        "carbs": 33.0,
        "fat": 10.0,
        "fibre": 2.3
    },

    "donut": {
        "calories": 452,
        "protein": 4.9,
        "carbs": 51.0,
        "fat": 25.0,
        "fibre": 2.0
    },

    "cake": {
        "calories": 257,
        "protein": 4.0,
        "carbs": 53.0,
        "fat": 3.0,
        "fibre": 0.5
    },

    "hot dog": {
        "calories": 290,
        "protein": 10.0,
        "carbs": 18.0,
        "fat": 20.0,
        "fibre": 1.0
    }
}

# ==========================================================
# FOOD EMOJIS
# ==========================================================

food_emojis = {
    "apple": "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "strawberry": "🍓",
    "pineapple": "🍍",
    "papaya": "🧡",
    "broccoli": "🥦",
    "carrot": "🥕",
    "sandwich": "🥪",
    "pizza": "🍕",
    "donut": "🍩",
    "cake": "🍰",
    "hot dog": "🌭"
}

# ==========================================================
# LOAD AI MODEL
# ==========================================================

# Open-vocabulary food classes. YOLO-World can be prompted with custom
# categories, unlike the original COCO-only YOLO11n model.
FOOD_CLASSES = [
    "apple", "banana", "orange", "pizza", "cake", "donut",
    "mango", "grapes", "watermelon", "strawberry", "pineapple",
    "papaya", "carrot", "broccoli", "sandwich"
]


@st.cache_resource
def load_model():
    # This model downloads its weights on first launch; allow extra startup time.
    detector = YOLO("yolov8s-worldv2.pt")
    detector.set_classes(FOOD_CLASSES)
    return detector


model = load_model()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🍎 FoodLens AI")

    st.caption(
        "AI Food Recognition & Nutrition Analyzer"
    )

    st.divider()

    st.markdown("### 🧠 AI System")

    st.success("● System Ready")

    st.write("**Computer Vision Model**")
    st.code("YOLO-World (custom food classes)")

    st.write("**Analysis Mode**")
    st.write("Educational Nutrition Analysis")

    st.divider()

    st.markdown("### 🔎 Detection")

    st.markdown(
        '<div class="food-breakdown-note">'
        '<strong>AI Food Recognition</strong><br>'
        'Upload a clear food image and FoodLens will analyze '
        'the visible food items using computer vision.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "FoodLens connects computer-vision detection "
        "with a nutrition database to generate an "
        "educational estimate."
    )

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🍎 FoodLens AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI Food Recognition & Nutrition Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-card">'
    '<span class="analysis-badge">● AI-POWERED FOOD ANALYSIS</span>'
    '<h3 style="margin:12px 0 6px 0;">Turn a food image into useful insights.</h3>'
    '<p style="color:#6b7280; margin:0;">'
    'Upload a clear food image and FoodLens uses computer vision '
    'to identify supported food items and connect them to an '
    'educational nutrition database.'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# TOP INFORMATION CARDS
# ==========================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        '<div class="dashboard-card">'
        '<div class="small-label">AI MODEL</div>'
        '<h3>YOLO-World</h3>'
        '<p>Real-time object detection</p>'
        '</div>',
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        '<div class="dashboard-card">'
        '<div class="small-label">FOOD DETECTION</div>'
        '<h3>AI Detection</h3>'
        '<p>Recognizes supported food items</p>'
        '</div>',
        unsafe_allow_html=True
    )

with info3:

    st.markdown(
        '<div class="dashboard-card">'
        '<div class="small-label">OUTPUT</div>'
        '<h3>Nutrition</h3>'
        '<p>Calories • Macros • Fibre</p>'
        '</div>',
        unsafe_allow_html=True
    )

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.markdown(
    '<div class="section-title">📸 Food Image Analysis</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a clear image of your food",
    type=["jpg", "jpeg", "png"],
    help="For best results, make sure the food is clearly visible."
)

# ==========================================================
# MAIN ANALYSIS
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.write("")

    image_col, details_col = st.columns([1.6, 1])

    with image_col:

        st.markdown("### 🖼️ Uploaded Image")

        st.image(
            image,
            caption="Original Food Image",
            width="stretch"
        )

    with details_col:

        st.markdown("### 🔎 Analysis Ready")

        st.info(
            "Your image is ready. Start the AI analysis to detect "
            "food items and generate the nutrition estimate."
        )

        analyze_button = st.button(
            "🔍 Analyze Food",
            type="primary",
            width="stretch"
        )

    if analyze_button:

        with st.spinner(
            "🤖 FoodLens AI is analyzing the image..."
        ):

            # Use one confidence threshold for both the preview and the summary.
            # YOLO applies Non-Maximum Suppression (NMS) during prediction.
            results = model.predict(
                source=image,
                conf=0.35,
                iou=0.50,
                max_det=30,
                verbose=False
            )

            # Draw the detections once, with smaller and cleaner labels.
            # Ultralytics returns a BGR NumPy image; convert it to RGB for Streamlit.
            annotated = results[0].plot(
                labels=True,
                conf=True,
                line_width=2,
                font_size=12
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Store EVERY detected instance, not just one confidence per food type.
            # Example: 5 visible apples should count as 5 apple detections.
            detected_foods = {}

            for result in results:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    food_name = model.names[class_id].lower()

                    if confidence >= 0.35 and food_name in nutrition:
                        detected_foods.setdefault(food_name, []).append(confidence)

        st.success("✅ AI analysis complete — results are ready below.")

        # ==================================================
        # AI DETECTION RESULTS
        # ==================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 AI Detection Results'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            annotated,
            caption="AI Detection Preview",
            width="stretch"
        )

        # ==================================================
        # DETECTED FOODS
        # ==================================================

        if detected_foods:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🍽️ Detected Foods'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="food-breakdown-note">'
                '<strong>How to read this:</strong> confidence shows '
                'how strongly the vision model matched each detected item. '
                'Nutrition values are educational estimates based on '
                'reference values per 100g.'
                '</div>',
                unsafe_allow_html=True
            )

            metric1, metric2, metric3 = st.columns(3)

            all_confidences = [
                confidence
                for confidences in detected_foods.values()
                for confidence in confidences
            ]
            highest_confidence = max(all_confidences)
            total_detected_instances = len(all_confidences)

            metric1.metric(
                "Detected Food Items",
                total_detected_instances
            )

            metric2.metric(
                "Highest Confidence",
                f"{highest_confidence * 100:.1f}%"
            )

            metric3.metric(
                "AI Status",
                "Ready"
            )

            st.write("")

            # ==================================================
            # CONFIDENCE RESULTS
            # ==================================================

            for food, confidences in detected_foods.items():

                left, middle, right = st.columns([2, 2, 1])
                average_confidence = sum(confidences) / len(confidences)
                item_count = len(confidences)

                with left:
                    emoji = food_emojis.get(food, "🍽️")
                    st.markdown(f"### {emoji} {food.title()} × {item_count}")

                with middle:
                    st.progress(min(int(average_confidence * 100), 100))
                    st.caption(f"Average confidence: {average_confidence * 100:.1f}%")

                with right:
                    if average_confidence >= 0.75:
                        st.success("High")
                    elif average_confidence >= 0.50:
                        st.warning("Medium")
                    else:
                        st.warning("Low")

            # ==================================================
            # DETECTION TABLE
            # ==================================================

            st.write("")

            table_rows = []

            for food, confidences in detected_foods.items():

                average_confidence = sum(confidences) / len(confidences)
                if average_confidence >= 0.75:
                    status = "High confidence"
                elif average_confidence >= 0.50:
                    status = "Moderate confidence"
                else:
                    status = "Low confidence"

                table_rows.append(
                    {
                        "Food": food.title(),
                        "Count": len(confidences),
                        "Average Confidence": f"{average_confidence * 100:.1f}%",
                        "Detection Status": status
                    }
                )

            detection_table = pd.DataFrame(
                table_rows
            )

            st.dataframe(
                detection_table,
                width="stretch",
                hide_index=True
            )

            # ==================================================
            # NUTRITION CALCULATION
            # ==================================================

            total_calories = 0
            total_protein = 0
            total_carbs = 0
            total_fat = 0
            total_fibre = 0

            for food, confidences in detected_foods.items():

                data = nutrition[food]
                item_count = len(confidences)

                total_calories += data["calories"] * item_count
                total_protein += data["protein"] * item_count
                total_carbs += data["carbs"] * item_count
                total_fat += data["fat"] * item_count
                total_fibre += data["fibre"] * item_count

            # ==================================================
            # DETECTED FOOD BREAKDOWN
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🧾 Detected Food Breakdown'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Approximate nutrition values are shown per 100g "
                "for each detected food. AI confidence indicates "
                "how strongly the computer-vision model recognized it."
            )

            breakdown_rows = []

            for food, confidences in detected_foods.items():

                data = nutrition[food]
                emoji = food_emojis.get(food, "🍽️")
                item_count = len(confidences)
                average_confidence = sum(confidences) / item_count

                breakdown_rows.append(
                    {
                        "Food": f"{emoji} {food.title()}",
                        "Count": item_count,
                        "Average AI Confidence": f"{average_confidence * 100:.1f}%",
                        "Calories (estimate)": f"{data['calories'] * item_count:.0f} kcal",
                        "Protein": f"{data['protein'] * item_count:.1f} g",
                        "Carbs": f"{data['carbs'] * item_count:.1f} g",
                        "Fat": f"{data['fat'] * item_count:.1f} g",
                        "Fibre": f"{data['fibre'] * item_count:.1f} g"
                    }
                )

            breakdown_table = pd.DataFrame(breakdown_rows)

            st.dataframe(
                breakdown_table,
                width="stretch",
                hide_index=True
            )

            # ==================================================
            # TOTAL NUTRITION SUMMARY
            # ==================================================

            st.markdown("### 🧮 Total Estimated Nutrition")

            total_breakdown_col1, total_breakdown_col2, total_breakdown_col3 = st.columns(3)

            with total_breakdown_col1:
                st.metric(
                    "🔥 Total Calories",
                    f"{total_calories:.0f} kcal"
                )

            with total_breakdown_col2:
                st.metric(
                    "💪 Total Protein",
                    f"{total_protein:.1f} g"
                )

            with total_breakdown_col3:
                st.metric(
                    "🌾 Total Carbohydrates",
                    f"{total_carbs:.1f} g"
                )

            total_breakdown_col4, total_breakdown_col5 = st.columns(2)

            with total_breakdown_col4:
                st.metric(
                    "🥑 Total Fat",
                    f"{total_fat:.1f} g"
                )

            with total_breakdown_col5:
                st.metric(
                    "🌿 Total Fibre",
                    f"{total_fibre:.1f} g"
                )

            st.caption(
                "Estimates assume roughly 100g per detected item. Actual fruit sizes vary, "
                "so these totals are illustrative rather than exact portion measurements."
            )

            # ==================================================
            # NUTRITION DASHBOARD
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📊 Nutrition Dashboard'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Estimated values assume approximately 100g per detected item."
            )

            # FIRST ROW

            n1, n2, n3 = st.columns(3)

            n1.metric(
                "🔥 Calories",
                f"{total_calories:.0f} kcal"
            )

            n2.metric(
                "💪 Protein",
                f"{total_protein:.1f} g"
            )

            n3.metric(
                "🌾 Carbohydrates",
                f"{total_carbs:.1f} g"
            )

            st.write("")

            # SECOND ROW

            n4, n5 = st.columns(2)

            n4.metric(
                "🥑 Fat",
                f"{total_fat:.1f} g"
            )

            n5.metric(
                "🌿 Fibre",
                f"{total_fibre:.1f} g"
            )

            # ==================================================
            # NUTRIENT CHART
            # ==================================================

            st.write("")

            chart_data = pd.DataFrame(
                {
                    "Nutrient": [
                        "Calories",
                        "Protein",
                        "Carbohydrates",
                        "Fat",
                        "Fibre"
                    ],
                    "Value": [
                        total_calories,
                        total_protein,
                        total_carbs,
                        total_fat,
                        total_fibre
                    ]
                }
            )

            st.markdown(
                "### 📈 Nutrient Overview"
            )

            st.bar_chart(
                chart_data.set_index("Nutrient")
            )

            # ==================================================
            # EDUCATIONAL NOTE
            # ==================================================

            st.info(
                "ℹ️ **Educational estimate:** Nutrition values "
                "are approximate and are based on 100g portions "
                "of the detected food types. Actual nutrition "
                "varies with portion size, variety and preparation."
            )

        else:

            st.warning(
                "😕 No supported food was detected."
            )

            st.write(
                "Try a clearer image with the food clearly visible "
                "and well separated from the background."
            )

else:

    st.info(
        "👆 Upload an image above to begin your FoodLens AI analysis."
    )

# ==========================================================
# HOW FOODLENS WORKS
# ==========================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🧠 How FoodLens AI Works'
    '</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3)

with step1:

    st.markdown(
        '<div class="dashboard-card">'
        '<h3>01 • Upload</h3>'
        '<p>'
        'The user provides an image containing food.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

with step2:

    st.markdown(
        '<div class="dashboard-card">'
        '<h3>02 • Detect</h3>'
        '<p>'
        'YOLO computer vision analyzes the image and '
        'identifies supported food objects.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

with step3:

    st.markdown(
        '<div class="dashboard-card">'
        '<h3>03 • Analyze</h3>'
        '<p>'
        'Detected food names are connected to a nutrition '
        'database to generate an educational summary.'
        '</p>',
        unsafe_allow_html=True
    )

# ==========================================================
# ABOUT TECHNOLOGY
# ==========================================================

with st.expander("🔬 About the Technology"):

    st.markdown(
        '<div class="food-breakdown-note">'
        '<strong>Inside FoodLens AI</strong><br>'
        'The project combines computer vision with structured '
        'nutrition data to turn image recognition into an '
        'easy-to-understand educational result.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "**FoodLens AI combines two components:**"
    )

    st.write(
        "• **YOLO-World:** An open-vocabulary detector configured for the selected food classes. "
        "used to detect objects in images."
    )

    st.write(
        "• **Nutrition Database:** A collection of approximate "
        "nutrition values used to generate the analysis."
    )

    st.write(
        "The project demonstrates how artificial intelligence "
        "can connect image recognition with structured data "
        "to create a useful application."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">'
    'FoodLens AI • Educational AI Project • '
    'Computer Vision + Nutrition Analysis'
    '</div>',
    unsafe_allow_html=True
)
