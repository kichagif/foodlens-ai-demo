
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
