
# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #10251b, #183c2b);
    color: #ffffff;
}

/* Main text */
.stApp p,
.stApp label,
.stApp span,
.stMarkdown,
.stMetric label,
.stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Title */
.main-title {
    text-align: center;
    color: #a5d6a7;
    font-size: 38px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #c8e6c9;
    font-size: 18px;
}

/* Result box */
.result-box {
    background-color: #244936;
    color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #81c784;
    margin-top: 15px;
}

.result-box h3,
.result-box h4 {
    color: #ffffff !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background-color: #244936;
    padding: 15px;
    border-radius: 10px;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #81c784;
}

</style>
""", unsafe_allow_html=True)
