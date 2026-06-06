import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

MODEL_PATH = "models/pdm_pipeline.joblib"

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================
# PREDICT FUNCTION
# =========================

def predict(data):

    prediction = pipeline.predict(data)[0]

    probability = None

    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(data)[0][1]

    return prediction, probability

# =========================
# VALIDATION FUNCTION
# =========================

def validate_input(df):

    # Product ID tidak boleh kosong
    if df["Product ID"].iloc[0].strip() == "":
        return False, "Product ID tidak boleh kosong"

    # Tidak boleh ada nilai kosong
    if df.isnull().sum().sum() > 0:
        return False, "Masih terdapat nilai kosong"

    # Process temperature harus lebih tinggi dari air temperature
    if df["Process temperature [K]"].iloc[0] <= \
       df["Air temperature [K]"].iloc[0]:
        return False, (
            "Process temperature harus lebih tinggi "
            "dari Air temperature"
        )

    return True, None

# =========================
# HEADER
# =========================

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown(
    """
    Dashboard ini digunakan untuk memprediksi kemungkinan **Machine Failure**
    berdasarkan data operasional mesin.
    """
)

# =========================
# SIDEBAR INPUT
# =========================

st.sidebar.header("Input Data Mesin")

udi = st.sidebar.number_input(
    "UDI",
    min_value=0,
    value=9999,
    step=1
)

product_id = st.sidebar.text_input(
    "Product ID",
    value="M14999"
)

machine_type = st.sidebar.selectbox(
    "Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=301.5,
    step=0.1
)

process_temp = st.sidebar.number_input(
    "Process temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=311.0,
    step=0.1
)

rot_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    min_value=0,
    value=1200,
    step=1
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    value=60.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    min_value=0,
    value=220,
    step=1
)

# =========================
# DISPLAY INPUT
# =========================

st.subheader("📋 Data Input")

col1, col2 = st.columns(2)

with col1:
    st.metric("Machine Type", machine_type)
    st.metric("UDI", udi)

with col2:
    st.metric("Rotational Speed", f"{rot_speed} rpm")
    st.metric("Torque", f"{torque} Nm")

input_df = pd.DataFrame([{
    "UDI": udi,
    "Product ID": product_id,
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

st.dataframe(input_df, use_container_width=True)

# =========================
# PREDICTION
# =========================

st.divider()

if st.button("🚀 Predict", use_container_width=True):

    valid, msg = validate_input(input_df)

    if not valid:
        st.error(msg)

    else:

        try:

            prediction, probability = predict(input_df)

            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error("⚠ MACHINE FAILURE PREDICTED")
            else:
                st.success("✅ MACHINE OPERATING NORMALLY")

            if probability is not None:

                st.metric(
                    "Failure Probability",
                    f"{probability:.2%}"
                )

                st.progress(float(probability))

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Predictive Maintenance Dashboard | Machine Learning Deployment using Streamlit"
)

