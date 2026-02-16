import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# ----------------------------------
# Streamlit Seiteneinstellungen
# ----------------------------------
st.set_page_config(page_title="KI Ball Klassifikation", page_icon="⚽")

st.title("⚽ KI Ball Klassifikation")
st.write("Lade ein Bild hoch und die KI erkennt, welche Art Ball es ist.")

np.set_printoptions(suppress=True)

# ----------------------------------
# Modell & Labels laden (nur 1x)
# ----------------------------------
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# ----------------------------------
# Bild Upload
# ----------------------------------
uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # ----------------------------------
    # Bild vorbereiten (wie Teachable Machines)
    # ----------------------------------
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ----------------------------------
    # Vorhersage
    # ----------------------------------
    prediction = model.predict(data)
    prediction = prediction[0]

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[index])

    # ----------------------------------
    # Ergebnis anzeigen
    # ----------------------------------
    st.subheader("🔎 Ergebnis")
    st.success(f"Erkannter Ball: **{class_name[2:]}**")
    st.info(f"Sicherheit: **{confidence_score * 100:.2f}%**")
    st.progress(confidence_score)

    # ----------------------------------
    # Top 3 Vorhersagen
    # ----------------------------------
    st.subheader("📊 Top 3 Vorhersagen")

    top_3_indices = prediction.argsort()[-3:][::-1]

    for i in top_3_indices:
        name = class_names[i].strip()[2:]
        score = float(prediction[i])
        st.write(f"**{name}** – {score * 100:.2f}%")
