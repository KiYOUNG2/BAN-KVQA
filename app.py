import streamlit as st
import io
from PIL import Image
from predict import load_model, get_prediction
from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout='wide')

def main():
    st.title("Bilinear Attention Networks for Visual Question Answering")
    
    model = load_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    uploaded_file

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image")
        query = st.text_input("Any Questions?", "")

        if query:
            st.write(f'Question: {query}')
            st.write("Answering...")
            pred = get_prediction(model, query, image_bytes)
            st.write(f"Prediction Response is {pred}")

root_password = "test"

password = st.text_input("password", type="password")

@cache_on_button_press("Authenticate")
def authenticate(password) -> bool:
    st.write(type(password))
    return password == root_password

if authenticate(password):
    st.success("You are authenticated!")
    main()
else:
    st.error("The password is invaild.")