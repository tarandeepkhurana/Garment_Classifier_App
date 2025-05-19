import streamlit as st
from predict import predict_image
from PIL import Image

st.set_page_config(layout="wide")

col1, col2 = st.columns([2,2])

with col1:
    st.title("👕 Garment Classifier")
    st.markdown("""
The model is trained to predict among the following garment types:  
1. Long sleeve dress  
2. Long sleeve top  
3. Short sleeve dress  
4. Short sleeve top  
5. Shorts  
6. Skirts  
7. Trousers  
8. Vest  
9. Vest dress
""")


def resize_image(image, max_size=200):
    ratio = max_size / max(image.size)
    new_size = tuple([int(dim * ratio) for dim in image.size])
    return image.resize(new_size)
            
with col2:
    upload_col, result_col = st.columns([1, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Resize image for display only (won't affect saved image for prediction)
            image_resized = resize_image(image, max_size=300)

            # Display resized image side-by-side with prediction
            # col1, col2 = st.columns([1, 2])


            st.image(image_resized, caption='Uploaded Image', use_container_width=True)

            # Save original image temporarily for prediction
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            pred, conf = predict_image("temp.jpg")

            with result_col:
               st.markdown("### Prediction")
               st.success(f"**{pred}** ({conf*100:.2f}% confidence)")
    

    
