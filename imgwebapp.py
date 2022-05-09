import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Potato Image Classifier")
st.text("Provide an image of Potato Leaf")

menu = ["Image","Camera"]
choice = st.sidebar.selectbox("Menu",menu)


@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/1')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


if choice == "Image":
    st.subheader("Image")
    file= st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if file is not None:

        image= Image.open(file)
    
        img_array = np.array(image)
    
        img_tensor = tf.cast(img_array, tf.float32)
        img = tf.image.resize(img_tensor,[256,256])
        img = np.expand_dims(img, axis = 0)
        
        st.write("Predicted Class :")
        with st.spinner('classifying.....'):
          pred = model.predict(img)
          
          label =np.argmax(pred,axis=1)
          
          confidence = "{:.2f}".format(100*np.max(pred))
          
          st.write(classes[label[0]]) 
          
          st.write("Confidence :", confidence) 
          st.write("")
          image = Image.open(file)
          st.image(image, caption='Classifying Potato Image', width=400)


elif choice == "Camera":
  st.subheader("Camera")
  
  picture = st.camera_input("Take a picture")

  if picture:
      st.image(picture)


