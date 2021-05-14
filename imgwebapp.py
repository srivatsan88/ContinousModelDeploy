import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Bean Image Classifier")
st.text("Provide URL of bean Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['angular_leaf_spot','bean_rust','healthy']

def decode_img(image):
  img = tf.image.resize(image,[224,224])
  img = tf.image.decode_jpeg(img, channels=3)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','http://barmac.com.au/wp-content/uploads/sites/3/2016/01/Angular-Leaf-Spot-Beans1.jpg')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Bean Image', use_column_width=True)
