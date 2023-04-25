import streamlit as st
from PIL import Image
import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mtcnn
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

face_dict=joblib.load(open('faces_embedding.pkl','rb'))
mtcnn_model=mtcnn.MTCNN()
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

    
def save_img(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False

selfie=st.file_uploader("Enter your selfie")
if selfie :

    if save_img(selfie):
    
        
        img=os.path.join('uploads',selfie.name)
        img=cv2.imread(img)
        st.image(img)
        face_box=mtcnn_model.detect_faces(img)
        x1,y1,w,h=face_box[0]['box']
        cropped_face=img[y1:y1+h,x1:x1+w]
        cropped_face=cv2.resize(cropped_face,(224,224))
        cropped_face=image.img_to_array(cropped_face)
        cropped_face=np.expand_dims(cropped_face,axis=0)
        cropped_face=preprocess_input(cropped_face)
        emb=model.predict(cropped_face)
        st.text(emb) 
        st.text(emb.shape)
        print('done')


submit=st.button('Press to search you Images')
if submit:
    for file,embeds in face_dict.items():
        for j in embeds:
            simi=cosine_similarity(j,emb)
            if simi > 0.65:
                result_img=Image.open(file)
                st.image(result_img)
                break