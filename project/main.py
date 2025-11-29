import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction

def model_prediction(test_sample):
    model=tf.keras.models.load_model('project/efficient_net.keras')
    image=tf.keras.preprocessing.image.load_img(test_sample,target_size=(128,128))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)
    input_arr= np.array([input_arr])# convert to batch so convert to np array
    prediction= model.predict(input_arr)
    result_index= np.argmax(prediction)
    return result_index
# Sidebar
st.sidebar.title("Dashboard")
app_mode =st.sidebar.selectbox("Select Page",["Home","About","Pest Identification"])

# 
if(app_mode=="Home"):
    st.header("Agricultural Pests Identification") 
    image_path="project/opening1.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown(""" Welcome 
                ### Yay""")
    
elif(app_mode == "About"):
    st.header("About")
    st.markdown("About Dataset")

elif(app_mode== "Pest Identification"):
    st.header("Pest Identification")
    test_image= st.file_uploader("Upload an image:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    if(st.button("Search Pest")):
        with st.spinner("Predicting... Please wait ⏳"):
            st.write("Result")
            result_index= model_prediction(test_image)
        #Defining class
            class_name = ['ants',
                        'bees',
                        'beetle',
                        'catterpillar',
                        'earthworms',
                        'earwig',
                        'grasshopper',
                        'moth',
                        'slug',
                        'snail',
                        'wasp',
                        'weevil']
        st.success("Model has predicted this pest as {}".format(class_name[result_index]))
        st.ballons()
       
        
        