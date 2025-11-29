import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

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
app_mode =st.sidebar.selectbox("Select Page",["Home","About","Model Performance","Pest Identification"])

#
if(app_mode=="Home"):
    st.header("# Agricultural Pest Detection Using Machine Learning") 
    image_path="project/opening1.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""

## Background
Coming from a country where agriculture is a major occupation, I have seen how pest infestations cause major crop losses and financial hardship for farmers. Many still rely on traditional identification methods or broad pesticides, which are often ineffective and harmful to the environment.

## Project Goal
This project aims to solve these challenges using **machine learning for automated pest identification** from images.  
It focuses on:

- **Classifying agricultural pests** from uploaded images  
- **Analyzing environmental factors** such as soil type or plant species and how they impact classification  
- **Studying visual characteristics** (size, color, shape) that are most useful for identification  
- **Learning from image backgrounds** to understand habitats and pest–environment relationships  

## Why This Matters
By answering these questions, this project aims to develop **reliable and accessible pest-detection tools** that help farmers:

- Improve early diagnosis  
- Reduce misuse of pesticides  
- Promote sustainable farming  
- Minimize crop loss  
- Reduce food waste  

## Try the Model
Upload an image of a pest to see the model’s prediction and learn about its characteristics by navigating to the Pest Identification Page in the Dashboard
""")
    
elif(app_mode == "About"):
    st.header("#About")
    st.markdown("""
## About the Dataset

### Primary Dataset: Agricultural Pests Image Dataset from Kaggle

- The dataset contains images of **12 different types** of agricultural pests:  
  ants, bees, beetles, caterpillars, earthworms, earwigs, grasshoppers, moths, slugs, snails, wasps, and weevils.  
- It includes **5,494 images in total**.  
- The distribution is approximately **balanced**, with **about 470–500 images per pest type**.  

You can find the dataset here: [Agricultural Pests Image Dataset on Kaggle](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset)

---

### Purpose & Usage

This dataset is used for training and evaluating the machine learning model that powers this app.  
By using a balanced, diverse dataset across many pest types, the model aims to generalize well — improving accuracy in real-world conditions.

---

### Credits & Disclaimer

- Dataset provided by the original contributors on Kaggle.  
- The dataset labels and images are used *as-is*.  
- By using this dataset, this project does **not** claim ownership of the original images.  
- The model and app are provided for educational and agricultural‑support purposes only.


---

###  What You Can Do

- Upload your own pest image and see whether the app recognizes it correctly.  
- Compare the result with expected pest type.  
 
""")

elif (app_mode== "Model Performance"):
    st.header("Model Performance")
    st.markdown("""
    The pest detection model was trained on a dataset of **12 agricultural pests** and evaluated on a **validation set** — a separate portion of images the model had not seen during training.

    ### Validation Accuracy
    The model achieves **85.70% accuracy** on the validation set.  
    This means it can correctly classify unseen pest images approximately **85.70% of the time**.

    ###  Why Validation Accuracy?
    - Training accuracy shows how well the model fits the training data, but can be overly optimistic due to overfitting.  
    - Validation accuracy provides a **more realistic estimate of real-world performance**, indicating how well the model generalizes to new images.

    ### Note
    While the model performs well, it should be used as a **first-pass identification tool**.
    """)

    # -------------------------
    # Load per-class classification report
    st.subheader("Classification Report (Per-Class Metrics)")
    report_df = pd.read_csv("project\classification_report.csv", index_col=0)  # Save from notebook using output_dict=True
    st.dataframe(report_df.style.format("{:.3f}"))

    # -------------------------
    # Display accuracy plot
    st.subheader("Training vs Validation Accuracy")
    st.image("ptoject\accuracy_plot.png", caption="Training vs Validation Accuracy")

    # -------------------------
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    st.image("project\cm.png", caption="Confusion Matrix")

    # Explain the confusion matrix
    st.markdown("""
    ### Understanding the Confusion Matrix

    - Each **row** represents the **actual class** (true labels).  
    - Each **column** represents the **predicted class** by the model.  
    - **Diagonal values** indicate **correct predictions** — how many samples of a given class were classified correctly.  
    - **Off-diagonal values** show **misclassifications** — which classes were confused by the model.  

    **Example:**  
    If the value in row 'Ant' and column 'Bee' is 3, it means **3 Ant images were incorrectly predicted as Bees**.  

    By analyzing the confusion matrix, you can see which pests the model identifies well and which ones it tends to confuse, helping to improve model performance and understanding.
    """)
    

elif (app_mode== "Pest Identification"):
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
            st.ballons()
        st.success("Model has predicted this pest as {}".format(class_name[result_index]))
        
       
        
        