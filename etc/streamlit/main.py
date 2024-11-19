import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from vis_eda import VisualizeMetaData, VisualizeImageAndAnnotation

from time import time

if "train_df" not in st.session_state:
    st.session_state.train_df = pd.DataFrame()
if "output_df" not in st.session_state:
    st.session_state.output_df = pd.DataFrame()
if "flag" not in st.session_state:
    st.session_state.flag = False

st.sidebar.success("🔥HotCLIP")
# st.markdown("<h2 style='text-align: center;'>Segmentation</h2>", unsafe_allow_html=True)
option = st.sidebar.radio("option", ["EDA results", "visualize images & annotations"])

if option == "EDA results":
    instruction_area = st.empty()
    instruction_area.write("please enter the file path to view EDA")

    # 사이드바에 파일 경로 입력 및 확인 버튼
    file_path = st.sidebar.text_input("Enter the file path:", "data/meta_data.xlsx")
    load_data = st.sidebar.button("Load Data")
    
    if load_data:
        try:
            instruction_area.empty()
            eda = VisualizeMetaData(file_path)
            
            st.header("Exploratory Data Analysis")
            eda.plot_preview()

            st.subheader("Sex Distribution")
            eda.plot_sex_distribution()
            
            st.subheader("Histograms")
            eda.plot_histogram()

        except Exception as e:
            st.error(f"An error occured: {e}")
        
elif option == "visualize images & annotations":
    img_path = st.sidebar.text_input("Enter the img path:", "data/train/DCM")
    label_path = st.sidebar.text_input("Enter the label path", "data/train/outputs_json")
    test_path = st.sidebar.text_input("Enter the test path", "data/test/DCM")
    pred_path = None
    # load_data = st.sidebar.button("Load Data")
    
    # if load_data:
    
    sub_option = st.sidebar.radio("option", ["image only", "image with annotation", "ground truth & prediction", "prediction only"])
    
    if sub_option == "image_only":
        pass
    elif sub_option == "image with annotation":
        pass
    elif sub_option == "ground truth & prediction":
        pred_path = st.sidebar.text_input("Enter the pred csv path", "outputs/saved_models/temp_2020/output_baseline_100ep.csv")
    elif sub_option == "prediction only":
        pred_path = st.sidebar.text_input("Enter the pred csv path", "outputs/saved_models/temp_2020/output_baseline_100ep.csv")
        
    load_data = st.sidebar.button("Load Data")    
    if load_data:
        st.session_state.eda = VisualizeImageAndAnnotation(img_path, label_path, test_path, pred_path)
        st.session_state.flag = True
    
    if st.session_state.flag:
        if sub_option in {'image only', 'image with annotation'}:
            index = st.sidebar.number_input("Enter image index:", min_value=0, max_value=st.session_state.eda.get_train_count()-1, step=1)
            
            if sub_option == "image only":
                st.session_state.eda.plot_base_img(index)
            elif sub_option == "image with annotation":
                st.session_state.eda.plot_train_annotation(index)
            
        else:
            index = st.sidebar.number_input("Enter image index:", min_value=0, max_value=st.session_state.eda.get_test_count()-1, step=1)
            st.session_state.eda.set_csv(pred_path)
            st.session_state.eda.plot_pred_only(index)
            
else:
    pass