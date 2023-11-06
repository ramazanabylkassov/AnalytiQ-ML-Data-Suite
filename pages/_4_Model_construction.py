import streamlit as st

st.set_page_config(layout="wide", page_title='Model construction', page_icon='💻')

import sys
sys.path.append("..")
from functions import *

if 'fit_models' not in st.session_state:
    st.session_state.fit_models = {}

def main():
    if 'homepage_param' in st.session_state and st.session_state.homepage_param['file_uploaded']:
        processed_df = st.session_state.data_process['processed_df'].copy()
        target_feature = st.session_state.homepage_param['target_feature']

        st.markdown(f"<h1 style='text-align: center'>{st.session_state.dataframe['df_name'].title()}</h1>", unsafe_allow_html=True)
        st.write('---')

        st.markdown(f"<h1 style='text-align: center'>Model construction</h1>", unsafe_allow_html=True)
        st.write('---')
        dataframe_exploration(tabs_names=['Processed dataframe', 'Statistical info'], dataframes_dict={'Processed dataframe': processed_df})
        container_metrics = st.container()
        container_cv_analysis = st.container()
        container_FI_analysis = st.container()

        with st.sidebar:
            change_page_buttons(key='top', pages=['3 Data processing', '5 Model deployment'])
            st.write(f"<h1 style='text-align: center'>Control panel</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center'>Supervised Machine Learning</h2>", unsafe_allow_html=True)
            st.write('---')
            st.markdown(f"<h2 style='text-align: center'>Determine train/test ratio</h2>", unsafe_allow_html=True)
            train_test_data = {}
            train_test_data['X_train'], train_test_data['X_test'], train_test_data['y_train'], train_test_data['y_test'] = train_test(df=processed_df, target_feature=target_feature)
            st.write('---')

            if st.session_state.homepage_param['target_feature_type'] == 'Numeric':
                ML_regression_models(train_test_data=train_test_data, container_metrics=container_metrics, container_cv_analysis=container_cv_analysis, container_FI_analysis=container_FI_analysis)
            else:
                KNN_switch = st.toggle(label="K-NN classification")
                SVC_switch = st.toggle(label="SVC classification")
                logistic_regr_switch = st.toggle(label="Logistic regression")
                decision_tree_switch = st.toggle(label="Decision Tree Classifier")
                random_forest_switch = st.toggle(label="Random Tree Classifier")
        
            change_page_buttons(key='bottom', pages=['3 Data processing', '5 Model deployment'])

    else:
        st.write(f"<h2 style='text-align: center'>Upload file at the Homepage</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            st_lottie(data_animation_random(key='no file found'), height=300, width=300, key='model construction lottie')

if __name__ == '__main__':
    main()
