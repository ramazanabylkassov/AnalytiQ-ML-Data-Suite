import streamlit as st
st.set_page_config(layout="wide", page_title='Homepage', page_icon='🏠')
from functions import change_page_buttons, reupload_file_button, preprocess_dataframe, feature_type, dataframe_exploration, upload_data, readme

if 'dataframe' not in st.session_state:
    st.session_state.dataframe = {
        'df_name': None,
        'balloons': False,
        'original': None,
        'pre-processed': None,
    }
if 'homepage_param' not in st.session_state:
    st.session_state.homepage_param = {
        'file_uploaded': False,
        'drop_duplicates_switch': False,
        'dataframe_type': None,
        'target_feature': None,
        'target_feature_type': None,
        'numeric_features': None,
        'categorical_features': None,
        'remaining_features_choice': None,
        'chosen_features': None,
        'form_button_remain_feature_type': False,
        're_upload_button': False,
    }

def main():
    if 'data_process' in st.session_state and st.session_state.data_process['data_processed']:
        st.session_state.data_process['data_processed'] = False
        
    if 'homepage_param' in st.session_state and st.session_state.homepage_param['file_uploaded']:
        if st.session_state.homepage_param['balloons']:
            st.balloons()
            st.session_state.homepage_param['balloons'] = False
        
        st.session_state.regress_form_button = False
            
        df_original = st.session_state.dataframe['original']
        df_custom = df_original.copy()

        st.markdown(f"<h1 style='text-align: center'>{st.session_state.dataframe['df_name'].title()}</h1>", unsafe_allow_html=True)
        st.write('---')
        with st.expander('Control panel', expanded=True):
            container = st.container()

        with st.sidebar:
            change_page_buttons(key='top', pages=['2 Data understanding'])
            reupload_file_button()
            st.write(f"<h1 style='text-align: center'>Control panel</h1>", unsafe_allow_html=True)
            
            st.write('---')
            
            st.markdown(f"<h2 style='text-align: center'>Pre-process data</h2>", unsafe_allow_html=True)
            st.write('###')
            df_custom = preprocess_dataframe(df_custom, container) ### includes feature drop, feature imputer, Ordinal Encoding, One Hot Encoding, removing duplicate rows

            st.write('---')

            st.write(f"<h2 style='text-align: center'>Feature type distribution</h2>", unsafe_allow_html=True)
            st.write('###')
            feature_type(dataframe=df_custom, container=container)
            change_page_buttons(key='bottom', pages=['2 Data understanding'])
        
        st.write(f"<h2 style='text-align: center'>Dataframes</h2>", unsafe_allow_html=True)
        dataframe_exploration(
            tabs_names=['Pre-processed dataframe', 'Original dataframe', 'Statistical info'], 
            dataframes_dict={
                'Pre-processed dataframe': df_custom,
                'Original dataframe': df_original
                })
        # ask_AI(df=df_original)

    else:
        st.markdown(f"<h1 style='text-align: center'>Data Analysis and Machine Learning Toolkit</h1>", unsafe_allow_html=True)
        upload_data()
        readme()

if __name__ == '__main__':
    main()
