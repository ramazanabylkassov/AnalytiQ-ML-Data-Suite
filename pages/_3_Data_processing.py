import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide", page_title='Data processing', page_icon='⚙️')

import sys
sys.path.append("..")
from functions import process_outliers, Numeric_scale, Target_log, change_page_buttons, dataframe_exploration, st_lottie, data_animation_random

if 'data_process' not in st.session_state:
    st.session_state.data_process = {
        'data_processed': False,
        'processed_df': None,
    }
if 'data_processing_performed' not in st.session_state:
    st.session_state.data_processing_performed = {
        'target_log': False,
        'numeric_scale': False,
        'fitted_scaler': None,
    }

def main():
    if 'homepage_param' in st.session_state and st.session_state.homepage_param['file_uploaded']:
        df_original = st.session_state.dataframe['pre-processed'].copy()
        st.markdown(f"<h1 style='text-align: center'>{st.session_state.dataframe['df_name'].title()}</h1>", unsafe_allow_html=True)
        st.write('---')
        with st.expander('Control panel', expanded=True):
            container = st.container()

        df_process = df_original.copy()
        with st.sidebar:
            change_page_buttons(key='top', pages=['2 Data understanding', '4 Model construction'])
            st.write(f"<h1 style='text-align: center'>Control panel</h1>", unsafe_allow_html=True)
            st.write('---')
            process_data_tuple = []
            
            outlier_processing = st.toggle(label="Process outliers")
            if outlier_processing:
                process_data_tuple = process_outliers(tuple=process_data_tuple, container=container)
            
            numeric_scale = st.toggle(label="Scale features", help='''
                                      Rescales each feature such that it has a standard deviation of 1 and a mean of 0 \n
                                      [Learn more >](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
                                      ''')  
            if numeric_scale:
                st.session_state.data_processing_performed['numeric_scale'] = True
                process_data_tuple.append(('Numeric features scaling', Numeric_scale(container=container)))
            else:
                st.session_state.data_processing_performed['numeric_scale'] = False

            if st.session_state.homepage_param['target_feature_type'] == 'Numeric':
                target_norm = st.toggle(label="Normalize (log) target feature", help='''
                                        Converts a skewed distribution to a normal / less-skewed distribution \n
                                        [Learn more >](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html)
                                        ''')  
                if target_norm:
                    st.session_state.data_processing_performed['target_log'] = True
                    process_data_tuple.append(('Target feature logarithm', Target_log(container=container)))
                else:
                    st.session_state.data_processing_performed['target_log'] = False

            if process_data_tuple:
                st.session_state.data_process['data_processed'] = True
                preprocess_pipe = Pipeline(process_data_tuple)
                df_process = preprocess_pipe.fit_transform(df_process)
            else:
                st.session_state.data_process['data_processed'] = False
            
            st.session_state.data_process['processed_df'] = df_process

            change_page_buttons(key='bottom', pages=['2 Data understanding', '4 Model construction'])

        st.write(f"<h2 style='text-align: center'>Dataframes</h2>", unsafe_allow_html=True)
        dataframe_exploration(
            tabs_names=['Processed dataframe', 'Pre-processed dataframe', 'Statistical info'], 
            dataframes_dict={
                'Processed dataframe': df_process,
                'Pre-processed dataframe': df_original
                })
    else:
        st.write(f"<h2 style='text-align: center'>Upload file at the Homepage</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            st_lottie(data_animation_random(key='no file found'), height=300, width=300, key='data processing lottie')

if __name__ == '__main__':
    main()