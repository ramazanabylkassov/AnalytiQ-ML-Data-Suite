import streamlit as st

st.set_page_config(layout="wide", page_title='Model construction', page_icon='💻')

import sys
sys.path.append("..")
from functions import *

def main():
    if 'homepage_param' in st.session_state and st.session_state.homepage_param['file_uploaded']:
        pre_processed_df = st.session_state.dataframe['pre-processed'].copy()
        target_feature = st.session_state.homepage_param['target_feature']
        st.markdown(f"<h1 style='text-align: center'>{st.session_state.dataframe['df_name'].title()}</h1>", unsafe_allow_html=True)
        st.write('---')
        st.markdown(f"<h1 style='text-align: center'>Model deployment</h1>", unsafe_allow_html=True)
        st.write('---')

        dataframe_exploration(tabs_names=['Pre-processed dataframe', 'Statistical info'], dataframes_dict={'Pre-processed dataframe': pre_processed_df})
        

        with st.sidebar:
            change_page_buttons(key='top', pages=['4 Model construction'])
            st.write(f"<h1 style='text-align: center'>Control panel</h1>", unsafe_allow_html=True)
            st.write('###')
            test_data_row = {}
            groups = []
            for i in range(0, len(pre_processed_df.drop(target_feature, axis=1).columns), 2):
                groups.append(pre_processed_df.drop(target_feature, axis=1).columns[i:i+2])
            
            for group in groups:
                cols = st.columns(2)
                for i, feature in enumerate(group):
                    with cols[i]:
                        test_data_row[feature] = st.number_input(f'{feature}')
            
            df_test = pd.DataFrame(test_data_row, index=[0])

        st.write(df_test)
        st.write(st.session_state.linear_regression)
        
        prediction = st.session_state.linear_regression_fit(df_test) if not st.session_state.target_log else np.exp(st.session_state.linear_regression.predict(df_test))
        st.write(prediction)
        with st.sidebar:
            change_page_buttons(key='bottom', pages=['4 Model construction'])
    else:
        st.write(f"<h2 style='text-align: center'>Upload file at the Homepage</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            st_lottie(data_animation_random(key='no file found'), height=300, width=300, key='Model deployment lottie')

if __name__ == '__main__':
    main()
