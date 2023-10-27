import streamlit as st
import sys

sys.path.append("..")
from functions import *

try:
    st.set_page_config(layout="wide", page_title='Data understanding', page_icon='💡')
except:
    st.rerun()

def main():
    if 'homepage_param' in st.session_state and st.session_state.homepage_param['file_uploaded']:
        st.session_state.regress_form_button = False
        df = st.session_state.dataframe['pre-processed'].copy()
        
        st.markdown(f"<h1 style='text-align: center'>{st.session_state.dataframe['df_name'].title()}</h1>", unsafe_allow_html=True)
        st.write('---')

### ---Sidebar---
        with st.sidebar:
            change_page_buttons(key='top', pages=['1 Homepage', '3 Data processing'])

            st.write(f"<h1 style='text-align: center'>Control panel</h1>", unsafe_allow_html=True)
            st.write('---')

            working_dataframe, dataframe_name = choose_dataframe()

            st.markdown(f"<h2 style='text-align: center'>Choose features to analyze</h2>", unsafe_allow_html=True)
            st.write('###')
            df_analyze = features_to_analyze(df=working_dataframe)
            st.write('---')
        
            st.markdown(f"<h2 style='text-align: center'>Data manipulation</h2>", unsafe_allow_html=True)
            st.write('###')
            tabs_names, dataframes = data_manipulation(df=df_analyze, working_dataframe_name=dataframe_name)
        
            st.markdown(f"<h2 style='text-align: center'>Data analysis</h2>", unsafe_allow_html=True)
            st.write('###')
            
            if len(dataframes) > 1:
                keys = list(dataframes.keys())
                if 'Groupped dataframe' in keys:
                    keys.remove('Groupped dataframe')
                df_analyze_radio = st.radio('Choose dataframe to analyze', keys, horizontal=True)
                df_analyze = dataframes[df_analyze_radio]
                st.write('---')

            cols_pp = st.columns(2)
            pairplot_toggle = cols_pp[0].toggle(label='Pairplot the features')
            container_pairplot_warning = st.container()

            panda_profile_toggle = st.toggle(label="Analyze with Pandas Profile")
            outlier_analysis_toggle = st.toggle(label="Analyze the data outliers")
            if panda_profile_toggle or outlier_analysis_toggle:
                with st.form('data analysis sidebar'):
                    if panda_profile_toggle:
                        panda_profile_settings = panda_profile_sidebar()
                    if outlier_analysis_toggle:
                        df_outliers, outliers_analysis_settings = outliers_analysis_sidebar(df_analyze=df_analyze)
                    data_analysis_form_button = st.form_submit_button('Visualize', use_container_width=True)
            st.write('###')
            
            change_page_buttons(key='bottom', pages=['1 Homepage', '3 Data processing'])
### ---Main page---
        dataframe_exploration(tabs_names=tabs_names, dataframes_dict=dataframes)

        if pairplot_toggle:
            if df_analyze.select_dtypes(include='number').empty:
                container_pairplot_warning.warning('The dataframe does not contain numeric features to pairplot', icon='⚠️')
            else:
                st.write(f"<h3 style='text-align: center'>Pairplot analysis</h3>", unsafe_allow_html=True)
                hue=None
                hue_cat_list = [item for item in df_analyze.columns if item in st.session_state.homepage_param['categorical_features']]
                if cols_pp[1].checkbox('Choose feature to highlight (hue)'):
                    hue = cols_pp[1].selectbox('Choose hue feature', hue_cat_list, label_visibility='collapsed')
                plot_pairplot(df=df_analyze, hue=hue)

        if (panda_profile_toggle or outlier_analysis_toggle) and data_analysis_form_button:
            if panda_profile_toggle:
                panda_profile_main_page(df=df_analyze, settings=panda_profile_settings)
            if outlier_analysis_toggle:
                outliers_analysis_main_page(df=df_outliers, settings=outliers_analysis_settings, dataframe_name=dataframe_name)

    else:
        st.write(f"<h2 style='text-align: center'>Upload file at the Homepage</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            st_lottie(data_animation_random(key='no file found'), height=300, width=300, key='data understanding lottie')

if __name__ == '__main__':
    main()