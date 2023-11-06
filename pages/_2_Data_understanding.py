import streamlit as st
import sys
# import plotly.express as px

sys.path.append("..")
from functions import change_page_buttons, choose_dataframe, features_to_analyze, data_manipulation, outliers_analysis_sidebar, dataframe_exploration, outliers_analysis_main_page, data_animation_random, plot_pairplot, correlation_heatmap, st_lottie, px

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

            scatter_3d_toggle = st.toggle(label='3D scatter plot')
            if scatter_3d_toggle:
                cols = st.columns(2)
                x_feature = cols[0].selectbox(label='x', options=df_analyze.columns)
                y_feature = cols[1].selectbox(label='y', options=df_analyze.columns)
                cols = st.columns(2)
                z_feature = cols[0].selectbox(label='z', options=df_analyze.columns)
                color_feature = cols[1].selectbox(label='color feature', options=df_analyze.columns)

            heatmap_toggle = st.toggle(label="Built correlation heatmap")
            outlier_analysis_toggle = st.toggle(label="Analyze the data outliers")
            if outlier_analysis_toggle:
                with st.form('data analysis sidebar'):
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
                hue = cols_pp[1].selectbox('Choose feature to highlight (hue)', hue_cat_list, index=None)
                plot_pairplot(df=df_analyze, hue=hue)

        if outlier_analysis_toggle and data_analysis_form_button:
            outliers_analysis_main_page(df=df_outliers, settings=outliers_analysis_settings, dataframe_name=dataframe_name)

        if scatter_3d_toggle:
            fig = px.scatter_3d(df_analyze, x=x_feature, y=y_feature, z=z_feature, color=color_feature)
            st.plotly_chart(fig, use_container_width=True)

        if heatmap_toggle:
            correlation_heatmap(df=df_analyze)

    else:
        st.write(f"<h2 style='text-align: center'>Upload file at the Homepage</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            st_lottie(data_animation_random(key='no file found'), height=300, width=300, key='data understanding lottie')

if __name__ == '__main__':
    main()