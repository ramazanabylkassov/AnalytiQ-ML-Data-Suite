import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from sklearn import tree
from IPython.display import Image
from sklearn.tree import export_graphviz
import requests
from streamlit_lottie import st_lottie
import random

# import sketch

def preprocess_dataframe(df, container):
    feature_list = df.columns.tolist()
    preprocess_pipeline = []
    
    ### ---Drop features---
    features_to_drop = None 
    drop_features_switch = st.toggle(label="Drop features")
    if drop_features_switch:
        features_to_drop = st.multiselect('Features to drop:', feature_list, label_visibility='collapsed')
        preprocess_pipeline.append(('dropper', FeatureDropper(container=container, drop_list=features_to_drop)))
        feature_list = [item for item in feature_list if item not in features_to_drop]

    ### ---Impute features (deal with NaN cells)---
    impute_dict = {
        'features_with_value_list': None,
        'missing_value_input': None,
        'missing_value': None,
        'mean': [],
        'median': [],
        'most_frequent': [],
        'constant_str': [],
        'constant_str_value': [],
        'constant_num': [],
        'constant_num_value': [],
        'delete_rows': [],
    }
    col1, col2 = st.columns(2)
    with col1:
        features_to_impute_switch = st.toggle(label="Impute features", help='''
                                              Transforms empty (or manually selected values) in the dataframe \n 
                                              [Learn more >](https://scikit-learn.org/stable/modules/impute.html)
                                              ''')
    if features_to_impute_switch:
        impute_manually = col2.checkbox('Enter values to impute manually')
        if impute_manually:
            impute_dict['missing_value_input'] = st.text_input('Type in imputing value', value='NaN')
            impute_dict['missing_value'] = impute_dict['missing_value_input']
        else:
            impute_dict['missing_value'] = np.nan 
        
        dataframe_with_value = df[feature_list].loc[:, df.isin([impute_dict['missing_value']]).any()]
        
        features_with_value_list = dataframe_with_value.columns.tolist()
        impute_dict['features_with_value_list'] = features_with_value_list
        if len(features_with_value_list) != 0:
            apply_to_radio = st.radio('Apply to', ['All features', 'Choose manually'], horizontal=True, label_visibility='collapsed')

            if apply_to_radio == 'All features':
                numeric_list = dataframe_with_value.select_dtypes(include='number').columns.tolist()
                numeric_list_methods = ['mean', 'median', 'most_frequent', 'constant', 'delete_rows']
                categorical_list = dataframe_with_value.select_dtypes(include='object').columns.tolist()
                categorical_list_methods = ['most_frequent', 'constant', 'delete_rows']
                impute_strategy = st.selectbox('Choose impute strategy for numeric features', categorical_list_methods if categorical_list else numeric_list_methods)
                
                if impute_strategy == 'constant':
                    if categorical_list:
                        col1, col2 = st.columns(2)
                        impute_dict['constant_str'] = categorical_list
                        impute_dict['constant_str_value'] = col1.text_input('Choose constant value (dtype "object")') if numeric_list else st.text_input('Choose constant value (dtype "object")')
                        if numeric_list:
                            impute_dict['constant_num'] = numeric_list
                            impute_dict['constant_num_value'] = col2.number_input('Choose constant value (dtype "number")')
                    else:
                        impute_dict['constant_num'] = numeric_list
                        impute_dict['constant_num_value'] = st.number_input('Choose constant value (num)')  
                else:
                    impute_dict[impute_strategy] = features_with_value_list
            
            else:
                if df[features_with_value_list].select_dtypes(include='number').columns.tolist():
                    col1, col2 = st.columns(2)
                    impute_mean = col1.multiselect('Mean', df[features_with_value_list].select_dtypes(include='number').columns)
                    impute_dict['mean'] = impute_mean
                    features_with_value_list = [item for item in features_with_value_list if item not in impute_mean]
                    impute_median = col2.multiselect('Median', df[features_with_value_list].select_dtypes(include='number').columns)
                    impute_dict['median'] = impute_median
                    features_with_value_list = [item for item in features_with_value_list if item not in impute_median]
                
                col1, col2 = st.columns(2)
                impute_most_frequent = col1.multiselect('Most frequent', features_with_value_list)
                impute_dict['most_frequent'] = impute_most_frequent
                features_with_value_list = [item for item in features_with_value_list if item not in impute_most_frequent]
                impute_dict['delete_rows'] = col2.multiselect('Delete rows', features_with_value_list)
                features_with_value_list = [item for item in features_with_value_list if item not in impute_dict['delete_rows']]
                
                col1, col2 = st.columns(2)
                impute_constant = col1.multiselect('Constant', features_with_value_list, help='''
                * Input number for selected numeric features
                * Input text for selected categorical features
                ''')
                if impute_constant:
                    numeric_list = df[impute_constant].select_dtypes(include='number').columns.tolist()
                    categorical_list = df[impute_constant].select_dtypes(include='object').columns.tolist()
                    if numeric_list:
                        impute_dict['constant_num'] = numeric_list
                        impute_dict['constant_num_value'] = col2.number_input('Choose constant value (num)')
                    if categorical_list:
                        impute_dict['constant_str'] = categorical_list
                        impute_dict['constant_str_value'] = col2.text_input('Choose constant value (str)')
            preprocess_pipeline.append(('imputer', FeatureImputer(container=container, impute_dict=impute_dict)))
        else:
            preprocess_pipeline.append(('imputer', FeatureImputer(container=container, impute_dict=impute_dict)))

    ### ---Ordinal Encoding (Each unique category value is assigned an integer value)---
    features_to_oe = None
    oe_switch = st.toggle(label="Ordinal Encoder", help='''
                          Transforms each categorical feature to one new feature of integers (0 to n_categories - 1) \n 
                          [Learn more >](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features)
                          ''')
    if oe_switch:
        if len(df[feature_list].select_dtypes(include=['object']).columns) == 0:
            features_to_oe = []
        else:
            features_to_oe = st.multiselect('Features to OHE', df[feature_list].select_dtypes(include='object').columns.tolist())
            feature_list = [item for item in feature_list if item not in features_to_oe]
        preprocess_pipeline.append(('OrdinalEncoder', FeatureOrdinalEncoder(container=container, ord_encode_list=features_to_oe)))

    ### ---One Hot Encoder (integer encoded variable is removed and one new binary variable is added for each unique integer value in the variable) ---
    features_to_ohe = None
    ohe_switch = st.toggle(label="One Hot Encoder", help='''
                           Encode categorical features as a one-hot numeric array \n
                           [Learn more >](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
                           ''')
    if ohe_switch:
        features_to_ohe = st.multiselect('Features to OHE', feature_list)
        preprocess_pipeline.append(('OneHotEncoder', FeatureOneHotEncoder(container=container, encode_list=features_to_ohe)))

    ### ---Drop duplicate rows---
    drop_duplicates_switch = st.toggle(label="Drop duplicate rows")
    if drop_duplicates_switch:
        preprocess_pipeline.append(('duplicates', FeatureDuplicate(container=container)))
    
    preprocess_pipe = Pipeline(preprocess_pipeline)
    df = preprocess_pipe.fit_transform(df) if preprocess_pipe else df
    df.reindex(list(range(0, df.shape[0])))

    st.session_state.dataframe['pre-processed'] = df

    return df

def change_page_buttons(key=None, pages=None):
    if len(pages) == 1:
        if st.button(f'{"Next" if pages==["2 Data understanding"] else "Return to"}: **{pages[0].split(" ", 1)[1]}**', use_container_width=True, type='primary', key=key):
            st.session_state.homepage_param['form_button_remain_feature_type'] = False
            st.session_state.homepage_param['re_upload_button'] = False
            switch_page(pages[0])
    else:
        col1, col2 = st.columns(2)
        if col1.button(f'Return to: **{pages[0].split(" ", 1)[1]}**', use_container_width=True, type='primary', key=f'{key} 1'):
            st.session_state.homepage_param['form_button_remain_feature_type'] = False
            switch_page(pages[0])
        if col2.button(f'Next: **{pages[1].split(" ", 1)[1]}**', use_container_width=True, type='primary', key=f'{key} 2'):
            st.session_state.homepage_param['form_button_remain_feature_type'] = False
            switch_page(pages[1])

def reupload_file_button():
    cols = st.columns((3,2,1,1))        
    re_upload_button = cols[0].button('Upload another dataframe')
    if re_upload_button or st.session_state.homepage_param['re_upload_button']:
        st.session_state.homepage_param['re_upload_button'] = False
        cols[1].write(f"<h3 style='text-align: center'>Are you sure?</h3>", unsafe_allow_html=True)
        st.session_state.homepage_param['re_upload_button'] = True
        if cols[2].button('Yes'):
            st.session_state.homepage_param['file_uploaded'] = False
            st.rerun()
        if cols[3].button('No'):
            st.session_state.homepage_param['re_upload_button'] = False
            st.rerun()

class FeatureDuplicate(BaseEstimator, TransformerMixin):
    def __init__(self, container=None):
        self.container = container

    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        container = self.container
        df_no_dupl = df.drop_duplicates(ignore_index=True)
        rows_dropped = df.shape[0] - df_no_dupl.shape[0]
        if rows_dropped == 0:
            container.warning('The dataframe has no duplicate rows', icon='⚠️')
            return df
        else:
            container.success(f'''
                              **Duplicate rows dropped:** {rows_dropped}''', icon='✅')
            return df_no_dupl

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, container=None, drop_list=None):
        self.drop_list = drop_list
        self.container = container

    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        drop_list = self.drop_list
        container = self.container
        if drop_list:
            container.success(f'**Dropped {"features" if len(drop_list) != 1 else "feature"}:** {", ".join(drop_list)}', icon='✅')
            return df.drop(drop_list, axis=1)
        else:
            return df
    
class FeatureImputer(BaseEstimator, TransformerMixin):
    def __init__(self, container=None, impute_dict=None):
        self.impute_dict = impute_dict
        self.container = container

    def fit(self, df, y=None):
        return self
    
    def transform(self, dataframe):
        container = self.container
        impute_dict = self.impute_dict

        if len(impute_dict['features_with_value_list']) == 0:
            if impute_dict['missing_value_input'] == None:
                container.warning(f'The dataframe has no empty cells', icon='⚠️')
            else:
                container.warning(f'The dataframe has no cells containing {impute_dict["missing_value"]}', icon='⚠️')
            return dataframe
        else:
            if impute_dict['mean']:
                imputer = SimpleImputer(missing_values=impute_dict['missing_value'], strategy='mean')
                dataframe[impute_dict['mean']] = imputer.fit_transform(dataframe[impute_dict['mean']])
            if impute_dict['median']:
                imputer = SimpleImputer(missing_values=impute_dict['missing_value'], strategy='median')
                dataframe[impute_dict['median']] = imputer.fit_transform(dataframe[impute_dict['median']])
            if impute_dict['most_frequent']:
                imputer = SimpleImputer(missing_values=impute_dict['missing_value'], strategy='most_frequent')
                dataframe[impute_dict['most_frequent']] = imputer.fit_transform(dataframe[impute_dict['most_frequent']])
            if impute_dict['delete_rows']:
                if impute_dict['missing_value_input']:
                    for feature in impute_dict['delete_rows']:
                        dataframe.drop(dataframe[dataframe[feature] == impute_dict['missing_value']].index, inplace=True)
                    if not dataframe.loc[:, dataframe.isin([impute_dict['missing_value']]).any()].columns.tolist() and (impute_dict['constant_str'] or impute_dict['constant_num']):
                        container.warning(f'Selected features ({", ".join(impute_dict["constant_str"]) if impute_dict["constant_str"] else ", ".join(impute_dict["constant_num"])}) has no selected impute value', icon='⚠️')
                        impute_dict['constant_str'] = []
                        impute_dict['constant_num'] = []
                else:
                    dataframe = dataframe.dropna(subset=impute_dict['delete_rows'])
            if impute_dict['constant_str']:
                imputer = SimpleImputer(missing_values=impute_dict['missing_value'], strategy='constant', fill_value=impute_dict['constant_str_value'])
                dataframe[impute_dict['constant_str']] = imputer.fit_transform(dataframe[impute_dict['constant_str']])
            if impute_dict['constant_num']:
                imputer = SimpleImputer(missing_values=impute_dict['missing_value'], strategy='constant', fill_value=impute_dict['constant_num_value'])
                dataframe[impute_dict['constant_num']] = imputer.fit_transform(dataframe[impute_dict['constant_num']])
            if impute_dict['mean'] or impute_dict['median'] or impute_dict['most_frequent'] or impute_dict['constant_str'] or impute_dict['constant_num'] or impute_dict['delete_rows']:
                container.success(f'''
                                    **Feature Imputer:** ({"Selected" if impute_dict['missing_value_input'] else "Empty"} values replaced) \n
                                    {f"using mean: {', '.join(impute_dict['mean'])}" if impute_dict['mean'] else ''}    
                                    {f"using median: {', '.join(impute_dict['median'])}" if impute_dict['median'] else ''}  
                                    {f"using most frequent value: {', '.join(impute_dict['most_frequent'])}" if impute_dict['most_frequent'] else ''}  
                                    {f"by deleting rows: {', '.join(impute_dict['delete_rows'])}" if impute_dict['delete_rows'] else ''}  
                                    {f"using constant value: {', '.join(impute_dict['constant_num']+impute_dict['constant_str'])}" if (impute_dict['constant_num']+impute_dict['constant_str']) else ''} 
                                    ''', icon='✅')

            return dataframe
    
class FeatureOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, container=None, ord_encode_list=None):
        self.ord_encode_list = ord_encode_list
        self.container = container

    def fit(self, df, y=None):
        return self

    def transform(self, dataframe=None):
        ord_encode_list = self.ord_encode_list
        container = self.container        
        dataframe_oe = dataframe
        categories = []
        if len(dataframe.select_dtypes(include=['object']).columns) == 0:
            container.warning('The dataframe has no features for Ordinal Encoding', icon='⚠️')
            return dataframe
        for feature in ord_encode_list:
            enc = OrdinalEncoder()
            dataframe_oe[[feature]] = enc.fit_transform(dataframe[[feature]])
            buffer = []
            for i, key in enumerate(enc.categories_[0]):
                buffer.append(f'{key} ({i})')
            buffer2 = '; '.join(buffer)
            categories.append(f'**{feature}** - {buffer2}  ')
        categories_string = ' || '.join(categories)

        if ord_encode_list:
            container.success(f'''
            **Ordinal Encoder** \n
            {categories_string}
            ''', icon='✅')
        return dataframe_oe

class FeatureOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, container=None, encode_list=None):
        self.encode_list = encode_list
        self.container = container

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        encode_list = self.encode_list
        container = self.container
        if encode_list:
            for feature in encode_list:
                buffer_column = pd.get_dummies(df[feature], drop_first=True, prefix=feature, dtype=int)
                df = pd.concat([df.drop(feature, axis=1), buffer_column], axis=1)
            container.success(f'''
                              **One Hot Encoder** \n
                              OHE applied to: {", ".join(encode_list)}''', icon='✅')

        return df
    
def feature_type(dataframe=None, container=None):
    dataframe_type = st.sidebar.radio('Choose the type of dataframe', ['with supervisor (target feature)', 'withOUT supervisor'], horizontal=True)
    st.session_state.homepage_param['dataframe_type'] = dataframe_type
    if dataframe_type == 'with supervisor (target feature)':
        feature_types = ['Numeric', 'Categorical']
        st.subheader('Target feature')
        col1, col2 = st.columns((2, 1))
        target_feature = col1.selectbox('Target feature:', dataframe.columns, label_visibility='collapsed')
        target_type = col2.radio('Target feature type', feature_types, label_visibility='collapsed')
        st.session_state.homepage_param['target_feature'] = target_feature
        st.session_state.homepage_param['target_feature_type'] = target_type
    else:
        st.session_state.homepage_param['target_feature'] = None
        st.session_state.homepage_param['target_feature_type'] = None

    feature_types = ['Numeric', 'Categorical']
    target_feature = st.session_state.homepage_param['target_feature']
    st.subheader('Remaining features' if dataframe_type == 'with supervisor (target feature)' else 'Features')
    choose_feature = st.radio('Choose features: ', feature_types, horizontal=True, help='Not chosen features will be allocated to the opposite type')
    with st.form(key='remaining_feature_type'):
        col1, col2 = st.columns((3,1))
        if choose_feature == 'Numeric':
            default_features = [item for item in (dataframe.drop(target_feature, axis=1).columns if dataframe_type == 'with supervisor (target feature)' else dataframe.columns) if dataframe[item].nunique() > 10]
        else:
            default_features = [item for item in (dataframe.drop(target_feature, axis=1).columns if dataframe_type == 'with supervisor (target feature)' else dataframe.columns) if dataframe[item].nunique() < 10]
        chosen_features = col1.multiselect(f'Choose {choose_feature.lower()} features', dataframe.drop(target_feature, axis=1).columns if dataframe_type == 'with supervisor (target feature)' else dataframe.columns, label_visibility='collapsed', default=default_features)
        form_button_remaining_feature_type = col2.form_submit_button('Submit')
    if form_button_remaining_feature_type or st.session_state.homepage_param['form_button_remain_feature_type']:
        st.session_state.homepage_param['form_button_remain_feature_type'] = True
        if choose_feature == 'Numeric':
            numeric_features = chosen_features
            categorical_features = dataframe.drop(target_feature, axis=1).drop(chosen_features, axis=1).columns.tolist() if dataframe_type == 'with supervisor (target feature)' else dataframe.drop(chosen_features, axis=1).columns.tolist()
            st.session_state.homepage_param['numeric_features'] = numeric_features
            st.session_state.homepage_param['categorical_features'] = categorical_features
        else:
            categorical_features = chosen_features
            numeric_features = dataframe.drop(target_feature, axis=1).drop(chosen_features, axis=1).columns.tolist() if dataframe_type == 'with supervisor (target feature)' else dataframe.drop(chosen_features, axis=1).columns.tolist()
            st.session_state.homepage_param['numeric_features'] = numeric_features
            st.session_state.homepage_param['categorical_features'] = categorical_features
        
        numeric_features_string = ', '.join(st.session_state.homepage_param['numeric_features'])
        categorical_features_string = ', '.join(st.session_state.homepage_param['categorical_features'])
        if dataframe_type == 'with supervisor (target feature)':
            container.success(f'''
            **Feature type distribution:** \n 
            **Target feature:** {st.session_state.homepage_param['target_feature']}  
            **Target feature type:** {st.session_state.homepage_param['target_feature_type']}  
            **Numeric features:** {numeric_features_string if numeric_features_string else 'No features'}  
            **Categorical features:** {categorical_features_string if categorical_features_string else 'No features'}
            ''', icon='✅')
        else:
            container.success(f'''
            Feature type distribution: \n
            **Numeric features:** {numeric_features_string if numeric_features_string else 'No features'} \n
            **Categorical features:** {categorical_features_string if categorical_features_string else 'No features'}
            ''', icon='✅')

        st.session_state.homepage_param['remaining_features_choice'] = choose_feature
        st.session_state.homepage_param['chosen_features_type'] = chosen_features

def upload_data():
    df = None
    st.session_state.homepage_param['re_upload_button'] = False
    st.sidebar.markdown('<h1 style="text-align: center;">Upload your file</h1>', unsafe_allow_html=True)
    file = st.sidebar.file_uploader(label='Upload your file', label_visibility='collapsed')
    if file:
        if file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file, sep=None, engine='python')
        if '.' in file.name:
            st.session_state.dataframe['df_name'] = ' '.join(file.name.split('.')[0].split('_'))
        else:
            st.session_state.dataframe['df_name'] = file.name
        st.session_state.dataframe['original'] = df
        st.session_state.homepage_param['file_uploaded'] = True
        st.session_state.homepage_param['balloons'] = True
        st.rerun()

@st.cache_data
def statistical_data(df):
    output = {
        'Feature': [],
        'Number of non NaN values': [],
        'Data type': [],
        'Unique values': [],
        'Number of NaN cells': [],
        '% of NaN cells': [],
        'Mean': [],
        'Mode': [],
        'Median': [],
        'Min': [],
        'Max': [],
        'Standard Deviation': [],
        'Variance': [],
        'Quantile 50%': [],
        'Quantile 75%': [],
        'Quantile 99%': [],
        "Skewness": [],
        "Kurtosis": [],
    }

    for column in df.columns:
        output['Feature'].append(column.upper())
        output['Number of non NaN values'].append(df[column].notna().sum())
        output['Data type'].append(df[column].dtype)
        output['Unique values'].append(df[column].nunique())
        output['Number of NaN cells'].append(df[column].isna().sum())
        output['% of NaN cells'].append(round(df[column].isna().sum()/len(df.index)*100, 2))
        if df[column].dtype in ['int', 'float', 'int64']:
            output['Mean'].append(round(df[column].mean(), 2))
            output['Mode'].append(df[column].mode().values.tolist()[0])
            output['Median'].append(round(df[column].median(), 2))
            if column:
                output['Min'].append(round(df[column].min(), 2))
                output['Max'].append(round(df[column].max(), 2))
            output['Standard Deviation'].append(round(df[column].std(), 2))
            output['Variance'].append(round(df[column].var(), 2))
            output['Quantile 50%'].append(round(df[column].quantile(q=0.5), 2))
            output['Quantile 75%'].append(round(df[column].quantile(q=0.75), 2))
            output['Quantile 99%'].append(round(df[column].quantile(q=0.99), 2))
            output["Skewness"].append(skew(df[column]))
            output["Kurtosis"].append(kurtosis(df[column]))
        else:
            output['Mean'].append(None)
            output['Mode'].append(None)
            output['Median'].append(None)
            if column:
                output['Min'].append(None)
                output['Max'].append(None)
            output['Standard Deviation'].append(None)
            output['Variance'].append(None)
            output['Quantile 50%'].append(None)
            output['Quantile 75%'].append(None)
            output['Quantile 99%'].append(None)
            output["Skewness"].append(None)
            output["Kurtosis"].append(None)

    return pd.DataFrame(output).set_index("Feature")

def download_custom(df_to_download=None, file_name=None):
    @st.cache_data 
    def convert_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    df_converted = convert_to_csv(df_to_download)

    st.download_button(
        f"Download **{file_name}**",
        df_converted,
        f"{file_name}.csv",
        f"text/csv",
        key=f'{file_name}'
        )

def ask_AI(df=None):
    question = st.text_input('Ask KattanAI about your dataframe')
    if st.button('Ask my question'):
        # import asyncio
        # from contextlib import contextmanager
        # @contextmanager
        # def setup_event_loop():
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     try:
        #         yield loop
        #     finally:
        #         loop.close()
        #         asyncio.set_event_loop(None)
        # with setup_event_loop() as loop:
            

        answer = df.sketch.ask(question)
        st.markdown(answer, unsafe_allow_html=True)

def features_to_analyze(df=None):
    features_to_analize_choices = ['All features', 'choose features']
    features_to_analize_radio = st.radio('Choose data:', features_to_analize_choices, horizontal=True, label_visibility='collapsed')
    pp_features_list = 'All features'
    if features_to_analize_radio == 'choose features':
        pp_features_list= st.multiselect('Fatures to analyze', df.columns, placeholder='choose features', label_visibility='collapsed')
        if len(pp_features_list) > 1:
            return df[pp_features_list]
        else:
            st.error('Choose at least **two** features', icon='🛑')
            st.stop()
    else:
        return df

def data_manipulation(df=None, working_dataframe_name=None):
    tabs_names = [working_dataframe_name]
    dataframes = {}

    dataframes[working_dataframe_name] = df

    data_filter = st.toggle(label="Filter data", key="data_filterer")
    if data_filter:
        filtered_df = dataframe_explorer(df, case=False)
        filtered_df.index = range(len(filtered_df))
        dataframes['Filtered dataframe'] = filtered_df
        tabs_names.append('Filtered dataframe')
        st.write('---')

    data_groupper = st.toggle(label="Group data", key="data_groupperer")
    if data_groupper:
        if any(item in df.columns.tolist() for item in st.session_state.homepage_param['categorical_features']):
            if data_filter:
                df_to_use = st.radio('Use dataframe', ['Pre-processed dataframe', 'Filtered dataframe'], horizontal=True)
                if df_to_use == 'Pre-processed dataframe':
                    groupped_df = group_by(df)
                    dataframes['Groupped dataframe'] = groupped_df
                else:
                    groupped_df = group_by(filtered_df)
                    dataframes['Groupped dataframe'] = groupped_df
            else:
                groupped_df = group_by(df)
                dataframes['Groupped dataframe'] = groupped_df
            tabs_names.append('Groupped dataframe')
        else:
            st.warning('No categorical features in the dataframe to group by', icon='⚠️')

    st.write('---')

    tabs_names.append('Statistical info')
    tabs_names.append('Simple Graphs')

    return tabs_names, dataframes

def dataframe_exploration(tabs_names=None, dataframes_dict=None):
    dataframes = []
    for df_name in dataframes_dict:
        dataframes.append(dataframes_dict[f'{df_name}'])

    tabs_names_copy = tabs_names.copy()
    tabs_names_copy.remove('Statistical info')
    if 'Simple Graphs' in tabs_names_copy:
        tabs_names_copy.remove('Simple Graphs')

    tabs = st.tabs(tabs_names)
    for i in range(len(tabs)):
        with tabs[i]:
            if tabs_names[i] == 'Statistical info':
                if len(tabs_names_copy) > 1:
                    df_choice_stat = st.radio('Choose dataframe', tabs_names_copy, horizontal=True, label_visibility='collapsed', key='df_choice_stat')
                    if not dataframes_dict[f'{df_choice_stat}'].empty:
                        st.dataframe(statistical_data(dataframes_dict[f'{df_choice_stat}']), use_container_width=True)
                else:
                    st.dataframe(statistical_data(dataframes_dict[f'{tabs_names[0]}']), use_container_width=True)
            elif tabs_names[i] == 'Simple Graphs':
                if len(tabs_names_copy) > 1:
                    df_choice_graphs = st.radio('Choose dataframe', tabs_names_copy, horizontal=True, label_visibility='collapsed', key='df_choice_graph')
                    simple_graph(dataframes_dict[f'{df_choice_graphs}'])
                else:
                    simple_graph(dataframes_dict[f'{tabs_names[0]}'])
            else:
                st.dataframe(dataframes_dict[f'{tabs_names_copy[i]}'], use_container_width=True)
                download_custom(df_to_download=dataframes_dict[f'{tabs_names_copy[i]}'], file_name=tabs_names_copy[i])

def simple_graph(df):
    target_feature = st.session_state.homepage_param['target_feature']
    target_feature_type = st.session_state.homepage_param['target_feature_type']
    cols = st.columns((2,1,2))
    simple_graph_feature_1 = cols[0].selectbox('Choose the feature', df.columns, label_visibility='collapsed', key='simple_graph_feature_1')
    cols[1] = cols[1].empty()
    cols[1].markdown("<h5 style='text-align: center '> VS </h5>", unsafe_allow_html=True)
    simple_graph_feature_2 = cols[2].selectbox('Choose the feature', df.columns, label_visibility='collapsed', key='simple_graph_feature_2')
    
    tabs = st.tabs([f'{simple_graph_feature_1}', f'{simple_graph_feature_2}', f'Relation'])
    with tabs[0]:
        if simple_graph_feature_1 in st.session_state.homepage_param['categorical_features'] or (simple_graph_feature_1==target_feature and target_feature_type=='Categorical'):
            fig = plt.figure()
            plt.pie(df[simple_graph_feature_1].value_counts(), labels = df[simple_graph_feature_1].unique(), autopct='%.0f%%')
            st.pyplot(fig)
        else:
            fig = plt.figure()
            sns.histplot(data=df[simple_graph_feature_1], palette='pastel')
            st.pyplot(fig)
    with tabs[1]:
        if simple_graph_feature_2 in st.session_state.homepage_param['categorical_features']:
            fig = plt.figure()
            plt.pie(df[simple_graph_feature_2].value_counts(), labels = df[simple_graph_feature_2].unique(), autopct='%.0f%%')
            st.pyplot(fig)
        else:
            fig = plt.figure()
            sns.histplot(data=df[simple_graph_feature_2], palette='pastel')
            st.pyplot(fig)
    with tabs[2]:
        if st.session_state.homepage_param['numeric_features']:
            feature_types = ['numeric' if (simple_graph_feature_1 in st.session_state.homepage_param['numeric_features'] or (simple_graph_feature_1 == target_feature and target_feature_type == 'Numeric')) else 'categorical', 'numeric' if simple_graph_feature_2 in st.session_state.homepage_param['numeric_features'] or (simple_graph_feature_2 == target_feature and target_feature_type == 'Numeric')  else 'categorical']
        else:
            feature_types = ['categorical', 'categorical']
        if feature_types.count('numeric') == 0:
            fig = plt.figure()
            sns.histplot(x=simple_graph_feature_1, hue=simple_graph_feature_2, data=df, stat="count", multiple="stack")
            plt.xticks(rotation=90, ha='right')
            st.pyplot(fig)
        elif feature_types.count('numeric') == 1:
            categorical = simple_graph_feature_2 if simple_graph_feature_2 in st.session_state.homepage_param['categorical_features'] or (simple_graph_feature_2 == target_feature and target_feature_type == 'Categorical') else simple_graph_feature_1
            numerical = simple_graph_feature_1 if categorical == simple_graph_feature_2 else simple_graph_feature_2
            fig = plt.figure()
            sns.violinplot(x=df[categorical], y=df[numerical])
            st.pyplot(fig)
        else:
            fig = plt.figure()
            sns.regplot(x=df[simple_graph_feature_2], y=df[simple_graph_feature_1], color='green')
            st.pyplot(fig)

class Visualization:
    def __init__(self, dataframe=None, settings=None):
        self.dataframe = dataframe
        self.CI_level = settings['CI_level']
        self.grid_number = settings['grid_number']
        self.SD_level = settings['SD_level']
        self.outl_stat = settings['outlier_stat_toggle']
        self.outl_df = settings['outlier_df_toggle']
        self.boxplot_present_toggle = settings['IQR_presentation_toggle']
     
    def boxplot(self):
        numeric_df = self.dataframe.select_dtypes(include='number')
        outl_stat = self.outl_stat
        outl_df = self.outl_df
        m = self.grid_number
        boxplot_present_toggle = self.boxplot_present_toggle

        st.markdown(f"<h3 style='text-align: center'>InterQuartile Range method</h3>", unsafe_allow_html=True)
        st.write('###')

        tabs_names = ['Graphs']
        if outl_df:
            tabs_names.append('Outliers dataframe')
        if outl_stat:
            tabs_names.append('Outlier statistics')

        if len(tabs_names) > 1:
            tabs = st.tabs(tabs_names)
            if outl_df:
                upper_limits, lower_limits = identify_outlier_borders(df=numeric_df, method='IQR')
                with tabs[1]:
                    outlier_dataframe(df=numeric_df, upper_limits=upper_limits, lower_limits=lower_limits, method='IQR', level=None)
                if outl_stat:
                    with tabs[2]:
                        outlier_statistics(numeric_df, upper_limits, lower_limits)
            else:
                upper_limits, lower_limits = identify_outlier_borders(df=numeric_df, method='IQR')
                with tabs[1]:
                    outlier_statistics(numeric_df, upper_limits, lower_limits)

        if numeric_df.shape[1] > 0:
            if boxplot_present_toggle:
                groups = []
                for i in range(0, len(numeric_df.columns), m):
                    groups.append(numeric_df.columns[i:i+m])
                if outl_stat:
                    cols = tabs[0].columns(m)
                else:
                    cols = st.columns(m)
                for group in groups:
                    for i, feature in enumerate(group):
                        with cols[i]:       
                            st.markdown(f"<div style='text-align: center'>{feature}</div>", unsafe_allow_html=True)
                            st.write('###')
                            fig = plt.figure()
                            sns.boxplot(data=numeric_df[feature])
                            st.pyplot(fig)
            else:
                if outl_stat:
                    with tabs[0]:
                        fig = plt.figure()
                        ax = plt.gca()
                        sns.boxplot(data=numeric_df)
                        ax.set_xticklabels(numeric_df.columns, rotation=40,ha='right',va='top')
                        st.pyplot(fig)
                else:
                    fig = plt.figure()
                    ax = plt.gca()
                    sns.boxplot(data=numeric_df)
                    ax.set_xticklabels(numeric_df.columns, rotation=40,ha='right',va='top')
                    st.pyplot(fig)
    
    def standard_deviation(self):
        dataframe = self.dataframe
        choice = self.SD_level
        m = self.grid_number
        outl_stat = self.outl_stat
        outl_df = self.outl_df

        pre_processed_df = st.session_state.dataframe['pre-processed'][dataframe.columns].select_dtypes(include='number')
        upper_limits, lower_limits = identify_outlier_borders(df=pre_processed_df, method='SD', level=choice)

        st.markdown(f"<h3 style='text-align: center'>Standard Deviation method ({choice})</h3>", unsafe_allow_html=True)
        st.write('###')
        
        tabs_names = ['Graphs']
        if outl_df:
            tabs_names.append('Outliers dataframe')
        if outl_stat:
            tabs_names.append('Outlier statistics')

        if len(tabs_names) > 1:
            tabs = st.tabs(tabs_names)
            if outl_df:
                with tabs[1]:
                    outlier_dataframe(df=dataframe, upper_limits=upper_limits, lower_limits=lower_limits, method='SD', level=choice)
                if outl_stat:
                    with tabs[2]:
                        outlier_statistics(dataframe, upper_limits, lower_limits)
            else:
                with tabs[1]:
                        outlier_statistics(dataframe, upper_limits, lower_limits)
        
        groups = []
        for i in range(0, len(dataframe.columns), m):
            groups.append(dataframe.columns[i:i+m])

        for group in groups:
            if outl_stat:
                cols = tabs[0].columns(m)
            else:
                cols = st.columns(m)
            for i, feature in enumerate(group):
                with cols[i]:
                    fig = plt.figure()
                    st.markdown(f"<div style='text-align: center'>{feature}</div>", unsafe_allow_html=True)
                    plt.plot(dataframe[feature], 'bo', alpha=0.5)
                    plt.fill_between(range(len(dataframe[feature])), lower_limits[feature], upper_limits[feature], color="green", alpha=0.3, label=f"{choice} SDs")
                    plt.xlabel('Sample')
                    plt.ylabel(feature)
                    plt.legend()
                    st.pyplot(fig)
    
    def confidence_interval(self):
        dataframe = self.dataframe
        choice = self.CI_level
        m = self.grid_number
        outl_stat = self.outl_stat
        outl_df = self.outl_df

        pre_processed_df = st.session_state.dataframe['pre-processed'][dataframe.columns].select_dtypes(include='number')
        upper_limits, lower_limits = identify_outlier_borders(df=pre_processed_df, method='CI', level=choice)

        confidence_intervals_list = {}
        groups = []

        confidence = choice / 100

        for column in pre_processed_df.columns:
            sample_1 = pre_processed_df[column]
            mean, sigma = np.mean(sample_1), np.std(sample_1)
            confidence_intervals_list[column] = stats.norm.interval(confidence, loc=mean, scale=sigma)

        for i in range(0, len(dataframe.columns), m):
            groups.append(dataframe.columns[i:i+m])
        
        st.markdown(f"<h3 style='text-align: center'>Confidence Interval method ({choice}%)</h3>", unsafe_allow_html=True)
        st.write('###')

        tabs_names = ['Graphs']
        if outl_df:
            tabs_names.append('Outliers dataframe')
        if outl_stat:
            tabs_names.append('Outlier statistics')

        if len(tabs_names) > 1:
            tabs = st.tabs(tabs_names)
            if outl_df:
                with tabs[1]:
                    outlier_dataframe(df=dataframe, upper_limits=upper_limits, lower_limits=lower_limits, method='CI', level=choice)
                if outl_stat:
                    with tabs[2]:
                        outlier_statistics(dataframe, upper_limits, lower_limits)
            else:
                with tabs[1]:
                        outlier_statistics(dataframe, upper_limits, lower_limits)

        for group in groups:
            if outl_stat or outl_df:
                cols = tabs[0].columns(m)
            else:
                cols = st.columns(m)
            for i, feature in enumerate(group):
                with cols[i]:
                    fig = plt.figure()
                    st.markdown(f"<div style='text-align: center'>{feature}</div>", unsafe_allow_html=True)
                    plt.plot(dataframe[feature], 'bo', alpha=0.5)
                    plt.fill_between(range(len(dataframe[feature])), lower_limits[feature], upper_limits[feature], color='red', alpha=0.3,
                                    label=f'{choice} Confidence Interval')
                    plt.xlabel('Sample')
                    plt.ylabel(feature)
                    plt.legend()
                    st.pyplot(fig)
        
def outlier_statistics(dataframe_os, upper_limits, lower_limits):
    upper_outliers_summary = {
        "Feature": [],
        "Feature corridor [min, max]": [],
        "Amount of grtr outliers": [],
        "Grtr outlier Max": [],
        "Grtr outlier Min": [],
        "Grtr outlier Mean": [],
        "Grtr outlier Median": [],
        "Grtr outlier Mode": [],
    }

    lower_outliers_summary = {
        "Feature": [],
        "Feature corridor [min, max]": [],
        "Amount of lwr outliers": [],
        "Lwr outlier Max": [],
        "Lwr outlier Min": [],
        "Lwr outlier Mean": [],
        "Lwr outlier Median": [],
        "Lwr outlier Mode": [],
    }

    for column in dataframe_os.columns:
        upper_outliers_list = []
        lower_outliers_list = []
        for i in dataframe_os[column].index:
            if dataframe_os[column][i] > upper_limits[column]:
                upper_outliers_list.append(dataframe_os[column][i])
            elif dataframe_os[column][i] < lower_limits[column]:
                lower_outliers_list.append(dataframe_os[column][i])

        if len(upper_outliers_list) > 0:
            upper_outliers_summary["Feature"].append(column.upper())
            upper_outliers_summary["Amount of grtr outliers"].append((dataframe_os[column] > upper_limits[column]).sum())
            upper_outliers_summary["Feature corridor [min, max]"].append(
                [round(lower_limits[column], 2), round(upper_limits[column], 2)])
            upper_outliers_summary["Grtr outlier Max"].append(round(max(upper_outliers_list), 2))
            upper_outliers_summary["Grtr outlier Min"].append(round(min(upper_outliers_list), 2))
            upper_outliers_summary["Grtr outlier Mean"].append(round(np.average(upper_outliers_list), 2))
            upper_outliers_summary["Grtr outlier Median"].append(round(statistics.median(upper_outliers_list), 2))
            upper_outliers_summary["Grtr outlier Mode"].append(round(statistics.mode(upper_outliers_list), 2))

        if len(lower_outliers_list) > 0:
            lower_outliers_summary["Feature"].append(column.upper())
            lower_outliers_summary["Amount of lwr outliers"].append((dataframe_os[column] < lower_limits[column]).sum())
            lower_outliers_summary["Feature corridor [min, max]"].append(
                [round(lower_limits[column], 2), round(upper_limits[column], 2)])
            lower_outliers_summary["Lwr outlier Max"].append(round(max(lower_outliers_list), 2))
            lower_outliers_summary["Lwr outlier Min"].append(round(min(lower_outliers_list), 2))
            lower_outliers_summary["Lwr outlier Mean"].append(round(np.average(lower_outliers_list), 2))
            lower_outliers_summary["Lwr outlier Median"].append(round(statistics.median(lower_outliers_list), 2))
            lower_outliers_summary["Lwr outlier Mode"].append(round(statistics.mode(lower_outliers_list), 2))

    if pd.DataFrame(upper_outliers_summary).empty:
        st.warning('No greater outliers', icon='⚠️')
    else:
        st.write("Greater outliers summary:")
        st.dataframe(pd.DataFrame(upper_outliers_summary).set_index("Feature"), use_container_width=True)
    if pd.DataFrame(lower_outliers_summary).empty:
        st.warning('No lower outliers', icon='⚠️')
    else:
        st.write("Lower outliers summary:")
        st.dataframe(pd.DataFrame(lower_outliers_summary).set_index("Feature"), use_container_width=True)

def group_by(df):
    col1, col2 = st.sidebar.columns((3,1))
    group_by_column = col1.selectbox('Group by: ', [item for item in st.session_state.homepage_param['categorical_features'] if item in df.columns.tolist()])
    methods = ['mean', 'median']
    group_by_method = col2.radio('Choose the method: ', methods, label_visibility='hidden', horizontal=False)
    if group_by_method == 'mean':
        result_df = round(df.groupby(group_by_column).mean(), 2)
    elif group_by_method == 'median':
        result_df = df.groupby(group_by_column).median()
    return result_df

def panda_profile_sidebar():
    st.markdown(f"<h3 style='text-align: center'>Analyze your data with Pandas Profiling</h3>", unsafe_allow_html=True)
    st.write('###')

    panda_profile_settings = {
        'title': 'DataFrame Profiling',
    }
    col1, col2 = st.columns(2)
    samples = col1.checkbox('Samples')
    correlations = col2.checkbox('Correlations')
    col1, col2 = st.columns(2)
    missing_diagrams = col1.checkbox('Missing diagrams')
    duplicates = col2.checkbox('Duplicates')
    col1, _ = st.columns(2)
    interactions = col1.checkbox('Interactions')
    if not samples:
        panda_profile_settings['samples'] = None
    if not correlations:
        panda_profile_settings['correlations'] = None
    if not missing_diagrams:
        panda_profile_settings['missing_diagrams'] = None
    if not duplicates:
        panda_profile_settings['duplicates'] = None
    if not interactions:
        panda_profile_settings['interactions'] = None    

    st.write('---')

    return panda_profile_settings

def panda_profile_main_page(df=None, settings=None):
    with st.expander('Panda Profile Report', expanded=True):
        profile = ProfileReport(df, **settings)
        st_profile_report(profile)

def outliers_analysis_sidebar(df_analyze=None):
    df_outliers = df_analyze.select_dtypes(include='number')
    
    st.markdown(f"<h3 style='text-align: center'>Analyze your outliers</h3>", unsafe_allow_html=True)
    st.write('###')

    if df_outliers.shape[1] == 0:
        st.warning('No numeric features to be analyzed for outliers, encode the features first', icon='⚠️')
        exit()
    else:
        if df_outliers.shape[1] != df_analyze.shape[1]:
            st.warning(f'''
            Next features will not be analyzed for outliers: **{", ".join(list(set(df_analyze.columns) - set(df_outliers.columns)))}** \n
            Hint: Encode at Homepage first
            ''', icon='⚠️')
    
    outliers_analysis_settings = {
        'outlier_stat_toggle': None,
        'outlier_df_toggle': None,
        'CI_checkbox': None,
        'CI_level': None,
        'SD_checkbox': None,
        'SD_level': None,
        'IQR_checkbox': None,
        'grid_number': None,
        'IQR_presentation_toggle': None,
    }

    st.write('###')
    cols = st.columns(2)
    st.write('###')
    outlier_stat_toggle = cols[0].toggle(label="Outlier statistics", value=True)
    outliers_analysis_settings['outlier_stat_toggle'] = outlier_stat_toggle
    outlier_df_toggle = cols[1].toggle(label="Outlier dataframe", help=f'''
                                        Shows dataframe with highlighted cells containing outliers (:red[{'red = greater outliers'}], :blue[{'blue = lower outliers'}])
                                        ''')
    outliers_analysis_settings['outlier_df_toggle'] = outlier_df_toggle
    col1, col2 = st.columns(2)
    with col1:    
        CI_checkbox = st.checkbox('Using CI', help='''
                                  ***Confidence Interval (CI) Method***:
                                  * **Approach**: The confidence interval method uses a statistical confidence interval to identify outliers. It typically assumes that the data follows a normal distribution.
                                  * **Calculation**: It calculates a confidence interval around the mean of the data and considers data points outside this interval as potential outliers.
                                  * **Sensitivity** to Data Distribution: This method is sensitive to the distribution of data. If your data deviates from normality, it may not work effectively.
                                  * **Use of Z-Score**: It often involves the use of Z-scores to determine whether a data point falls outside the confidence interval.
                                  * **Applicability**: It is suitable for data that reasonably follows a normal distribution.
                                  ''')
        outliers_analysis_settings['CI_checkbox'] = CI_checkbox
    with col2:
        CI_level = st.number_input("Type in confidence level (%): ", value=99)
        outliers_analysis_settings['CI_level'] = CI_level
    col1, col2 = st.columns(2)
    with col1: 
        SD_checkbox = st.checkbox('Using SD', help='''
                                  ***Standard Deviation (SD) Method***:
                                  * **Approach**: The standard deviation method identifies outliers based on how many standard deviations a data point is away from the mean.
                                  * **Calculation**: It calculates the mean and standard deviation of the data and considers data points that are more than a specified number of standard deviations away from the mean as outliers. A common threshold is ±3 standard deviations.
                                  * **Sensitivity** to Data Distribution: Like the confidence interval method, it assumes normal distribution and may not work well with non-normally distributed data.
                                  * **Applicability**: It can be applied to data that is approximately normally distributed, but it can be sensitive to extreme values.
                                  ''')
        outliers_analysis_settings['SD_checkbox'] = SD_checkbox
    with col2:
        SD_list = ["1x", "2x", "3x"]
        SD_level = st.radio("Choose SD corridor: ", SD_list, horizontal=True, index=2)
        outliers_analysis_settings['SD_level'] = SD_level
    col1, col2 = st.columns(2)
    with col1:
        IQR_checkbox = st.checkbox('Using InterQuartile Range', help='''
                                   ***Interquartile Range (IQR) Method***:
                                   * **Approach**: The IQR method identifies outliers based on the spread of the data as represented by the interquartile range.
                                   * **Calculation**: It calculates the IQR, which is the range between the first quartile (Q1) and the third quartile (Q3) of the data. Data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers.
                                   * **Sensitivity** to Data Distribution: The IQR method is less sensitive to the shape of the data distribution compared to the previous methods and can be used with skewed or non-normal data.
                                   * **Applicability**: It is particularly useful for data that may not follow a normal distribution.
                                   ''')
        outliers_analysis_settings['IQR_checkbox'] = IQR_checkbox
        grid_number = st.number_input(label='Choose grid width', min_value=1, max_value=7 if df_outliers.shape[1] >= 7 else df_outliers.shape[1], value=4 if df_outliers.shape[1] >= 4 else df_outliers.shape[1])
        outliers_analysis_settings['grid_number'] = grid_number
    with col2:
        IQR_presentation_toggle = st.toggle(label="Show each feature separately")
        outliers_analysis_settings['IQR_presentation_toggle'] = IQR_presentation_toggle

    return df_outliers, outliers_analysis_settings

@st.cache_data
def outliers_analysis_main_page(df=None, settings=None, dataframe_name=None):
    if settings['CI_checkbox'] or settings['SD_checkbox'] or settings['IQR_checkbox']:
        st.write(f"<h3 style='text-align: center'>Outlier analysis in the {dataframe_name.lower()}</h3>", unsafe_allow_html=True)
        visualization = Visualization(dataframe=df, settings=settings)
        if settings['CI_checkbox']:
            visualization.confidence_interval()
        if settings['SD_checkbox']:
            visualization.standard_deviation()
        if settings['IQR_checkbox']:
            visualization.boxplot()

def process_outliers(tuple=None, container=None):
    numeric_features = st.session_state.homepage_param['numeric_features'].copy()
    top_code_list = numeric_features
    
    cols = st.sidebar.columns(2)
    n = 0
    if st.session_state.homepage_param['target_feature']:
        include_target_feature = cols[n].checkbox('Include target feature outliers')
        if include_target_feature:
            numeric_features.append(st.session_state.homepage_param['target_feature'])
            top_code_list = numeric_features
        n += 1
    if st.session_state.homepage_param['categorical_features']:
        include_cat_var = cols[n].checkbox('Include categorical features outliers')
        if include_cat_var:
            cat_features_outliers = st.radio('Choose categorical features', ['All categorical features', 'Choose manually'], horizontal=True, label_visibility='collapsed')
            if cat_features_outliers == 'Choose manually':
                top_code_cat = st.sidebar.multiselect('Choose categorical features', st.session_state.homepage_param['categorical_features'])
                top_code_list += top_code_cat
            else:
                top_code_list += st.session_state.homepage_param['categorical_features']
    else:
        top_code_list = numeric_features
    
    st.write('---')
    
    outlier_method_list = ['IQR', 'SD', 'CI']
    outlier_method = st.radio('Select method to identify the outliers', outlier_method_list, horizontal=True)
    if outlier_method == 'SD':
        outlier_level = st.radio('Choose the Standard Deviation corridor', ['1x', '2x', '3x'], horizontal=True, index=2)
    elif outlier_method == 'CI':
        outlier_level = st.number_input('Input the Confidence Interval corridor (%)', value=99)
    else:
        outlier_level = '1.5*IQR'
    
    st.write('---')

    top_code_method_list = ["Equalize to corridor borders", "Equalize to mean", "Equalize to median", "Equalize to mode", "Remove an entire row"]
    top_code_method = st.radio('Select method to topcode the outliers', top_code_method_list)
    tuple.append(('Outlier processing', Top_code_outliers(container=container, feature_list=top_code_list, outlier_method=outlier_method, outlier_level=outlier_level, top_code_method=top_code_method)))                

    return tuple

class Target_log(BaseEstimator, TransformerMixin):
    def __init__(self, container=None):
        self.container = container
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df_log):
        container = self.container
        target_feature = st.session_state.homepage_param['target_feature']
        df_log[target_feature] = np.log(df_log[target_feature])
        container.success(f'Target feature ({target_feature}) is logarithmized', icon='✅')
        return df_log

class Numeric_scale(BaseEstimator, TransformerMixin):
    def __init__(self, container=None):
        self.container = container
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df_scale):
        container = self.container
        numeric_features = st.session_state.homepage_param['numeric_features']
        scaler = StandardScaler()
        scaler.fit(df_scale[numeric_features])
        df_scale[numeric_features] = scaler.transform(df_scale[numeric_features])
        container.success(f'Numeric features ({", ".join(numeric_features)}) are scaled', icon='✅')
        return df_scale

class Top_code_outliers(BaseEstimator, TransformerMixin):
    def __init__(self, container=None, feature_list=None, outlier_method=None, outlier_level=None, top_code_method=None):
        self.container = container
        self.feature_list = feature_list
        self.outlier_method = outlier_method
        self.outlier_level = outlier_level
        self.top_code_method = top_code_method
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df_outliers):        
        container = self.container
        top_code_list = self.feature_list
        top_code_method = self.top_code_method
        outlier_method = self.outlier_method
        outlier_level = self.outlier_level

        top_outlier_border, bottom_outlier_border = identify_outlier_borders(df=df_outliers[top_code_list], method=outlier_method, level=outlier_level)

        feature_mean = {}
        feature_median = {}
        feature_mode = {}

        for feature in top_code_list:
            feature_mean[f'{feature}'] = round(df_outliers[feature].mean(), 2)
            feature_median[f'{feature}'] = df_outliers[feature].median()
            feature_mode[f'{feature}'] = df_outliers[feature].mode()[0]

        for feature in top_code_list: 
            for i in df_outliers[feature].index:
                current_value = df_outliers.loc[i, feature]                
                if current_value > top_outlier_border[f'{feature}'] or current_value < bottom_outlier_border[f'{feature}']:
                    if top_code_method == "Equalize to median":
                        df_outliers.loc[i, feature] = feature_median[f'{feature}']
                    elif top_code_method == "Equalize to mean":
                        df_outliers.loc[i, feature] = feature_mean[f'{feature}']
                    elif top_code_method == "Equalize to mode":
                        df_outliers.loc[i, feature] = feature_mode[f'{feature}']
                    elif top_code_method == "Remove an entire row":
                        df_outliers = df_outliers.drop(i)
                    elif top_code_method == "Equalize to corridor borders":
                        if current_value > top_outlier_border[f'{feature}']:
                            df_outliers.loc[i, feature] = round(top_outlier_border[f'{feature}'], 2)
                        elif current_value < bottom_outlier_border[f'{feature}']:
                            df_outliers.loc[i, feature] = round(bottom_outlier_border[f'{feature}'], 2)
                    
        if top_code_method == "Remove an entire row":
            df_outliers.index = range(len(df_outliers))

        container.success(f'Outliers ({outlier_method}) are topcoded using method - "{top_code_method}"', icon='✅')
        return df_outliers

def identify_outlier_borders(df=None, method=None, level=None):
    top_outlier_border = {}
    bottom_outlier_border = {}

    for feature in df.columns: 
        if method == 'SD':
            if level == '3x':
                mult = 3
            elif level == '2x':
                mult = 2
            else:
                mult = 1
            bottom_outlier_border[f'{feature}'] = df[feature].mean() - mult * df[feature].std()
            top_outlier_border[f'{feature}'] = df[feature].mean() + mult * df[feature].std()
        elif method == 'CI':
            sample_1 = df[feature]
            mean, sigma = np.mean(sample_1), np.std(sample_1)
            bottom_outlier_border[f'{feature}'], top_outlier_border[f'{feature}'] = stats.norm.interval(level/100, loc=mean, scale=sigma)
        else:
            Q1, Q3 = np.percentile(df[feature], [25, 75], method='midpoint')
            IQR = Q3 - Q1
            top_outlier_border[f'{feature}'] = Q3 +1.5*IQR
            bottom_outlier_border[f'{feature}'] = Q1 - 1.5*IQR
    
    return top_outlier_border, bottom_outlier_border

def outlier_dataframe(df=None, upper_limits=None, lower_limits=None, method=None, level=None):
    def apply_style(value=None, upper_limit=None, lower_limit=None):
        if value>upper_limit:
            return 'background-color : #FF7F7F'
        elif value<lower_limit:
            return 'background-color : #ADD8E6'
        else:
            return ''
    styled_df = df.style.apply(lambda column: [apply_style(value, upper_limits[column.name], lower_limits[column.name]) for value in df[column.name]], axis=0).format(precision=1)
    st.dataframe(styled_df, use_container_width=True)
    # if st.button("Download Styled DataFrame as Excel", key=f'{method}'):
    #     styled_df.to_excel(f'Outlier dataframe {method} {f": ({level})" if level else ""}.xlsx', engine='openpyxl', index=False)

class MachineLearning:
    def __init__(self, train_test=None):
        self.X_train = train_test['X_train']
        self.X_test = train_test['X_test']
        self.y_train = train_test['y_train']
        self.y_test = train_test['y_test']

    def KNN_classification(self, X_train, X_test, y_train, y_test):
        dataframe = self.dataframe
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.inspection import permutation_importance

        error_rate = {}

        for i in range(1,50):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train,y_train)
            pred_i = knn.predict(X_test)
            error_rate[i]=(np.mean(pred_i != y_test))

        model_knn = KNeighborsClassifier(n_neighbors=min(error_rate, key=error_rate.get))
        model_knn.fit(X_train,y_train)

        pred_knn = model_knn.predict(X_test)
        knn_result = permutation_importance(knn, X_train, y_train, n_repeats=10, random_state=101)
        # knn_importances = knn_result.importances_mean

        st.write(f'Confusion matrix:')
        st.write(confusion_matrix(y_test, pred_knn))
        st.write(f'Classification report:')
        st.write(classification_report(y_test, pred_knn))

        # predictions['KNeighborsClassifier'] = pred_knn
        # models['KNeighborsClassifier'] = model_knn

    def SVC_classification(self, X_train, X_test, y_train, y_test):
        dataframe = self.dataframe
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
        grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
        grid.fit(X_train,y_train)
        grid.best_estimator_

        pred_SVC = grid.predict(X_test)
        # grid_importances = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=101).importances_mean

        st.write(f'Confusion matrix:')
        st.write(confusion_matrix(y_test, pred_SVC))
        st.write(f'Classification report:')
        st.write(classification_report(y_test, pred_SVC))

    def logistic_regression(self, X_train, X_test, y_train, y_test):
        dataframe = self.dataframe
        model_log = LogisticRegression(random_state=101)
        model_log.fit(X_train,y_train)

        pred_log = model_log.predict(X_test)
        # logreg_importances = abs(model_log.coef_[0])

        st.write(f'Confusion matrix:')
        st.write(confusion_matrix(y_test, pred_log))
        st.write(f'Classification report:')
        st.write(classification_report(y_test, pred_log))

    def decision_tree(self, X_train, X_test, y_train, y_test):
        dataframe = self.dataframe
        dtree_model = DecisionTreeClassifier(max_depth=5)
        dtree_model.fit(X_train, y_train)

        pred_dtree = dtree_model.predict(X_test)
        # dtree_importances = dtree_model.feature_importances_

        st.write(f'Confusion matrix:')
        st.write(confusion_matrix(y_test, pred_dtree))
        st.write(f'Classification report:')
        st.write(classification_report(y_test, pred_dtree))

        features = list(dataframe.columns[1:])
        dot_data = tree.export_graphviz(
                    dtree_model,
                    out_file=None, 
                    feature_names=features,
                    filled=True, 
                    rounded=True
            )

        graph = pydotplus.graph_from_dot_data(dot_data)

        Image(graph.create_png())

    def random_forest(self, X_train, X_test, y_train, y_test):
        randforest_model = RandomForestClassifier(n_estimators=100)
        randforest_model.fit(X_train, y_train)

        pred_randtree = randforest_model.predict(X_test)
        randtree_importances = randforest_model.feature_importances_

        st.write(f'Confusion matrix:')
        st.write(confusion_matrix(y_test, pred_randtree))
        st.write(f'Classification report:')
        st.write(classification_report(y_test, pred_randtree))

def regression_metrics(regression_name=None, alpha=None, y_test=None, y_pred=None):
    if st.session_state.target_log == False:
        output = {
            'Regression model': regression_name,
            'Hyper-parameter': alpha,
            'MAE': round(metrics.mean_absolute_error(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4),
            'R2': round( metrics.r2_score(y_test, y_pred), 4),
            'MAPE': round(mean_absolute_percentage_error(y_test, y_pred), 4),
        }
    else:
        output = {
            'Regression model': regression_name,
            'Hyper-parameter': alpha,
            'MAE': round(metrics.mean_absolute_error(np.exp(y_test), np.exp(y_pred)), 4),
            'RMSE': round(np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(y_pred))), 4),
            'R2': round( metrics.r2_score(y_test, y_pred), 4),
            'MAPE': round(mean_absolute_percentage_error(y_test, y_pred), 4),
        }
    return output

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def choose_dataframe():
    dataframe_name = 'Pre-processed dataframe'
    if 'data_process' in st.session_state and st.session_state.data_process['data_processed']:
        dataframe_name = st.radio('Choose dataframe', ['Pre-processed dataframe', 'Processed dataframe'], help='''
        ***Dataframes***:
        * **Pre-processed** = dataframe pre-processed at the Home Page
        * **Processed** = dataframe processed at the Data processing Page (the corridor is used from the pre-processed dataframe)
        ''', horizontal=True)
        if dataframe_name == 'Pre-processed dataframe':
            working_dataframe = st.session_state.dataframe['pre-processed']
        else:
            working_dataframe = st.session_state.data_process['processed_df']
    else:
        working_dataframe = st.session_state.dataframe['pre-processed']
    
    return working_dataframe, dataframe_name

def train_test(df=None, target_feature=None):
    def num_input_train_size():
        st.session_state.train_size = 100-st.session_state.test_size
    def num_input_test_size():
        st.session_state.test_size = 100-st.session_state.train_size
    cols = st.columns(2)
    cols[0].write('Train size (%)')
    cols[0].number_input('Train size (%)', max_value=99, min_value=1, key='test_size', value=70, on_change=num_input_train_size, label_visibility='collapsed')
    cols[1].write('Test size (%)')
    test_size = cols[1].number_input('Test size (%)', max_value=99, min_value=1, key='train_size', value=30, on_change=num_input_test_size, label_visibility='collapsed')
    return train_test_split(df.drop(target_feature,axis=1), df[target_feature], test_size=test_size, random_state=32)

@st.cache_resource
def linear_regression(train_test=None, cv_number=None, feature_importance=None):
    X_train = train_test['X_train']
    X_test = train_test['X_test']
    y_train = train_test['y_train']
    y_test = train_test['y_test']

    regression_name='Linear regression'
    regression_ouput = {
        'regression_metric': None,
        'cv_analysis': None,
        'FI_analysis': None, 
    }

    model_regression = LinearRegression()
    model_regression.fit(X_train, y_train)
    y_pred_linear = model_regression.predict(X_test)

    lin_regr_metric = regression_metrics(regression_name=regression_name, y_test=y_test, y_pred=y_pred_linear)
    regression_ouput['regression_metric'] = lin_regr_metric

    if cv_number:
        cv_analysis_lin = ML_analysis_cross_validation(model_name=regression_name, model=model_regression, X_train=X_train, y_train=y_train, cv_number=cv_number)
        regression_ouput['cv_analysis'] = cv_analysis_lin
    if feature_importance:
        FI_analysis_lin = ML_analysis_feature_importance(model_name=regression_name, model=model_regression, feature_list=X_train.columns)
        regression_ouput['FI_analysis'] = FI_analysis_lin

    st.session_state.linear_regression_fit = model_regression
    return regression_ouput, model_regression

@st.cache_resource
def polynomial_regression(train_test=None, degree=None, cv_number=None, feature_importance=None, search_cv_type=None, n_iter_number=None):
    X_train = train_test['X_train']
    X_test = train_test['X_test']
    y_train = train_test['y_train']
    y_test = train_test['y_test']

    regression_name='Polynomial regression'
    regression_ouput = {
        'regression_metric': None,
        'cv_analysis': None,
        'FI_analysis': None, 
    }

    if search_cv_type == 'GridSearchCV':
        steps = [
            ('poly', PolynomialFeatures()),
            ('regressor', LinearRegression())
        ]
        pipeline = Pipeline(steps)
        param_grid = {
            'poly__degree': [1, 2, 3, 4, 5],
        }

        grid_cv_polynom = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_cv_polynom.fit(X_train, y_train)
        degree_buffer = grid_cv_polynom.best_params_['poly__degree']
        best_estimator = grid_cv_polynom.best_estimator_
        y_pred_polynom = best_estimator.predict(X_test)
        model_for_analysis = best_estimator.named_steps['regressor']
        
        pipeline_poly = best_estimator.named_steps['poly']
        X_train_poly = pipeline_poly.transform(X_train)
        feature_names = pipeline_poly.get_feature_names_out(input_features=X_train.columns)
    elif search_cv_type == 'RandomizedSearchCV':
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('regressor', LinearRegression())
        ])
        param_dist = {
            'poly__degree': randint(1, 6)
        }
        
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_number, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)
        
        degree_buffer = random_search.best_params_['poly__degree']
        best_estimator = random_search.best_estimator_
        y_pred_polynom = best_estimator.predict(X_test)
        model_for_analysis = best_estimator.named_steps['regressor']
        
        pipeline_poly = best_estimator.named_steps['poly']
        X_train_poly = pipeline_poly.transform(X_train)
        feature_names = pipeline_poly.get_feature_names_out(input_features=X_train.columns)
    else:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        feature_names = poly.get_feature_names_out(input_features=X_train.columns)
        model_polynom = LinearRegression()
        model_polynom.fit(X_train_poly, y_train)
        y_pred_polynom = model_polynom.predict(X_test_poly)
        degree_buffer = degree
        model_for_analysis = model_polynom

    
    pol_regr_metric = regression_metrics(regression_name=regression_name, alpha=f'degree: {degree_buffer}', y_test=y_test, y_pred=y_pred_polynom)
    regression_ouput['regression_metric'] = pol_regr_metric

    if cv_number:
        cv_analysis_pol = ML_analysis_cross_validation(model_name=regression_name, model=model_for_analysis, X_train=X_train_poly, y_train=y_train, cv_number=cv_number)
        regression_ouput['cv_analysis'] = cv_analysis_pol
    if feature_importance:
        FI_analysis_pol = ML_analysis_feature_importance(model_name=regression_name, model=model_for_analysis, feature_list=feature_names)
        regression_ouput['FI_analysis'] = FI_analysis_pol

    return regression_ouput, model_for_analysis

@st.cache_resource
def lasso_regression(train_test=None, alpha=None, cv_number=None, feature_importance=None, search_cv_type=None, n_iter_number=None):
    X_train = train_test['X_train']
    X_test = train_test['X_test']
    y_train = train_test['y_train']
    y_test = train_test['y_test']

    regression_name='Lasso regression'
    regression_ouput = {
        'regression_metric': None,
        'cv_analysis': None,
        'FI_analysis': None, 
    }

    if search_cv_type == 'GridSearchCV':
        param_grid = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 5, 10]}     
        model_lasso = Lasso() 
        grid_cv_lasso = GridSearchCV(model_lasso, param_grid, cv = 5)
        grid_cv_lasso.fit(X_train, y_train)
        y_pred_lasso = grid_cv_lasso.predict(X_test)
        alpha_buffer = grid_cv_lasso.best_params_["alpha"]
        model_for_analysis = grid_cv_lasso.best_estimator_
    elif search_cv_type == 'RandomizedSearchCV':
        model = Lasso()
        param_dist = {
            'alpha': uniform(loc=0.0, scale=1.0),
        }
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_number, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)

        y_pred_lasso = random_search.predict(X_test)
        alpha_buffer = random_search.best_params_["alpha"]
        model_for_analysis = random_search.best_estimator_
    else:
        model_lasso = Lasso(alpha=alpha) 
        model_lasso.fit(X_train, y_train)
        y_pred_lasso = model_lasso.predict(X_test)
        alpha_buffer = alpha
        model_for_analysis = model_lasso
    
    lass_regr_metric = regression_metrics(regression_name=regression_name, alpha=f'alpha: {alpha_buffer}', y_test=y_test, y_pred=y_pred_lasso)
    regression_ouput['regression_metric'] = lass_regr_metric

    if cv_number:
        cv_analysis_lasso = ML_analysis_cross_validation(model_name=regression_name, model=model_for_analysis, X_train=X_train, y_train=y_train, cv_number=cv_number)
        regression_ouput['cv_analysis'] = cv_analysis_lasso

    if feature_importance:
        FI_analysis_lasso = ML_analysis_feature_importance(model_name=regression_name, model=model_for_analysis, feature_list=X_train.columns)
        regression_ouput['FI_analysis'] = FI_analysis_lasso

    return regression_ouput, model_for_analysis

@st.cache_resource
def ridge_regression(train_test=None, alpha=None, cv_number=None, feature_importance=None, search_cv_type=None, n_iter_number=None):
    X_train = train_test['X_train']
    X_test = train_test['X_test']
    y_train = train_test['y_train']
    y_test = train_test['y_test']

    regression_name='Ridge regression'
    regression_ouput = {
        'regression_metric': None,
        'cv_analysis': None,
        'FI_analysis': None, 
    }

    if search_cv_type == 'GridSearchCV':
        param_grid = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 5, 10]}
        model_ridge = Ridge() 
        grid_cv_ridge = GridSearchCV(model_ridge, param_grid, cv = 5)
        grid_cv_ridge.fit(X_train, y_train)
        y_pred_ridge = grid_cv_ridge.predict(X_test)
        alpha_buffer = grid_cv_ridge.best_params_["alpha"]
        model_for_analysis = grid_cv_ridge.best_estimator_
    elif search_cv_type == 'RandomizedSearchCV':
        model = Ridge()
        param_dist = {
            'alpha': uniform(loc=0.0, scale=1.0),
        }
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_number, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)

        y_pred_ridge = random_search.predict(X_test)
        alpha_buffer = random_search.best_params_["alpha"]
        model_for_analysis = random_search.best_estimator_
    else:
        model_ridge = Ridge(alpha=alpha)
        model_ridge.fit(X_train, y_train)
        y_pred_ridge = model_ridge.predict(X_test)
        alpha_buffer = alpha
        model_for_analysis = model_ridge
    
    ridge_regr_metric = regression_metrics(regression_name=regression_name, alpha=f'alpha: {alpha_buffer}', y_test=y_test, y_pred=y_pred_ridge)
    regression_ouput['regression_metric'] = ridge_regr_metric

    if cv_number:
        cv_analysis_ridge = ML_analysis_cross_validation(model_name=regression_name, model=model_for_analysis, X_train=X_train, y_train=y_train, cv_number=cv_number)
        regression_ouput['cv_analysis'] = cv_analysis_ridge

    if feature_importance:
        FI_analysis_ridge = ML_analysis_feature_importance(model_name=regression_name, model=model_for_analysis, feature_list=X_train.columns)
        regression_ouput['FI_analysis'] = FI_analysis_ridge

    return regression_ouput, model_for_analysis

def ML_analysis_feature_importance(model_name=None, model=None, feature_list=None):     
    FI_output = {
        'dataframe': None,
        'graph': None,
        }
    
    featureImportance = pd.DataFrame({"feature": feature_list, f"{model_name}": model.coef_})
    featureImportance.sort_values([f"{model_name}"], ascending=False, inplace=True)
    FI_output['dataframe'] = featureImportance.reset_index(drop=True)
    
    fig, ax = plt.subplots()
    featureImportance[f"{model_name}"].plot(kind='bar', figsize=(10, 6))
    plt.xticks(ticks=featureImportance.index, labels=featureImportance['feature'], rotation=45)
    ax.set_xticklabels(featureImportance['feature'], rotation=44, ha='right')
    FI_output['graph'] = fig
    
    return FI_output

def ML_analysis_cross_validation(model_name=None, model=None, X_train=None, y_train=None, cv_number=None):
    if st.session_state.target_log:
        scores = -cross_val_score(model, X_train, np.exp(y_train), scoring='neg_mean_absolute_error',  cv=cv_number)
    else:
        scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error',  cv=cv_number)
    
    CV_output = {}
    CV_output['Regression model'] = model_name
    CV_output['Mean MAE'] = round(scores.mean(), 4)
    for i, score in enumerate(scores):
        CV_output[f'MAE of CV sample #{i+1}'] = round(score, 4)
    
    return CV_output

def ML_regression_models(train_test_data=None, container_metrics=None, container_cv_analysis=None, container_FI_analysis=None):
    st.markdown(f"<h2 style='text-align: center'>Regression ML models</h2>", unsafe_allow_html=True)
    st.write('###')
    apply_gridsearch_toggle = st.toggle('Apply GridSearchCV', value=True, help=f'''
                                        Systematic approach to find the best combination of hyperparameters for a model by performing an exhaustive search over a predefined hyperparameter grid. \n
                                        [Learn more >](https://scikit-learn.org/stable/modules/grid_search.html#grid-search)
                                        ''')
    n_iter_number = None
    if apply_gridsearch_toggle:
        search_cv_type = st.radio('Choose type of SearchCV', ['GridSearchCV', 'RandomizedSearchCV'], horizontal=True)
        if search_cv_type == 'RandomizedSearchCV':
            n_iter_number = st.number_input('Input the number of iterations', value=3, min_value=1)
    analyze_models_toggle = st.toggle(label="Model analysis", value=False)
    feature_importance = False
    CV_samples = None
    CV_checkbox = False
    if analyze_models_toggle:
        feature_importance = st.checkbox(label='Feature Importance (FI) analysis', help=f'''
                                         **Feature Importance:** \n
                                         **Definition:** Feature importance is a technique used to assess the contribution of each feature (input variable) in a machine learning model's predictions. It quantifies the relevance or significance of each feature in making accurate predictions.  
                                         **Purpose:** Feature importance helps identify which features have the most impact on the model's performance, allowing you to focus on the most influential variables and potentially simplify the model by removing less important ones. \n
                                         [Learn more >](https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html#randomforest-feature-importances)                                         
                                         ''')
        cols = st.columns((2))
        CV_checkbox = cols[0].checkbox(label='Cross-Validation (CV) analysis', help=f'''
                                       **Cross-validation:** \n
                                       **Definition:** Cross-validation is a statistical technique in machine learning and data analysis where a dataset is partitioned into subsets for the purpose of evaluating the performance of a predictive model.  
                                       **Purpose:** The primary purpose of cross-validation is twofold:  
                                       *Model Performance Evaluation:* Cross-validation provides a more robust and accurate assessment of a model's performance compared to a single train-test split.  
                                       *Overfitting Detection:* Cross-validation helps in identifying overfitting, a common problem in machine learning where a model performs exceptionally well on the training data but poorly on new data. \n
                                       [Learn more >](https://scikit-learn.org/0.17/modules/cross_validation.html)
                                       ''')
        if CV_checkbox:
            CV_samples = cols[1].number_input('Amount of CV samples', value=5, step=1)

    st.write('###')
    with st.form('Regression models'):
        cols = st.columns(2)
        linear_regress_toggle = cols[0].toggle(label="Linear regression", help=f'''
                                                ***Linear Regression:*** \n
                                                **Objective:** Linear regression aims to find a linear relationship between the dependent variable and the independent variables. It assumes that the relationship is a straight line.  
                                                **Regularization:** Linear regression doesn't include any regularization terms, making it prone to overfitting if you have a lot of features.  
                                                **Use Case:** It's used when there's a clear linear relationship between the variables, and there's no concern about overfitting. \n
                                                [Learn more >](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
                                                ''')
        
        cols = st.columns(2)
        polynomial_regress_toggle = cols[0].toggle(label='Polynomial regression', help=f'''
                                                    ***Polynomial Regression:*** \n
                                                    **Objective:** Polynomial regression is a type of regression that models the relationship between variables as an nth-degree polynomial. It allows for capturing non-linear relationships between the dependent and independent variables.  
                                                    **Regularization:** Polynomial regression can be combined with L1 (Lasso) or L2 (Ridge) regularization if needed.  
                                                    **Use Case:** Polynomial regression is used when the relationship between variables is not linear but can be approximated by a polynomial. It's especially useful when dealing with data that exhibits curves and bends. \n
                                                    [Learn more >](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features)
                                                    ''')
        pol_regr_degree = None
        if not apply_gridsearch_toggle:
            pol_regr_degree = cols[1].number_input(label='Input the degree:', value=2, max_value=5)
        cols = st.columns(2)
        lasso_regress_toggle = cols[0].toggle(label="Lasso regression", help=f'''
                                                ***Lasso Regression (L1 Regularization):*** \n
                                                **Objective:** Lasso regression extends linear regression by adding a regularization term (L1 norm) to the loss function. This regularization encourages sparsity, meaning it can set some of the coefficients to exactly zero, effectively performing feature selection.  
                                                **Use Case:** Lasso is useful when you have many features, and you want to perform feature selection, keeping only the most important ones. It can also help with handling multicollinearity. \n
                                                [Learn more >](https://scikit-learn.org/stable/modules/linear_model.html#lasso)                                                              
                                                ''')
        lasso_regr_alpha = None
        if not apply_gridsearch_toggle:
            lasso_regr_alpha = cols[1].number_input(label='Input the alpha:', key='lasso_regr', value=1.0, format='%.3f')
        cols = st.columns(2)
        ridge_regress_toggle = cols[0].toggle(label="Ridge regression", help=f'''
                                                ***Ridge Regression (L2 Regularization):*** \n
                                                **Objective:** Ridge regression also extends linear regression but uses a different regularization term (L2 norm). This regularization doesn't make coefficients exactly zero but shrinks them towards zero, reducing the impact of less important features.  
                                                **Use Case:** Ridge regression is helpful when you have many features and suspect multicollinearity. It can also prevent overfitting by reducing the impact of less relevant features. \n
                                                [Learn more >](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
                                                ''')
        ridge_regr_alpha = None
        if not apply_gridsearch_toggle:
            ridge_regr_alpha = cols[1].number_input(label='Input the alpha:', key='ridge_regr', value=1.0, format='%.3f')
        regress_form_button = st.form_submit_button(label='Construct models')
    st.write('###')

    if regress_form_button:
        regression_metrics = []
        regression_cv_analysis = []
        regression_FI_analysis = []

        if linear_regress_toggle:
            lin_regr_output, _ = linear_regression(train_test=train_test_data, cv_number=CV_samples, feature_importance=feature_importance)
            regression_metrics.append(lin_regr_output['regression_metric'])
            if lin_regr_output['cv_analysis'] != None:
                regression_cv_analysis.append(lin_regr_output['cv_analysis'])
            if lin_regr_output['FI_analysis'] != None:
                regression_FI_analysis.append(lin_regr_output['FI_analysis'])
        if polynomial_regress_toggle:
            pol_regr_output, _ = polynomial_regression(train_test=train_test_data, degree=pol_regr_degree, cv_number=CV_samples, feature_importance=feature_importance, search_cv_type=search_cv_type, n_iter_number=n_iter_number)
            regression_metrics.append(pol_regr_output['regression_metric'])
            if pol_regr_output['cv_analysis'] != None:
                regression_cv_analysis.append(pol_regr_output['cv_analysis'])
            if pol_regr_output['FI_analysis'] != None:
                regression_FI_analysis.append(pol_regr_output['FI_analysis'])
        if lasso_regress_toggle:
            lass_regr_output, _ = lasso_regression(train_test=train_test_data, alpha=lasso_regr_alpha, cv_number=CV_samples, feature_importance=feature_importance, search_cv_type=search_cv_type, n_iter_number=n_iter_number)
            regression_metrics.append(lass_regr_output['regression_metric'])
            if lass_regr_output['cv_analysis'] != None:
                regression_cv_analysis.append(lass_regr_output['cv_analysis'])
            if lass_regr_output['FI_analysis'] != None:
                regression_FI_analysis.append(lass_regr_output['FI_analysis'])
        if ridge_regress_toggle:
            ridge_regr_output, _ = ridge_regression(train_test=train_test_data, alpha=ridge_regr_alpha, cv_number=CV_samples, feature_importance=feature_importance, search_cv_type=search_cv_type, n_iter_number=n_iter_number)
            regression_metrics.append(ridge_regr_output['regression_metric'])
            if ridge_regr_output['cv_analysis'] != None:
                regression_cv_analysis.append(ridge_regr_output['cv_analysis'])
            if ridge_regr_output['FI_analysis'] != None:
                regression_FI_analysis.append(ridge_regr_output['FI_analysis'])

        if regression_metrics:
            metrics_results = pd.DataFrame(regression_metrics).set_index('Regression model').T
            container_metrics.write(f"<h3 style='text-align: center'>ML model: metrics</h3>", unsafe_allow_html=True)
            container_metrics.dataframe(metrics_results, use_container_width=True)
        else:
            st.sidebar.warning('Choose models to construct', icon='⚠️')

        if regression_cv_analysis:
            cv_analysis_results = pd.DataFrame(regression_cv_analysis).set_index('Regression model').T
            container_cv_analysis.write(f"<h3 style='text-align: center'>ML model: CrossValidation analysis</h3>", unsafe_allow_html=True)
            container_cv_analysis.dataframe(cv_analysis_results, use_container_width=True)
        
        
        if regression_FI_analysis:
            container_FI_analysis.write(f"<h3 style='text-align: center'>ML model: Feature Importance analysis</h3>", unsafe_allow_html=True)
            for result in regression_FI_analysis:
                container_FI_analysis.write(f"<h4 style='text-align: center'>{result['dataframe'].columns[1]}</h4>", unsafe_allow_html=True)
                cols = container_FI_analysis.columns((1,3))
                cols[0].dataframe(result['dataframe'].set_index('feature'))
                cols[1].pyplot(result['graph'])

@st.cache_data
def plot_pairplot(df=None, hue=None):
    fig = sns.pairplot(df, hue=hue)
    st.pyplot(fig)

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def data_animation_random(key=None):      
    if key == 'no file found':
        json_list = [
            'https://lottie.host/e1e299c2-9ef5-4fa2-8b95-7f881c2da03a/Z2IjXKI87w.json',
            'https://lottie.host/04d16130-c492-419c-a587-8de20f4a3b0a/8gXeSvWLy9.json',
            'https://lottie.host/6d688eff-6512-4267-9d77-74e7b29c4902/5LsbUYmlEv.json',
        ]
    else: 
        json_list = [
            'https://lottie.host/e7067d99-22c7-4bba-ba33-0d498828494b/QJOsiWRSol.json',
            'https://lottie.host/e85f4f18-4bb5-4026-b45f-72067f4895e5/wSZLsbtSCL.json',
            'https://lottie.host/ff78c70c-435e-48cd-b365-9154723377cc/TY08GBoTXG.json',
            'https://lottie.host/1c3f5e7d-a2c2-4c4e-b0be-c9f1a2fa1240/rz3tOtlonW.json',
            'https://lottie.host/b82ff1c4-592b-4530-b34b-b768563f279d/IspyFrFfH9.json',
        ]
    
    lottie_coding = random.choice(json_list)   
    return load_lottie_url(lottie_coding)

def readme():
    cols = st.columns((2,1))
    with cols[1]:
        st_lottie(data_animation_random(key='intro'), height=200, width=300, key='load_data_animation')
    cols[0].write('''
            ## Introduction

            Welcome to the Data Analysis and Machine Learning Toolkit! This program is designed to simplify the process of working with datasets that have a target feature, whether it's numeric or categorical. With this toolkit, you can perform various data pre-processing tasks, conduct data analysis with visualization, handle outlier analysis, and implement supervised machine learning models for both regression and classification tasks.
            ''')  
    st.write('''
                ## Features

                1. **Homepage - Data Pre-processing:**
                - Select and drop unnecessary columns from the dataset.
                - Impute missing values based on a predetermined strategy or remove rows with missing data if necessary.
                - Apply ordinal encoding to categorical features with an inherent order.
                - Utilize one-hot encoding for categorical features without a natural order.
                - Identify and drop duplicate rows, if any, to ensure data integrity.
                - Manually define the feature types, including the target variable, numerical features, and categorical features.

                2. **Data understanding - Data Analysis and Visualization:**
                - Generate descriptive statistics for your dataset.
                - Visualize data distributions, correlations, and more.
                - Explore the relationship between features and the target variable.   
                - Identify and visualize outliers.

                3. **Data understanding - Outlier Analysis:**
                - Detect and visualize outliers using various methods such as Z-score, IQR, or visualization techniques like scatter plots and box plots.

                4. **Data Processing:**
                - Choose to treat outliers based on your analysis.
                - Prepare the data for machine learning models.
                - Perform any required normalization or standardization of numerical features.

                5. **Model construction - Supervised Machine Learning:**
                - Split the dataset into training and testing subsets for further analysis and modeling.
                - Implement regression models for numeric target features.
                - Implement classification models for categorical target features.
                - Fine-tune model hyperparameters.
                - Evaluate model performance using various metrics.

                ## Contributors

                - Ramazan Abylkassov, MD

                ## Support

                If you have any questions, encounter issues, or want to contribute to the toolkit, please contact - [LinkedIn](https://www.linkedin.com/in/ramazan-abylkassov-23965097/).

                Happy data analysis and machine learning!
             ''')

