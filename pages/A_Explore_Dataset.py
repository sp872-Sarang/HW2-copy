import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request
from itertools import combinations

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

feature_lookup = {
    'longitude': '**longitude** - longitudinal coordinate',
    'latitude': '**latitude** - latitudinal coordinate',
    'housing_median_age': '**housing_median_age** - median age of district',
    'total_rooms': '**total_rooms** - total number of rooms per district',
    'total_bedrooms': '**total_bedrooms** - total number of bedrooms per district',
    'population': '**population** - total population of district',
    'households': '**households** - total number of households per district',
    'median_income': '**median_income** - median income',
    'ocean_proximity': '**ocean_proximity** - distance from the ocean',
    'median_house_value': '**median_house_value**',
    'city':'city location of house',
    'road':'road of the house',
    'county': 'county of house',
    'postcode':'zip code',
    'rooms_per_household':'average number of rooms per household',
    "number_bedrooms":'number of bedrooms',
    "number_bathrooms": 'number of bathrooms',
    

}
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

# Checkpoint 1
def compute_correlation(X, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings 
        with description of magnitude and direction of correlation

    Input: 
        - X: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = None
    cor_summary_statements = []

    # Add code here

    st.write('compute_correlation not implemented yet.')
    return correlation, cor_summary_statements

# Helper Function
def user_input_features(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - dictionary of sidebar filters on features
    """
    side_bar_data = []

    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data

# Helper Function
def display_features(df, feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup).

    Inputs:
        - df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
        - feature_lookup (dict): A dictionary containing the descriptions for the features.
    Outputs: None
    """
    numeric_columns = list(df.select_dtypes(include='number').columns)
    #for idx, col in enumerate(df.columns):
    for idx, col in enumerate(numeric_columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s' % (idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s' % (idx, col))

# Helper Function
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This function fetches a dataset from a URL, saves it in .tgz format, and extracts it to a specified directory path.

    Inputs:
    - housing_url (str): The URL of the dataset to be fetched.
    - housing_path (str): The path to the directory where the extracted dataset should be saved.

    Outputs: None
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Helper function
@st.cache
def convert_df(df):
    """
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    """
    return df.to_csv().encode('utf-8')

###################### FETCH DATASET #######################
df = None
# df = ... Add code here: read in data and store in st.session_state

if df is not None:

    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')

    # Restore dataset if already in memory
    st.session_state['data'] = df

    # Display feature names and descriptions (from feature_lookup)
    display_features(df, feature_lookup)

    # Display dataframe as table
    st.dataframe(df.describe())

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')
    
    # Specify Input Parameters

    # Collect user plot selection using user_input_features()

    # Draw plots

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")
    # Collect features for correlation analysis using multiselect
    
    # Compute correlation between selected features

    # Display correlation of all feature pairs with description of magnitude and direction of correlation

    # Store dataset in st.session_state
    # Add code here ...

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.markdown('#### Continue to Preprocess Data')