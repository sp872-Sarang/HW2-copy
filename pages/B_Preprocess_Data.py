import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from pages.A_Explore_Dataset import user_input_features
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

# Checkpoint 2
def summarize_missing_data(df, top_n=3):
    """
    This function summarizes missing values in the dataset

    Input: 
        - df: the pandas dataframe
        - top_n: top n features with missing values, default value is 3
    Output: 
        - a dictionary containing the following keys and values: 
            - 'num_categories': counts the number of features that have missing values
            - 'average_per_category': counts the average number of missing values across features
            - 'total_missing_values': counts the total number of missing values in the dataframe
            - 'top_missing_categories': lists the top n features with missing values
    """
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': []}

    # Used for top categories with missing data

    # Compute missing statistics

    # Display missing statistics

    st.write('summarize_missing_data not implemented yet.')
    return out_dict

# Checkpoint 3
def remove_nans(df):
    """
    This function removes all NaN values in the dataframe

    Input: 
        - df: pandas dataframe
    Output: 
        - df: updated df with no Nan observations
    """
    # Add code here

    st.write('remove_nans not implemented yet.')
    return df

# Checkpoint 4
def remove_outliers(df, feature):
    """
    This function removes the outliers of the given feature(s)

    Input: 
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output: 
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    dataset, lower_bound, upper_bound = None, -1, -1

    # Add code here

    st.write('remove_outliers not implemented yet.')
    return dataset, lower_bound, upper_bound

# Checkpoint 5
def one_hot_encode_feature(df, feature):
    """
    This function performs one-hot-encoding on the given features using pd.get_dummies

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform one-hot-encoding
    Output: 
        - df: dataframe with one-hot-encoded feature
    """    
    # Add code here

    st.write('one_hot_encode_feature not implemented yet.')
    #st.write('Feature {} has been one-hot encoded.'.format(feature))
    return df

# Checkpoint 5
def integer_encode_feature(df, feature):
    """
    This function performs integer-encoding on the given features using OrdinalEncoder()

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform integer-encoding
    Output: 
        - df: dataframe with integer-encoded feature
    """
    # Add code here

    st.write('integer_encode_feature not implemented yet.')
    #st.write('Feature {} has been integer encoded.'.format(feature))
    return df

# Checkpoint 6
def create_feature(df, math_select, math_feature_select, new_feature_name):
    """
    Create a new feature and update the pandas dataframe using four mathematical operations including 
        1) addition, 2) subtraction, 3) multiplication, 4) division, and 5) square root, 6) ceil, and 7) floor.

    Input: 
        - df: the pandas dataframe
        - math_select: the math operation to perform on the selected features
        - math_feature_select: the features to be performed on
        - new_feature_name: the name for the new feature
    Output: 
        - df: the udpated dataframe
    """
    # Drop Nans in df

    # Compute updated fetaure and store in df
    st.write('create_feature not implemented yet.')
    return df

# Complete this helper function from HW1
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """
    This function computes some statistical values of the selected features

    Input: 
        - X: the pandas dataframe
        - stats_feature_select: the features to be calculated stats on
        - stats_select: the stats values to be calculated
    Output: 
        - out_dict: a dictionary contains the computed the given stats on the given features, with the following keys and values
            - 'mean': the mean of each selected feature
            - 'median': the median of each selected feature
            - 'max': the max of each selected feature
            - 'min': the min of each selected feature
        - output_str: a string contains the stats information (also to be rendered on the page) in the following format:
            'stats_feature_select[i] stats_select[j]: value    |'
    """
    output_str = ''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }

    # Add code here

    st.write('create_feature not implemented yet.')
    return output_str, out_dict

# Complete this helper function from HW1 
def impute_dataset(X, impute_method):
    """
    This function imputes the NaN in the dataframe with three possible ways

    Input: 
        - X: the pandas dataframe
        - impute_method: the method to impute the NaN in the dataframe, options are -
            - 'Zero': to replace NaN with zero
            - 'Mean': to replace NaN with the mean of the corressponding feature column
            - 'Median': to replace NaN with the median of the corressponding feature column
    Output: 
        - X: the updated dataframe
    """
    # Add code here

    st.write('impute_dataset not implemented yet.')
    return X

# Complete this helper function from HW1 
def remove_features(X, removed_features):
    """
    This function drops selected feature(s)

    Input: 
        - X: the pandas dataframe
        - removed_features: the features to be removed
    Output: 
        - X: the updated dataframe
    """
    # Add code here

    st.write('remove_features not implemented yet.')
    return X

# Complete this helper function from HW1
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """
    This function computes some statistical values of the selected features

    Input: 
        - X: the pandas dataframe
        - stats_feature_select: the features to be calculated stats on
        - stats_select: the stats values to be calculated
    Output: 
        - out_dict: a dictionary contains the computed the given stats on the given features, with the following keys and values
            - 'mean': the mean of each selected feature
            - 'median': the median of each selected feature
            - 'max': the max of each selected feature
            - 'min': the min of each selected feature
        - output_str: a string contains the stats information (also to be rendered on the page) in the following format:
            'stats_feature_select[i] stats_select[j]: value    |'
    """
    output_str = ''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    # Add code here

    st.write('compute_descriptive_stats not implemented yet.')
    return output_str, out_dict

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
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.dataframe(df)

    # Show summary of missing values including 
    missing_data_summary = summarize_missing_data(df)

    ############################################# MAIN BODY #############################################

    # Remove feature
    st.markdown('### Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        df.columns,
    )
    if (removed_features):
        df = remove_features(df, removed_features)

    # Display updated dataframe
    st.dataframe(df)

    # Handle NaNs
    remove_nan_col, impute_col = st.columns(2)

    with (remove_nan_col):
        # Remove Nans
        st.markdown('### Remove Nans')
        if st.button('Remove Nans'):
            df = remove_nans(df)
            st.write('Nans Removed')
        else:
            st.write('Dataset might contain Nans')

    with (impute_col):
        # Clean dataset
        st.markdown('### Impute data')
        st.markdown('Transform missing values to 0, mean, or median')

        # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
        impute_method = st.selectbox(
            'Select cleaning method',
            ('None', 'Zero', 'Mean', 'Median')
        )

        # Call impute_dataset function to resolve data handling/cleaning problems
        if (impute_method):
            df = impute_dataset(df, impute_method)

    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.dataframe(df)

    # Remove outliers
    st.markdown("### Inspect Features and Remove Outliers")
    numeric_columns = list(df.select_dtypes(include='number').columns)

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot'],
        key='sidebar_chart'
    )

    # Draw plots
    # Add code here

    st.markdown('### Inspect Features for outliers')
    outlier_feature_select = None
    outlier_feature_select = st.selectbox(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df, lower_bound, upper_bound = remove_outliers(
            df, outlier_feature_select)
        st.write('Outliers for feature {} are lower than {} and higher than {}'.format(
            outlier_feature_select, lower_bound, upper_bound))
        st.write(df)

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_feature_select_int = st.selectbox(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            if 'integer_encode' not in st.session_state:
                st.session_state['integer_encode'] = {}
            if text_feature_select_int not in st.session_state['integer_encode']:
                st.session_state['integer_encode'][text_feature_select_int] = True
            else:
                st.session_state['integer_encode'][text_feature_select_int] = True
            df = integer_encode_feature(df, text_feature_select_int)
    
    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            if 'one_hot_encode' not in st.session_state:
                st.session_state['one_hot_encode'] = {}
            if text_feature_select_onehot not in st.session_state['one_hot_encode']:
                st.session_state['one_hot_encode'][text_feature_select_onehot] = True
            else:
                st.session_state['one_hot_encode'][text_feature_select_onehot] = True
            df = one_hot_encode_feature(df, text_feature_select_onehot)

    # Show updated dataset
    st.write(df)

    # Create New Features
    st.markdown('## Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        df = create_feature(
                            df, math_select, math_feature_select, new_feature_name)
                        st.write(df)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    df = create_feature(df, math_select, [
                                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(df)

    # Descriptive Statistics
    st.markdown('### Summary of Descriptive Statistics')

    stats_feature_select = st.multiselect(
        'Select features for statistics',
        numeric_columns,
    )

    # Select statistic to compute
    if (stats_feature_select):
        stats_select = st.multiselect(
            'Select statistics to display',
            ['Mean', 'Median', 'Max', 'Min']
        )

        # Compute Descriptive Statistics including mean, median, min, max
        display_stats, _ = compute_descriptive_stats(
            df, stats_feature_select, stats_select)

    # Display updated dataframe
    # Add code here ...
    
    # Store dataset in st.session_state
    # Add code here ...

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Preprocess: Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.write('Continue to Train Model')