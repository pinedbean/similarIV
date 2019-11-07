import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .instant_function import data_vars


def create_categorical_onehot(df,category_columns):
    category_dataframe = []
    for category_column in category_columns:
        category_dataframe.append(pd.get_dummies(df[category_column],prefix='col_'+category_column))
    
    category_dataframe_feature = pd.concat(category_dataframe,axis=1)
    return category_dataframe_feature


def create_norm_continuos_columns(df, continuos_columns):
    df_norm = df[continuos_columns].fillna(0)
    norm_continuos_columns = (df_norm[continuos_columns]-df_norm[continuos_columns].mean())/(df_norm[continuos_columns].std())
    mean_dict = dict(df[continuos_columns].mean())
    std_dict = dict(df[continuos_columns].std())
    return norm_continuos_columns, mean_dict, std_dict


def combine_continus_norm_and_categorical_onehot_and_sep_target(df, continuos_columns, category_columns, target_columns):
    norm_continuos_columns, mean_dict, std_dict = create_norm_continuos_columns(df, continuos_columns)
    category_dataframe_feature = create_categorical_onehot(df,category_columns)
    target_df = df[target_columns]
    
    feature_df = pd.concat([norm_continuos_columns, category_dataframe_feature], axis=1)
    feature_columns = feature_df.columns
    return feature_df, target_df, mean_dict, std_dict, feature_columns


def get_IV(feature_df, target_df):
    final_iv, IV = data_vars(feature_df, target_df)
    ivs = np.zeros(len(feature_df.columns))
    for i,col in enumerate(feature_df.columns):
        ivs[i] = IV[IV['VAR_NAME']==col]['IV'].values[0]
    return IV, ivs 


def norm_mat(x):
    return x/(np.sqrt(np.sum(x**2,1))).reshape(-1,1)


def get_pos_feat(feature_df, target_df, ivs):
    pos_feat = feature_df.loc[target_df[target_df==1].index].values*ivs
    pos_feat_norm = norm_mat(pos_feat)
    return pos_feat_norm

def process_test_data(df, continuos_columns, category_columns, mean_dict, std_dict, feature_columns):
    df_norm = df[continuos_columns].fillna(0)
    norm_continuos_columns = (df[mean_dict] - list(mean_dict.values()))/list(std_dict.values())
    
    category_dataframe = []
    for category_column in category_columns:
        category_dataframe.append(pd.get_dummies(df[category_column],prefix='col_'+category_column))
    
    category_dataframe_feature = pd.concat(category_dataframe,axis=1)
    
    feature_test = pd.concat([norm_continuos_columns, category_dataframe_feature], axis=1)
    
    non_in_test_columns = list(set(list(feature_columns)) - set(list(feature_test.columns)))
    
    for non_in_test_column in non_in_test_columns:
        feature_test[non_in_test_column] = 0
    
    feature_test = feature_test[feature_columns]
    
    return feature_test 