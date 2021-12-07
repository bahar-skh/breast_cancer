# Copyright: Bahar (Fatemeh) Safikhani

import itertools

import numpy as np
import pandas as pd

from fast_ml.model_development import train_valid_test_split
from fast_ml.utilities import reduce_memory_usage

def dataset(ratio, random_state):
    '''
    ratio (list): list containing the ratio of train/validation/test sets in that order.
    '''
    attribute_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                   'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    attribute_types = ['mean', 'se', 'worst']

    col_names = ['Id', 'Diagnosis'] + \
                [f"{attr_name}_{attr_type}" for attr_type in attribute_types for attr_name in attribute_names]

    df = pd.read_csv('data/wdbc.data', header=None, names=col_names)
    
    if df.shape[0] == df['Id'].nunique():
        print("There is no data leakage between patients.")

    df = df.drop(columns='Id').copy(deep=True)
    df = reduce_memory_usage(df, convert_to_category=True)

    variables = list(set(df.columns.to_list()) - set(['Diagnosis']))
    target = 'Diagnosis'

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        train_valid_test_split(df, target='Diagnosis', train_size=ratio[0], valid_size=ratio[1], test_size=ratio[2], \
                               random_state=random_state)
    
    return df, X_train, y_train, X_valid, y_valid, X_test, y_test