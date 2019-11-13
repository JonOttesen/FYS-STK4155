import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

def read_file(file_name):
    """
    The function used for reading the credit card data
    """

    file_name = "default of credit card clients.xls"
    credit_data = pd.read_excel(file_name, index_col=0, index_row = 1)
    credit_data_np = np.copy(credit_data.to_numpy()[1:])

    X = np.delete(credit_data_np, -1, axis = 1).astype(np.float64)
    y = np.copy(credit_data_np[:, -1]).astype(np.float64)
    return X, y

def preprocess(X, y, method = 'Standardize'):
    """
    The function used for preprocessing of the credit card data
    """

    X[:, 11:23] = (X[:, 11:23] - np.mean(X[:, 11:23], axis = 0, keepdims = True))/np.std(X[:, 11:23], axis = 0, keepdims = True)  #Standardize the columns
    X[:, 0] = (X[:, 0] - np.mean(X[:, 0], axis = 0, keepdims = True))/np.std(X[:, 0], keepdims = True)  #Standardize the columns
    X[:, 4] = (X[:, 4] - np.mean(X[:, 4], axis = 0, keepdims = True))/np.std(X[:, 4], keepdims = True)  #Standardize the columns

    #del1 = np.logical_and(np.logical_and(X[:, 2] != 0, X[:, 2] != 5), np.logical_and(X[:, 2] != 6, X[:, 3] != 0))
    #X = X[del1]
    #y = y[del1]

    ohe = OneHotEncoder()  #One Hot Encoder for Education, sex and marrige status
    ohe.fit(X[:, [1, 2, 3]])
    transform1 = ohe.transform(X[:, [1, 2, 3]]).toarray()

    X_pay_columns = np.copy(X[:, 5:11])
    X_pay_columns[X_pay_columns > 0] = 1
    #X_pay_columns[X_pay_columns == -1] = 1

    ohe = OneHotEncoder(categories = 'auto', drop = [[1]]*6)  #One Hot Encoder for PAY_0 to PAY_6
    ohe.fit(X_pay_columns)
    transform2 = ohe.transform(X_pay_columns).toarray()

    X[:, 5:11][X[:, 5:11] < 0] = 0

    X = np.concatenate((np.delete(X, [1, 2, 3], axis = 1), transform1, transform2), axis = 1)
    return X, y
