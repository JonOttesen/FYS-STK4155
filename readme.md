# General readme about the projects in FYS-STK 4155

## Project 1

Project 1 is about linear regression and the three main programs are:
'''
regression.py
results.py
terrrain.py
'''

### 'regression.py'
This is a class consisting of the regression methods like OLS, ridge and lasso regression. It aslo contains
a general train test split and k-fold cross validation where the number of folds can be chosen by the user.
The main modules are the ** OLS, ridge and lasso** modules.

Other important modules are the 'z_tilde' module which uses the calculated beta and a given design matrix X
to create the model.
The remaining are:
'''
MSE
R_squared
beta_variance
sigma_squared
lambda_best_fit
'''
For better explanation see the documentation in the file.

### 'results.py'
This file creates all results from the Franke function data and saves the images in the Results folder.

### 'terrain.py'
This file creates all the results from the terrain data and saves the images in the Resutls_terrain folder.
