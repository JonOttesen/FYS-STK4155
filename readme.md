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
The main modules are the **OLS, ridge and lasso** modules.

Other important modules are the 'z_tilde' module which uses the calculated beta and a given design matrix X
to create the model.
The remaining are:

```
MSE(z_tilde, z)
R_squared(z_tilde, z)
beta_variance(sigma_squared, X = 'None', lam = 0)
sigma_squared(sz_tilde, z, p = 'polynomial order')
lambda_best_fit(method, fold = 4, n_lambda = 1001, l_min = -5.5, l_max = -0.5, random_num = True, use_seed = False, seed = 42, X = 'None', z = 2, max_iter = 1001, full = False, precompute = True)
```
For better explanation see the documentation in the file.

### 'results.py'
This file creates all results from the Franke function data and saves the images in the Results folder.

### 'terrain.py'
This file creates all the results from the terrain data and saves the images in the Resutls_terrain folder.
