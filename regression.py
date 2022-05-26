import numpy as np
import pandas as pd
from sklearn import linear_model


def data_model(n, sqrt_cov, w_star, sigma, 
               dist = 'Gaussian', response = 'well-specified'):

    """
    input:
    n        - sample size
    sqrt_cov - the square root of the data covariance matrix, shape = (d,d)
    w_star   - the ground truth linear coefficients, shape = (d, 1) 
    sigma    - the standard deviation of the residual

    output:
    X - the design matrix, shape = (n, d)
    Y - the response vector, shape = (n, 1)
    """
    
    d = len(sqrt_cov)
    
    # compute the design matrix
    if dist == 'Gaussian':
        X = np.random.normal(size = (n, d))

    elif dist == 'Uniform':
        X = (np.random.uniform(size = (n, d)) - 0.5) * np.sqrt(12)
        
    elif dist == 'Rademacher':
        X = (np.random.binomial(1, 0.5, size = n * d) - 0.5) * 2
        X = X.reshape(n, d)
        
    elif dist == 'Poisson':
        X = np.random.poisson(size = (n, d)) - 1
        
    elif dist == 'Laplace':
        X = np.random.laplace(size = (n, d)) / np.sqrt(2)
        
    elif dist == 'StudentT':
        X = np.random.standard_t(5, size = (n, d)) / np.sqrt(5/3)
        
    elif dist == 'Weibull':
        X = (np.random.weibull(0.5, size = (n, d)) - 2) / np.sqrt(20)
        
    elif dist == 'LogNormal':
        X = (np.random.lognormal(size = (n, d)) - np.exp(0.5)) / np.sqrt(np.exp(2) - np.exp(1))
        
    X = X * sqrt_cov.reshape(1, -1)

    # compute the response vector
    if response == 'well-specified':
        Y = X @ w_star + np.random.normal(scale = sigma, size = (n, 1))
        
    elif response == 'mis-specified':
        linear_signal = X @ w_star
        nonlinear_term = np.abs(X[:, [0]]) * np.cos(X[:, [1]]) 
        residual = X[:, [2]] * np.random.normal(scale = sigma, size = (n, 1))
        Y = linear_signal + nonlinear_term + residual
        
    return (X, Y)


def dist_summary(sqrt_cov, w_star, sigma, 
                 dist = 'Gaussian', response = 'well-specified'):
    
    if response == 'well-specified':

        null_risk = w_star.transpose() @ ((sqrt_cov.reshape(-1, 1) ** 2) * w_star) + (sigma ** 2)
        optimal_risk = sigma ** 2
        optimal_predictor = w_star
        return (null_risk[0, 0], optimal_risk, optimal_predictor)
    
    elif response == 'mis-specified':

        # estimate null risk, bayes risk and the optimal predictor by monte-carlo

        size = 1000000 # sample size
        if dist in ['Weibull', 'LogNormal']:
            size *= 100

        if dist == 'Gaussian':
            x = np.random.normal(size = size)

        elif dist == 'Uniform':
            x = (np.random.uniform(size = size) - 0.5) * np.sqrt(12)
            
        elif dist == 'Rademacher':
            x = (np.random.binomial(1, 0.5, size = size) - 0.5) * 2
            
        elif dist == 'Poisson':
            x = np.random.poisson(size = size) - 1
            
        elif dist == 'Laplace':
            x = np.random.laplace(size = size) / np.sqrt(2)

        elif dist == 'StudentT':
            x = np.random.standard_t(5, size = size) / np.sqrt(5/3)
            
        elif dist == 'Weibull':
            x = (np.random.weibull(0.5, size = size) - 2) / np.sqrt(20)
            
        elif dist == 'LogNormal':
            x = (np.random.lognormal(size = size) - np.exp(0.5)) / np.sqrt(np.exp(2) - np.exp(1))

        abs_x = np.mean(np.abs(x))
        cos_x = np.mean(np.cos(x))
        cos_2_x = np.mean(np.cos(x) ** 2)
        if dist in ['Poisson', 'Weibull', 'LogNormal']:
            x_abs_x = np.mean(x * np.abs(x))
            x_cos_x = np.mean(x * np.cos(x))
        else:
            x_abs_x = 0
            x_cos_x = 0

        null_risk = w_star.transpose() @ ((sqrt_cov.reshape(-1, 1) ** 2) * w_star) 
        null_risk += cos_2_x + (sigma ** 2)
        null_risk += 2 * (x_abs_x * cos_x * w_star[0,0] + abs_x * x_cos_x * w_star[1,0]) 
        optimal_risk = (sigma ** 2) + cos_2_x - (x_abs_x * cos_x) ** 2 - (abs_x * x_cos_x) ** 2 
        optimal_predictor = w_star.copy()
        optimal_predictor[0] += x_abs_x * cos_x
        optimal_predictor[1] += abs_x * x_cos_x
        
        return (null_risk[0, 0], optimal_risk, optimal_predictor)    


def isotropic_experiment(num_experiment, alphas, 
                         n, d, w_star, sigma, dist = 'Gaussian', response = 'well-specified', 
                         method = 'ridge'):

    sqrt_cov = np.array( [1] * d )
    null_risk, optimal_risk, optimal_predictor = dist_summary(sqrt_cov, w_star, sigma, 
                                                              dist = dist, response = response)

    L = np.zeros((num_experiment, len(alphas)))
    Lhat = np.zeros((num_experiment, len(alphas)))
    bound = np.zeros((num_experiment, len(alphas)))

    for i in range(num_experiment):

        X, Y = data_model(n, sqrt_cov, w_star, sigma, 
                          dist = dist, response = response)
        
        if method == 'ridge':

            u, s, vt = np.linalg.svd(X, full_matrices=False)
            v = np.transpose(vt)
            uTy = np.transpose(u) @ Y
            s = s.reshape(-1, 1)

            norm_x = np.sqrt(n * np.sum(sqrt_cov ** 2))

            for j in range(len(alphas)):
                d = s / (s ** 2 + np.exp(alphas[j]))
                w = v @ (d * uTy)
                w_perp = w -  (w.transpose() @ optimal_predictor) / np.sum(optimal_predictor ** 2) * optimal_predictor

                L[i, j] = np.sum((sqrt_cov.reshape(-1, 1) * (w - optimal_predictor)) ** 2) + optimal_risk
                Lhat[i, j] = np.mean((X @ w - Y) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w_perp) / n) ** 2

        elif method == 'LASSO':

            _, coef_path, _ = linear_model.lasso_path(X, Y, alphas=np.exp(alphas), max_iter = 10000)

            norm_x = np.zeros(100)
            for k in range(100):
                random_signs = (np.random.binomial(1, 0.5, size = (n, 1)) - 0.5) * 2
                norm_x[k] = np.linalg.norm(np.sum(X * random_signs, axis = 0), ord = np.inf)
            norm_x = np.mean(norm_x)

            for j in range(len(alphas)):
                w = coef_path[0, :, len(alphas) - 1 - j].reshape(-1, 1)
                w_perp = w -  (w.transpose() @ optimal_predictor) / np.sum(optimal_predictor ** 2) * optimal_predictor

                L[i, j] = np.sum((sqrt_cov.reshape(-1, 1) * (w - optimal_predictor)) ** 2) + optimal_risk
                Lhat[i, j] = np.mean((X @ w - Y) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w_perp, ord = 1) / n) ** 2


    # store experimental results in a pandas dataframe
    df = pd.DataFrame()
    df['Population risk'] = L.flatten('F')
    df['Empirical risk'] = Lhat.flatten('F')
    df['Risk bound'] = bound.flatten('F')
    df['Bayes risk'] = optimal_risk
    df['Null risk'] = null_risk
    df['Distribution'] = dist
    
    df.index = np.repeat(alphas, num_experiment)
    df.index.name = 'log regularization parameter'

    return df


def experiment(num_experiment, alphas, 
               n, sqrt_cov, w_star, sigma, 
               dist = 'Gaussian', response = 'well-specified',
               method = 'ridge', cov_split = 0):

    null_risk, optimal_risk, optimal_predictor = dist_summary(sqrt_cov, w_star, sigma, 
                                                              dist = dist, response = response)

    L = np.zeros((num_experiment, len(alphas)))
    Lhat = np.zeros((num_experiment, len(alphas)))
    bound = np.zeros((num_experiment, len(alphas)))

    for i in range(num_experiment):

        X, Y = data_model(n, sqrt_cov, w_star, sigma, 
                          dist = dist, response = response)
        
        if method == 'ridge':

            u, s, vt = np.linalg.svd(X, full_matrices=False)
            v = np.transpose(vt)
            uTy = np.transpose(u) @ Y
            s = s.reshape(-1, 1)

            norm_x = np.sqrt(n * np.sum(sqrt_cov[cov_split:] ** 2))

            for j in range(len(alphas)):
                d = s / (s ** 2 + np.exp(alphas[j]))
                w = v @ (d * uTy)

                L[i, j] = np.sum((sqrt_cov.reshape(-1, 1) * (w - optimal_predictor)) ** 2) + optimal_risk
                Lhat[i, j] = np.mean((X @ w - Y) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w) / n) ** 2

        elif method == 'LASSO':

            _, coef_path, _ = linear_model.lasso_path(X, Y, alphas=np.exp(alphas), max_iter = 10000)

            norm_x = np.zeros(100)
            for k in range(100):
                random_signs = (np.random.binomial(1, 0.5, size = (n, 1)) - 0.5) * 2
                norm_x[k] = np.linalg.norm(np.sum(X[:, cov_split:] * random_signs, axis = 0), ord = np.inf)
            norm_x = np.mean(norm_x)

            for j in range(len(alphas)):
                w = coef_path[0, :, len(alphas) - 1 - j].reshape(-1, 1)

                L[i, j] = np.sum((sqrt_cov.reshape(-1, 1) * (w - optimal_predictor)) ** 2) + optimal_risk
                Lhat[i, j] = np.mean((X @ w - Y) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w, ord = 1) / n) ** 2


    # store experimental results in a pandas dataframe
    df = pd.DataFrame()
    df['Population risk'] = L.flatten('F')
    df['Empirical risk'] = Lhat.flatten('F')
    df['Risk bound'] = bound.flatten('F')
    df['Bayes risk'] = optimal_risk
    df['Null risk'] = null_risk
    df['Distribution'] = dist
    
    df.index = np.repeat(alphas, num_experiment)
    df.index.name = 'log regularization parameter'

    return df






            
    