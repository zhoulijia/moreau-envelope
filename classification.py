import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


def data_model(n, sqrt_cov, w_star, b_star, 
               dist = 'Gaussian'):
    
    """
    input:
    n        - sample size
    sqrt_cov - the square root of the data covariance matrix, shape = (d,d)
    w_star   - the ground truth linear coefficients, shape = (d, 1) 
    b_star    - the ground truth bias term

    output:
    X - the design matrix, shape = (n, d)
    Y - the label vector, shape = (n, 1)
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
    eta = X @ w_star + b_star
    eta = 1 / (1 + np.exp(-1 * eta))
    Y = (np.random.binomial(1, eta.flatten()) - 0.5) * 2

    return (X, Y)


def population_error(w, b, 
                     sqrt_cov, w_star, b_star, dist = 'Gaussian', 
                     size = 10000):
    
    # use simulations to estimate the test error of (w, b)
    # when the data distribution is specified by sqrt_cov and (w_star, b_star)

    if dist == 'Gaussian':
        
        sqrt_Sigma_w = sqrt_cov.reshape(-1, 1) * w
        sqrt_Sigma_w_star = sqrt_cov.reshape(-1, 1) * w_star

        M_norm_w = np.linalg.norm(sqrt_Sigma_w)
        M_norm_w_star = np.linalg.norm(sqrt_Sigma_w_star)
        wSigmaw_star = np.transpose(sqrt_Sigma_w) @ (sqrt_Sigma_w_star)

        eta = np.random.normal(b_star, M_norm_w_star, size = (size, 1))
        mu = b + (wSigmaw_star / (M_norm_w_star ** 2)) * (eta - b_star)
        sigma = (M_norm_w ** 2) - ((wSigmaw_star / M_norm_w_star) ** 2)
        sigma = np.sqrt(np.maximum(0,sigma))
        eta_hat = np.random.normal(mu, sigma)

        sigmoid_eta = 1 / (1 + np.exp(-1 * eta))

        return np.mean(sigmoid_eta * (np.maximum(0, 1 - eta_hat) ** 2) + 
                       (1 - sigmoid_eta) * (np.maximum(0, 1 + eta_hat) ** 2))    
    
    else:
        
        X, Y = data_model(size, sqrt_cov, w_star, b_star,
                          dist = dist) 
        return np.mean(np.maximum(0, 1 - (X @ w + b) * Y.reshape(-1, 1)) ** 2)


def bayes(sqrt_cov, w_star, b_star, dist = 'Gaussian', 
          size = 100000, eta0 = 0.1):
    
    # generate a lot of data from the true distribution and then
    # run SGD to estimate the optimal test error achieved by a linear predictor
        
    X, Y = data_model(size, sqrt_cov[[0]], w_star[[0], :], b_star, 
                      dist = dist) 
    
    clf = SGDClassifier(loss = 'squared_hinge', alpha = 0,
                        learning_rate = 'invscaling', eta0 = eta0, average = True)
    clf.partial_fit(X, Y, classes = [-1., 1.])
    
    w = np.zeros((len(sqrt_cov), 1))
    w[0, 0] = clf.coef_[0, 0]
    b = clf.intercept_[0]
    
    return (w, population_error(w, b, sqrt_cov, w_star, b_star, dist = dist, size = size))


def isotropic_experiment(num_experiment, alphas, 
                         n, d, w_star, b_star, dist = 'Gaussian', 
                         method = 'l2'):

    sqrt_cov = np.array( [1] * d )
    optimal_predictor, optimal_risk= bayes(sqrt_cov, w_star, b_star, dist = dist)

    L = np.zeros((num_experiment, len(alphas)))
    Lhat = np.zeros((num_experiment, len(alphas)))
    bound = np.zeros((num_experiment, len(alphas)))

    for i in range(num_experiment):

        X, Y = data_model(n, sqrt_cov, w_star, b_star, dist = dist) 
        
        if method == 'l2':

            norm_x = np.sqrt(n * np.sum(sqrt_cov ** 2))

            for j in range(len(alphas)):

                clf = LinearSVC(random_state=0, max_iter = 1000, 
                                penalty = 'l2', loss = 'squared_hinge',
                                dual = True, C = 1 / np.exp(alphas[j]))
                clf.fit(X, Y)

                w = clf.coef_.reshape(-1, 1)
                b = clf.intercept_[0]
                w_perp = w -  (w.transpose() @ optimal_predictor) / np.sum(optimal_predictor ** 2) * optimal_predictor

                L[i, j] = population_error(w, b, 
                                           sqrt_cov, w_star, b_star, dist = dist)
                Lhat[i, j] = np.mean(np.maximum(0, 1 - (X @ w + b) * Y.reshape(-1, 1)) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w_perp) / n) ** 2

        elif method == 'l1':

            norm_x = np.zeros(100)
            for k in range(100):
                random_signs = (np.random.binomial(1, 0.5, size = (n, 1)) - 0.5) * 2
                norm_x[k] = np.linalg.norm(np.sum(X * random_signs, axis = 0), ord = np.inf)
            norm_x = np.mean(norm_x)

            for j in range(len(alphas)):
                
                clf = LinearSVC(random_state=0, max_iter = 1000, 
                                penalty = 'l1', loss = 'squared_hinge',
                                dual = False, C = 1 / np.exp(alphas[j]))
                clf.fit(X, Y)

                w = clf.coef_.reshape(-1, 1)
                b = clf.intercept_[0]
                w_perp = w -  (w.transpose() @ optimal_predictor) / np.sum(optimal_predictor ** 2) * optimal_predictor
                
                L[i, j] = population_error(w, b, 
                                           sqrt_cov, w_star, b_star, dist = dist)
                Lhat[i, j] = np.mean(np.maximum(0, 1 - (X @ w + b) * Y.reshape(-1, 1)) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w_perp, ord = 1) / n) ** 2


    # store experimental results in a pandas dataframe
    df = pd.DataFrame()
    df['Population risk'] = L.flatten('F')
    df['Empirical risk'] = Lhat.flatten('F')
    df['Risk bound'] = bound.flatten('F')
    df['Bayes risk'] = optimal_risk
    df['Distribution'] = dist
    
    df.index = np.repeat(alphas, num_experiment)
    df.index.name = 'log regularization parameter'

    return df


def experiment(num_experiment, alphas, 
               n, sqrt_cov, w_star, b_star, dist = 'Gaussian', 
               method = 'l2', cov_split = 0):

    np.random.seed(0)
    _, optimal_risk = bayes(sqrt_cov, w_star, b_star, dist = dist)

    L = np.zeros((num_experiment, len(alphas)))
    Lhat = np.zeros((num_experiment, len(alphas)))
    bound = np.zeros((num_experiment, len(alphas)))

    for i in range(num_experiment):

        X, Y = data_model(n, sqrt_cov, w_star, b_star, dist = dist) 

        if method == 'l2':

            norm_x = np.sqrt(n * np.sum(sqrt_cov[cov_split:] ** 2))

            for j in range(len(alphas)):

                clf = LinearSVC(random_state=0, max_iter = 1000, 
                                penalty = 'l2', loss = 'squared_hinge',
                                dual = True, C = 1 / np.exp(alphas[j]))
                clf.fit(X, Y)

                w = clf.coef_.reshape(-1, 1)
                b = clf.intercept_[0]

                L[i, j] = population_error(w, b, 
                                           sqrt_cov, w_star, b_star, dist = dist)
                Lhat[i, j] = np.mean(np.maximum(0, 1 - (X @ w + b) * Y.reshape(-1, 1)) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w) / n) ** 2

        elif method == 'l1': 

            norm_x = np.zeros(100)
            for k in range(100):
                random_signs = (np.random.binomial(1, 0.5, size = (n, 1)) - 0.5) * 2
                norm_x[k] = np.linalg.norm(np.sum(X[:, cov_split:] * random_signs, axis = 0), ord = np.inf)
            norm_x = np.mean(norm_x)

            for j in range(len(alphas)):
                
                clf = LinearSVC(random_state=0, max_iter = 1000, 
                                penalty = 'l1', loss = 'squared_hinge',
                                dual = False, C = 1 / np.exp(alphas[j]))
                clf.fit(X, Y)

                w = clf.coef_.reshape(-1, 1)
                b = clf.intercept_[0]
                
                L[i, j] = population_error(w, b, 
                                           sqrt_cov, w_star, b_star, dist = dist)
                Lhat[i, j] = np.mean(np.maximum(0, 1 - (X @ w + b) * Y.reshape(-1, 1)) ** 2)
                bound[i, j] = (np.sqrt(Lhat[i, j]) + norm_x * np.linalg.norm(w, ord = 1) / n) ** 2


    # store experimental results in a pandas dataframe
    df = pd.DataFrame()
    df['Population risk'] = L.flatten('F')
    df['Empirical risk'] = Lhat.flatten('F')
    df['Risk bound'] = bound.flatten('F')
    df['Bayes risk'] = optimal_risk
    df['Distribution'] = dist
    
    df.index = np.repeat(alphas, num_experiment)
    df.index.name = 'log regularization parameter'

    return df