import numpy as np
import scipy.optimize as opt

def negative_sharpe(weights, mean, cov, rf):
    portfolio_return = np.dot(weights, mean) - rf
        
    # total volatility = sqrt(portfolio return - risk-free rate)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    
    # if we have zero volatility, return infinity
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else -np.inf
    return -1 * sharpe


def maximize_sharpe(data, risk_free_rate=0.0, max_iter=1000, tol=1e-8):
    
    n_assets = data.shape[1]
    
    # mean and covariance matrix
    mean_returns = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    
    # initial point is all equal weighting
    initial_weights = np.ones(n_assets) / n_assets
    
    # constraints are that the weights sum to 1 and weights are non-negative
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # optimize
    result = opt.minimize(
        fun=negative_sharpe,
        x0=initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'ftol': tol, 'maxiter': max_iter}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")
    
    return result


def optimize_portfolio(data, model,risk_free_rate=0.0, max_iter=1000, tol=1e-8):

    # initialize weights and values to 0
    weights=[]
    sharpe_values = []

    # loop through each day
    for i in range(data.shape[0]):

        # make prediction
        np.append(data[i][:,:4], model.predict((data[i])[np.newaxis,:]) ,axis=0)

        # get returns for each asset
        returns =(data[i][:,:4][1:] - data[i][:,:4][:-1]) / data[i][:,:4][:-1]

        # optimize portfolio
        solver = maximize_sharpe(returns, risk_free_rate, max_iter, tol)

        # check for convergence
        if solver.success:
            weights.append(solver.x)
            sharpe_values.append(solver.fun)

    return weights, sharpe_values