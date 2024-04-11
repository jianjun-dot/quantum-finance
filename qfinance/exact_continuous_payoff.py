import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm
from typing import Union

class ContinuousExactPayoff():
    def __init__(self):
        pass
    
    def compute_expectation(self, option_type: str, asset_params: Union[list, dict], strike_price: Union[list, float], n_samples=1000000):
        if option_type == "vanilla-call":
            return self._compute_vanilla_call_expectation(asset_params, strike_price, n_samples)
        elif option_type == "spread-call":
            return self._compute_spread_call_expectation(asset_params, strike_price, n_samples)
        elif option_type == "basket-call":
            return self._compute_basket_call_expectation(asset_params, strike_price, n_samples)
        elif option_type == "call-on-max":
            return self._compute_call_on_max_expectation(asset_params, strike_price, n_samples)
        elif option_type =="call-on-min":
            return self._compute_call_on_min_expectation(asset_params, strike_price, n_samples)
        elif option_type == "best-of-call":
            assert type(strike_price) == list and len(strike_price) == 2
            return self._compute_best_of_call_expectation(asset_params, strike_price[0], strike_price[1], n_samples)
        else:
            raise ValueError("Option type not supported")
        
    def _compute_vanilla_call_expectation(self, asset_params: dict, strike_price: float)-> float:
        # unpack params
        S, vol, r, T = asset_params['S'], asset_params['vol'], asset_params['r'], asset_params['T']
        
        # lognormal params
        mu = (r - 0.5 * vol**2) * T + np.log(S)
        sigma = vol * np.sqrt(T)
        
        def payoff_function(S, r, T):
            return np.exp(-r * T) * max(S- strike_price, 0)
        
        scale = np.exp(mu)
        expected_value, _ = quad(lambda S_T: payoff_function(S_T, r, T) * lognorm.pdf(S_T, s=sigma, scale=scale), strike_price, np.inf)
        return expected_value
    
    def _compute_basket_call_expectation(self, asset_params: dict, strike_price: float, n_samples=1000000) -> float:
        asset1_params = asset_params[0]
        asset2_params = asset_params[1]
        cov = asset_params[3]
        S1, vol1, r1, T1 = asset1_params['S'], asset1_params['vol'], asset1_params['r'], asset1_params['T']
        S2, vol2, r2, T2 = asset2_params['S'], asset2_params['vol'], asset2_params['r'], asset2_params['T']
        
        # Ensure same r and T for both assets (for simplicity)
        r, T = r1, T1
        
        # Generate random numbers with the given covariance
        z = np.random.multivariate_normal([0, 0], cov, size=n_samples)
        
        # Simulate future values for both assets
        x1_T = S1 * np.exp((r - 0.5 * vol1**2) * T + vol1 * np.sqrt(T) * z[:, 0])
        x2_T = S2 * np.exp((r - 0.5 * vol2**2) * T + vol2 * np.sqrt(T) * z[:, 1])
        
        # Compute payoffs for each simulation
        payoffs = np.maximum(x1_T + x2_T - strike_price, 0)
        
        # Compute the expected payoff and discount it back to present value
        expected_value = np.mean(payoffs) * np.exp(-r * T)
        
        return expected_value
        
    def _compute_call_on_max_expectation(self, asset_params: dict, strike_price: float, n_samples= 1000000) -> float:
        asset1_params = asset_params[0]
        asset2_params = asset_params[1]
        cov = asset_params[3]
        S1, vol1, r1, T1 = asset1_params['S'], asset1_params['vol'], asset1_params['r'], asset1_params['T']
        S2, vol2, r2, T2 = asset2_params['S'], asset2_params['vol'], asset2_params['r'], asset2_params['T']
        
        # Ensure same r and T for both assets (for simplicity)
        r, T = r1, T1
        
        # Generate random numbers with the given covariance
        z = np.random.multivariate_normal([0, 0], cov, size=n_samples)
        
        # Simulate future values for both assets
        x1_T = S1 * np.exp((r - 0.5 * vol1**2) * T + vol1 * np.sqrt(T) * z[:, 0])
        x2_T = S2 * np.exp((r - 0.5 * vol2**2) * T + vol2 * np.sqrt(T) * z[:, 1])
        
        # Compute payoffs for each simulation
        payoffs = np.maximum(np.maximum(x1_T, x2_T) - strike_price, 0)
        
        # Compute the expected payoff and discount it back to present value
        expected_value = np.mean(payoffs) * np.exp(-r * T)
        
        return expected_value
    
    def _compute_call_on_min_expectation(self, asset_params: dict, strike_price: float, n_samples= 1000000) -> float:
        asset1_params = asset_params[0]
        asset2_params = asset_params[1]
        cov = asset_params[3]
        
        S1, vol1, r1, T1 = asset1_params['S'], asset1_params['vol'], asset1_params['r'], asset1_params['T']
        S2, vol2, r2, T2 = asset2_params['S'], asset2_params['vol'], asset2_params['r'], asset2_params['T']
        
        # Ensure same r and T for both assets (for simplicity)
        r, T = r1, T1
        
        # Generate random numbers with the given covariance
        z = np.random.multivariate_normal([0, 0], cov, size=n_samples)
        
        # Simulate future values for both assets
        x1_T = S1 * np.exp((r - 0.5 * vol1**2) * T + vol1 * np.sqrt(T) * z[:, 0])
        x2_T = S2 * np.exp((r - 0.5 * vol2**2) * T + vol2 * np.sqrt(T) * z[:, 1])
        
        # Compute payoffs for each simulation
        payoffs = np.maximum(np.minimum(x1_T, x2_T) - strike_price, 0)
        
        # Compute the expected payoff and discount it back to present value
        expected_value = np.mean(payoffs) * np.exp(-r * T)
        
        return expected_value
        
    def _compute_spread_call_expectation(self, asset_params: dict, strike_price: float, n_samples=1000000) -> float:
        asset1_params = asset_params[0]
        asset2_params = asset_params[1]
        cov = asset_params[3]
        
        S1, vol1, r1, T1 = asset1_params['S'], asset1_params['vol'], asset1_params['r'], asset1_params['T']
        S2, vol2, r2, T2 = asset2_params['S'], asset2_params['vol'], asset2_params['r'], asset2_params['T']
        
        # Ensure same r and T for both assets (for simplicity)
        r, T = r1, T1
        
        # Generate random numbers with the given covariance
        z = np.random.multivariate_normal([0, 0], cov, size=n_samples)
        
        # Simulate future values for both assets
        x1_T = S1 * np.exp((r - 0.5 * vol1**2) * T + vol1 * np.sqrt(T) * z[:, 0])
        x2_T = S2 * np.exp((r - 0.5 * vol2**2) * T + vol2 * np.sqrt(T) * z[:, 1])
        
        # Compute payoffs for each simulation
        payoffs = np.maximum(x1_T - x2_T - strike_price, 0)
        
        # Compute the expected payoff and discount it back to present value
        expected_value = np.mean(payoffs) * np.exp(-r * T)
        
        return expected_value
    
    def _compute_best_of_call_expectation(self, asset_params: dict, strike_price1: float, strike_price2: float, n_samples=1000000) -> float:
        asset1_params = asset_params[0]
        asset2_params = asset_params[1]
        cov = asset_params[3]
        
        S1, vol1, r1, T1 = asset1_params['S'], asset1_params['vol'], asset1_params['r'], asset1_params['T']
        S2, vol2, r2, T2 = asset2_params['S'], asset2_params['vol'], asset2_params['r'], asset2_params['T']
        
        # Ensure same r and T for both assets (for simplicity)
        r, T = r1, T1
        
        # Generate random numbers with the given covariance
        z = np.random.multivariate_normal([0, 0], cov, size=n_samples)
        
        # Simulate future values for both assets
        x1_T = S1 * np.exp((r - 0.5 * vol1**2) * T + vol1 * np.sqrt(T) * z[:, 0])
        x2_T = S2 * np.exp((r - 0.5 * vol2**2) * T + vol2 * np.sqrt(T) * z[:, 1])
        
        # Compute payoffs for each simulation
        payoffs = np.maximum(np.maximum(x1_T - strike_price1, x2_T -strike_price2), 0)
        
        # Compute the expected payoff and discount it back to present value
        expected_value = np.mean(payoffs) * np.exp(-r * T)
        
        return expected_value