import numpy as np
from qfinance.option_pricing import OptionParams
from typing import Union, List
import json
from qfinance.option_pricing import OptionPricing, OptionParams
from datetime import datetime


class TestParamsGenerator:
    def __init__(self):
        self.init_spot_prices = [round(x, 2) for x in np.linspace(1.5, 2.5, 10)]
        self.init_volatilities = [round(x, 2) for x in np.linspace(0.3, 0.5, 10)]
        self.init_interest_rates = [round(x, 3) for x in np.linspace(0.03, 0.05)]
        self.init_maturities = [round(x, 0) for x in np.linspace(30, 50, 10)]

    def set_spot_price_range(self, low: float, high: float, n_steps: int)-> None:
        self.init_spot_prices = [round(x, 2) for x in np.linspace(low, high, n_steps)]
        
    def set_volatility_range(self, low: float, high: float, n_steps: int)-> None:
        self.init_volatilities = [round(x, 2) for x in np.linspace(low, high, n_steps)]
        
    def set_interest_rate_range(self, low: float, high: float, n_steps: int)-> None:
        self.init_interest_rates = [round(x, 3) for x in np.linspace(low, high, n_steps)]
        
    def set_maturity_range(self, low: float, high: float, n_steps: int)-> None:
        self.init_maturities = [round(x, 0) for x in np.linspace(low, high, n_steps)]
        
    def generate_random_correlation(self):
        rng = np.random.default_rng()
        return round(rng.uniform(-0.9, 0.9), 2)

    def define_test_strike_price_range(
        self, option_type: str, option: OptionParams, num_test_cases: int
    ) -> Union[list[int], list[list]]:
        if option_type == "call":
            upper_bound = option.individual_params[0]["high"] * 0.9
            lower_bound = option.individual_params[0]["low"] * 1.1
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)

        elif option_type == "spread call":
            first_high = option.individual_params[0]["high"]
            second_high = option.individual_params[1]["high"]
            upper_bound = np.min([first_high, second_high]) * 0.9
            lower_bound = 0
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)

        elif option_type == "basket call":
            first_high = option.individual_params[0]["high"]
            first_low = option.individual_params[0]["low"] * 1.1
            second_high = option.individual_params[1]["high"]
            second_low = option.individual_params[1]["low"] * 1.1
            upper_bound = first_high + second_high
            lower_bound = first_low + second_low
            strike_prices = np.linspace(lower_bound, upper_bound * 0.9, num_test_cases)

        elif option_type == "call-on-max":
            first_high = option.individual_params[0]["high"]
            first_low = option.individual_params[0]["low"] * 1.1
            second_high = option.individual_params[1]["high"]
            second_low = option.individual_params[1]["low"] * 1.1
            upper_bound = np.max([first_high, second_high]) * 0.9
            lower_bound = np.min([first_low, second_low])
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)

        elif option_type == "call-on-min":
            first_high = option.individual_params[0]["high"]
            first_low = option.individual_params[0]["low"] * 1.1
            second_high = option.individual_params[1]["high"]
            second_low = option.individual_params[1]["low"] * 1.1
            upper_bound = np.min([first_high, second_high])
            lower_bound = np.min([first_low, second_low])
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)

        elif option_type == "best-of-call":
            first_high = option.individual_params[0]["high"]
            first_low = option.individual_params[0]["low"] * 1.1
            second_high = option.individual_params[1]["high"]
            second_low = option.individual_params[1]["low"] * 1.1
            first_strike_prices = np.linspace(first_low, first_high, num_test_cases)
            second_strike_prices = np.linspace(second_low, second_high, num_test_cases)
            strike_prices = []
            for i in range(num_test_cases):
                strike_prices.append([first_strike_prices[i], second_strike_prices[i]])

        return strike_prices
    
    def generate_random_params(self)-> dict:
        return {
            "S": np.random.choice(self.init_spot_prices),
            "vol": np.random.choice(self.init_volatilities),
            "r": np.random.choice(self.init_interest_rates),
            "T": np.random.choice(self.init_maturities),
        }

    def create_random_test_cases(
        self, num_uncertainty_qubits: int, option_type: str, num_test_cases: int
    ) -> List[OptionParams]:
        if option_type in ["spread call", "basket call", "call-on-max", "call-on-min"]:
            test_cases = []
            random_combinations = [
                [
                    self.generate_random_params(),
                    self.generate_random_params(),
                ]
                for _ in range(num_test_cases)
            ]
            for selection in random_combinations:
                option = OptionParams(
                    num_uncertainty_qubits=num_uncertainty_qubits,
                    option_type=option_type,
                )
                option.add_variable(selection[0])
                option.add_variable(selection[1])
                option.set_covariance_matrix(self.generate_random_correlation())
                strike_prices = self.define_test_strike_price_range(
                    option_type, option, num_test_cases
                )
                chosen_strike_price = np.random.choice(strike_prices)
                option.set_strike_prices(chosen_strike_price)
                test_cases.append(option)

        elif option_type in ["vanilla call"]:
            test_cases = []
            random_combinations = [
                self.generate_random_params() for _ in range(num_test_cases)
            ]
            for selection in random_combinations:
                option = OptionParams(
                    num_uncertainty_qubits=num_uncertainty_qubits,
                    option_type=option_type,
                )
                option.add_variable(selection)
                chosen_strike_price = np.random.choice(
                    self.define_test_strike_price_range(option_type, option, num_test_cases)
                )
                option.set_strike_prices(chosen_strike_price)
                test_cases.append(option)
        return test_cases


class Tester:
    def __init__(self, num_uncertainty_qubits):
        self.test_params = TestParamsGenerator()
        self.num_uncertainty_qubits = num_uncertainty_qubits
    
    def run_single_test(self):
        pass
    
