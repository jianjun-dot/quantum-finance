
import numpy as np
from package.option_pricing import OptionParams
from typing import Union
import json
from package.option_pricing import OptionPricing, OptionParams

class tester_params():
    def __init__(self):
        self.init_spot_prices = [2.0,2.25, 2.5, 2.75, 3]
        self.init_volatilities = [0.3, 0.35, 0.4, 0.45, 0.5]
        self.init_interest_rates = [0.03, 0.035, 0.04, 0.045, 0.05]
        self.init_maturities = [30,40,50,60]
        self.define_all_test_cases()
        
    def generate_covariance_matrix(self, first_var, second_var):
        A = np.random.randn(2,2)
        variances = [first_var, second_var]
        for i in range(2):
            original_var=  np.var(A[i,:])
            scaling_factor = np.sqrt(variances[i]/original_var)
            A[i,:] = A[i,:]*scaling_factor
            
        covariance_matrix = np.dot(A, A.T)
        return covariance_matrix
        
    def define_covariance_matrix(self, first_params, second_params):
        first_var = first_params['vol'] * np.sqrt(first_params['T']/365)
        second_var = second_params['vol'] * np.sqrt(second_params['T']/365)
        return self.generate_covariance_matrix(first_var, second_var)
    
    def define_all_test_cases(self):
        all_test_cases = []
        for spot_price in self.init_spot_prices:
            for volatility in self.init_volatilities:
                for interest_rate in self.init_interest_rates:
                    for maturity in self.init_maturities:
                        all_test_cases.append(
                            {
                                "S": spot_price,
                                "vol": volatility,
                                "r": interest_rate,
                                "T": maturity/365
                            }
                        )
        self.all_test_cases = all_test_cases
        return all_test_cases
    
    def define_test_strike_prices(self, option_type: str, option: OptionParams, num_test_cases: int) -> Union[list[int], list[list]]:
        if option_type == 'call':
            upper_bound = option.individual_params[0]['high']*0.9
            lower_bound = option.individual_params[0]['low']*1.1
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)
            
        elif option_type == 'spread call':
            first_high = option.individual_params[0]['high']
            second_high = option.individual_params[1]['high']
            upper_bound = np.min([first_high, second_high])*0.9
            lower_bound = 0
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)
            
        elif option_type == 'basket call':
            first_high = option.individual_params[0]['high']
            first_low = option.individual_params[0]['low']*1.1
            second_high = option.individual_params[1]['high']
            second_low = option.individual_params[1]['low']*1.1
            upper_bound = first_high + second_high
            lower_bound = first_low + second_low
            strike_prices = np.linspace(lower_bound, upper_bound * 0.9, num_test_cases)
        
        elif option_type == 'call-on-max':
            first_high = option.individual_params[0]['high']
            first_low = option.individual_params[0]['low']*1.1
            second_high = option.individual_params[1]['high']
            second_low = option.individual_params[1]['low']*1.1
            upper_bound = np.max([first_high, second_high])*0.9
            lower_bound = np.min([first_low, second_low])
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)
        
        elif option_type == 'call-on-min':
            first_high = option.individual_params[0]['high']
            first_low = option.individual_params[0]['low']*1.1
            second_high = option.individual_params[1]['high']
            second_low = option.individual_params[1]['low']*1.1
            upper_bound = np.min([first_high, second_high])
            lower_bound = np.min([first_low, second_low])
            strike_prices = np.linspace(lower_bound, upper_bound, num_test_cases)
        
        elif option_type == 'best-of-call':
            first_high = option.individual_params[0]['high']
            first_low = option.individual_params[0]['low']*1.1
            second_high = option.individual_params[1]['high']
            second_low = option.individual_params[1]['low']*1.1
            first_strike_prices = np.linspace(first_low, first_high, num_test_cases)
            second_strike_prices = np.linspace(second_low, second_high, num_test_cases)
            strike_prices = []
            for i in range(num_test_cases):
                strike_prices.append([first_strike_prices[i], second_strike_prices[i]])
    
        return strike_prices
            
    def define_random_test_cases(self, num_uncertainty_qubits, option_type, num_test_cases):
        if option_type in ['spread call', 'basket call', 'call-on-max', 'call-on-min']:
            test_cases = []
            random_combinations = [[np.random.choice(self.all_test_cases), np.random.choice(self.all_test_cases)] for _ in range(num_test_cases)]
            for selection in random_combinations:
                option = OptionParams(num_uncertainty_qubits=num_uncertainty_qubits, option_type=option_type)
                option.add_variable(selection[0])
                option.add_variable(selection[1])
                cov = self.define_covariance_matrix(selection[0], selection[1])
                option.define_covariance_matrix(cov)
                strike_prices = self.define_test_strike_prices(option_type, option, num_test_cases)
                chosen_strike_price = np.random.choice(strike_prices)
                option.define_strike_prices(chosen_strike_price)
                test_cases.append(option)
        
        elif option_type in ['call']:
            test_cases = []
            random_combinations = [np.random.choice(self.all_test_cases) for _ in range(num_test_cases)]
            for selection in random_combinations:
                option = OptionParams(num_uncertainty_qubits=num_uncertainty_qubits, option_type=option_type)
                option.add_variable(selection)
                chosen_strike_price = np.random.choice(self.define_test_strike_prices(option_type, option, num_test_cases))
                option.define_strike_prices(chosen_strike_price)
                test_cases.append(option)
        return test_cases
    
    def define_systematic_test_cases(self, num_uncertainty, option_type, num_test_cases):
        if option_type in ['spread call', 'basket call', 'call-on-max', 'call-on-min']:
            test_cases = []
            systematic_combinations = [[self.all_test_cases[i], self.all_test_cases[j]] for i in range(num_test_cases) for j in range(num_test_cases)]
            for selection in systematic_combinations:
                option = OptionParams(num_uncertainty_qubits=num_uncertainty, option_type=option_type)
                option.add_variable(selection[0])
                option.add_variable(selection[1])
                cov = self.define_covariance_matrix(selection[0], selection[1])
                option.define_covariance_matrix(cov)
                strike_prices = self.define_test_strike_prices(option_type, option, num_test_cases)
                for strike_price in strike_prices:
                    new_option = OptionParams(num_uncertainty_qubits=num_uncertainty, option_type=option_type)
                    new_option.add_variable(selection[0])
                    new_option.add_variable(selection[1])
                    new_option.define_covariance_matrix(cov)
                    new_option.define_strike_prices(strike_price)
                    test_cases.append(new_option)
        
        elif option_type in ['call']:
            test_cases = []
            systematic_combinations = [self.all_test_cases[i] for i in range(num_test_cases)]
            for selection in systematic_combinations:
                option = OptionParams(num_uncertainty_qubits=num_uncertainty, option_type=option_type)
                option.add_variable(selection)
                strike_prices = self.define_test_strike_prices(option_type, option, num_test_cases)
                for strike_price in strike_prices:
                    new_option = OptionParams(num_uncertainty_qubits=num_uncertainty, option_type=option_type)
                    new_option.add_variable(selection)
                    new_option.define_strike_prices(strike_price)
                    test_cases.append(new_option)
                    
        return test_cases
                
            

    
    

class Tester():
    def __init__(self, num_uncertainty_qubits):
        self.test_params = tester_params()
        self.num_uncertainty_qubits = num_uncertainty_qubits
        pass
    
    def test_call_options(self, num_test_cases, epsilon = 0.01, alpha=0.01, sample_size=1, file_name="vanilla_call_options"):
        test_cases = self.test_params.define_systematic_test_cases(self.num_uncertainty_qubits, "call", num_test_cases)
        test_results = {}
        for index, test_case in enumerate(test_cases):
            test_results["test {}".format(index)] = {}
            test_results["test {}".format(index)]["params"] = test_case.individual_params
            test_results["test {}".format(index)]["test_params"] = {
                "epsilon" : epsilon,
                "alpha" : alpha
            }
            myOptionPricer = OptionPricing(test_case)
            all_results = []
            for _ in range(sample_size):
                estimated_expectation, confidence_interval = myOptionPricer.estimate_expectation(epsilon=epsilon, alpha=alpha)
                exact_expectation = myOptionPricer.compute_exact_expectation()
                correct_confidence_interval = (confidence_interval[0] <= exact_expectation <= confidence_interval[1])
                curr_result = [exact_expectation, estimated_expectation, confidence_interval[0], confidence_interval[1], str(correct_confidence_interval)]
                all_results.append(curr_result)
            test_results["test {}".format(index)]["results"] = all_results
        
        print(test_results)
        with open('{}.json'.format(file_name), 'w') as f:
            json.dump(test_results, f)

    def test_basket_call_options(self, num_test_cases, epsilon = 0.01, alpha=0.01, sample_size=1, file_name="basket_call_options_test_results"):
        test_cases = self.test_params.define_systematic_test_cases(self.num_uncertainty_qubits, "basket call", num_test_cases)
        test_results = {}
        for index, test_case in enumerate(test_cases):
            test_results["test {}".format(index)] = {}
            test_results["test {}".format(index)]["params"] = test_case.individual_params
            curr_cov = []
            for i in range(2):
                    curr_cov.append([test_case.cov[i,0],test_case.cov[i,1]])  
            test_results["test {}".format(index)]["covariance"] = curr_cov
            test_results["test {}".format(index)]["test_params"] = {
                "epsilon" : epsilon,
                "alpha" : alpha
            }
            myOptionPricer = OptionPricing(test_case)
            all_results = []
            for _ in range(sample_size):
                estimated_expectation, confidence_interval = myOptionPricer.estimate_expectation(epsilon=epsilon, alpha=alpha)
                exact_expectation = myOptionPricer.compute_exact_expectation()
                correct_confidence_interval = (confidence_interval[0] <= exact_expectation <= confidence_interval[1])
                curr_result = [exact_expectation, estimated_expectation, confidence_interval[0], confidence_interval[1], str(correct_confidence_interval)]
                all_results.append(curr_result)
            test_results["test {}".format(index)]["results"] = all_results
        
        print(test_results)
        with open('{}.json'.format(file_name), 'w') as f:
            json.dump(test_results, f)

    def test_spread_call_options(self, num_test_cases, epsilon = 0.01, alpha=0.01, sample_size=1, file_name="spread_call_options_test_results"):
        test_cases = self.test_params.define_systematic_test_cases(self.num_uncertainty_qubits, "spread call", num_test_cases)
        test_results = {}
        for index, test_case in enumerate(test_cases):
            test_results["test {}".format(index)] = {}
            test_results["test {}".format(index)]["params"] = test_case.individual_params
            curr_cov = []
            for i in range(2):
                    curr_cov.append([test_case.cov[i,0],test_case.cov[i,1]])  
            test_results["test {}".format(index)]["covariance"] = curr_cov
            test_results["test {}".format(index)]["test_params"] = {
                "epsilon" : epsilon,
                "alpha" : alpha
            }
            myOptionPricer = OptionPricing(test_case)
            all_results = []
            for _ in range(sample_size):
                estimated_expectation, confidence_interval = myOptionPricer.estimate_expectation(epsilon=epsilon, alpha=alpha)
                exact_expectation = myOptionPricer.compute_exact_expectation()
                correct_confidence_interval = (confidence_interval[0] <= exact_expectation <= confidence_interval[1])
                curr_result = [exact_expectation, estimated_expectation, confidence_interval[0], confidence_interval[1], str(correct_confidence_interval)]
                all_results.append(curr_result)
            test_results["test {}".format(index)]["results"] = all_results
        
        print(test_results)
        with open('{}.json'.format(file_name), 'w') as f:
            json.dump(test_results, f)

    def test_call_on_min_options(self, num_test_cases, epsilon = 0.01, alpha=0.01, sample_size=1, file_name="call_on_min_options_test_results"):
        test_cases = self.test_params.define_systematic_test_cases(self.num_uncertainty_qubits, "call-on-min", num_test_cases)
        test_results = {}
        for index, test_case in enumerate(test_cases):
            test_results["test {}".format(index)] = {}
            test_results["test {}".format(index)]["params"] = test_case.individual_params
            curr_cov = []
            for i in range(2):
                    curr_cov.append([test_case.cov[i,0],test_case.cov[i,1]])  
            test_results["test {}".format(index)]["covariance"] = curr_cov
            test_results["test {}".format(index)]["test_params"] = {
                "epsilon" : epsilon,
                "alpha" : alpha
            }
            myOptionPricer = OptionPricing(test_case)
            all_results = []
            for _ in range(sample_size):
                estimated_expectation, confidence_interval = myOptionPricer.estimate_expectation(epsilon=epsilon, alpha=alpha)
                exact_expectation = myOptionPricer.compute_exact_expectation()
                correct_confidence_interval = (confidence_interval[0] <= exact_expectation <= confidence_interval[1])
                curr_result = [exact_expectation, estimated_expectation, confidence_interval[0], confidence_interval[1], str(correct_confidence_interval)]
                all_results.append(curr_result)
            test_results["test {}".format(index)]["results"] = all_results
        
        print(test_results)
        with open('{}.json'.format(file_name), 'w') as f:
            json.dump(test_results, f)

    def test_call_on_max_options(self, num_test_cases, epsilon = 0.01, alpha=0.01, sample_size=1, file_name="call_on_max_options_test_results"):
        test_cases = self.test_params.define_systematic_test_cases(self.num_uncertainty_qubits, "call-on-max", num_test_cases)
        test_results = {}
        for index, test_case in enumerate(test_cases):
            test_results["test {}".format(index)] = {}
            test_results["test {}".format(index)]["params"] = test_case.individual_params
            curr_cov = []
            for i in range(2):
                    curr_cov.append([test_case.cov[i,0],test_case.cov[i,1]])  
            test_results["test {}".format(index)]["covariance"] = curr_cov
            test_results["test {}".format(index)]["test_params"] = {
                "epsilon" : epsilon,
                "alpha" : alpha
            }
            myOptionPricer = OptionPricing(test_case)
            all_results = []
            for _ in range(sample_size):
                estimated_expectation, confidence_interval = myOptionPricer.estimate_expectation(epsilon=epsilon, alpha=alpha)
                exact_expectation = myOptionPricer.compute_exact_expectation()
                correct_confidence_interval = (confidence_interval[0] <= exact_expectation <= confidence_interval[1])
                curr_result = [exact_expectation, estimated_expectation, confidence_interval[0], confidence_interval[1], str(correct_confidence_interval)]
                all_results.append(curr_result)
            test_results["test {}".format(index)]["results"] = all_results
        
        print(test_results)
        with open('{}.json'.format(file_name), 'w') as f:
            json.dump(test_results, f)

    def test_best_of_call_options(self):
        pass
    
    def run_test(self, num_uncertainty_qubits, num_tests):
        pass
    
        
    
        
        