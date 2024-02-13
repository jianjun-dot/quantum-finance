from tester import Tester
import numpy as np


def main():
    
    num_uncertainty_qubits = 3
    tester = Tester(num_uncertainty_qubits=num_uncertainty_qubits)
    # tester.test_call_options(num_test_cases=3, sample_size=10)
    epsilon_range = np.linspace(0.001, 0.01, 10)
    for epsilon in epsilon_range:
        tester.test_call_options(num_test_cases=3, sample_size=5, epsilon=epsilon)
    # tester.test_basket_call_options(num_test_cases=3, sample_size=10)
    # tester.test_call_on_max_options(num_test_cases=3, sample_size=10)
    # tester.test_call_on_min_options(num_test_cases=3, sample_size=10)


if __name__ == "__main__":
    main()