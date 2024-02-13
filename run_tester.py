from tester import Tester


def main():
    
    num_uncertainty_qubits = 3
    tester = Tester(num_uncertainty_qubits=num_uncertainty_qubits)
    # tester.test_call_options(num_test_cases=2, sample_size=3)
    # tester.test_basket_call_options(num_test_cases=2, sample_size=3)
    # tester.test_call_on_max_options(num_test_cases=2, sample_size=3)
    tester.test_call_on_min_options(num_test_cases=2, sample_size=3)


if __name__ == "__main__":
    main()