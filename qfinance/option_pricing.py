import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Gate
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import (
    LinearAmplitudeFunction,
    DraperQFTAdder,
)
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution
from qfinance.ModifiedIQAE.mod_iae_updated import (
    ModifiedIterativeAmplitudeEstimation,
)
from qfinance.ModifiedIQAE.amplitude_estimator import AmplitudeEstimatorResult
from typing import List, Union, Tuple, Callable
from .qArithmetic import QComp, subtractorDraper
from .helper import loadNumber
from .helper import define_covariance_matrix


############
# option input should be a JSON object
###########
"""
option_params = {
    'option_type'[str]: call, call, basket call, spread call, call-on-max, call-on-min, best-of-call
    # this defines the type of option that we will be pricing
    'option_params'[dict]:{
        'r'[float]: 0.04, # annual interest rate of 4%
        'vol'[float]: 0.4, # volatility of 40%
        'T'[float]: 40/365, # 40 days to maturity
        'S'[float]: 0.5, # initial spot price
        'strike_price'[float]: 0.01, # strike price
    }
}
"""


class OptionParams:
    def __init__(self, num_uncertainty_qubits: int, option_type):
        self.individual_params = []
        self.cov = None
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.strike_prices = None
        self.option_type = option_type

    def __str__(self):
        # enumerate all self variables
        return str(
            {
                "individual_params": self.individual_params,
                "cov": self.cov,
                "num_uncertainty_qubits": self.num_uncertainty_qubits,
                "strike_prices": self.strike_prices,
                "option_type": self.option_type,
            }
        )

    def set_strike_prices(self, strike_prices: Union[List[float], float]):
        if isinstance(strike_prices, float):
            self.strike_prices = [strike_prices]
        else:
            self.strike_prices = strike_prices

    def set_covariance_matrix(self, correlation: float):
        variables = self.individual_params
        if len(variables) == 1:
            raise Exception("Covariance matrix not needed for single variable!")
        else:
            sigma_1 = variables[0]["sigma"]
            sigma_2 = variables[1]["sigma"]
            cov = define_covariance_matrix(sigma_1, sigma_2, correlation)
            self.cov = cov

    def add_variable(self, variable: Union[list, dict]):
        if type(variable) is dict:
            variable = self._process_variable(variable)
            self.individual_params.append(variable)
        elif type(variable) is list:
            for var in variable:
                curr_variable = self._process_variable(var)
                self.individual_params.append(curr_variable)

    def _process_variable(self, variable: dict):
        mu = (variable["r"] - 0.5 * variable["vol"] ** 2) * variable["T"] + np.log(
            variable["S"]
        )
        sigma = variable["vol"] * np.sqrt(variable["T"])
        mean = np.exp(mu + sigma**2 / 2)
        std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
        if self.option_type == "spread call":
            low = 0
        else:
            low = np.maximum(0, mean - 3 * std)
        high = mean + 3 * std
        variable["mu"] = mu
        variable["sigma"] = sigma
        variable["mean"] = mean
        variable["std"] = std
        variable["low"] = low
        variable["high"] = high
        return variable


class OptionPricing:

    def __init__(self, options_params: OptionParams) -> None:
        self.num_variables = len(options_params.individual_params)
        self.num_uncertainty_qubits = options_params.num_uncertainty_qubits
        self.options_params = options_params
        self.objective_index = None
        self.option = None
        self.objective = None
        self.strike_price = options_params.strike_prices
        self.discount_factor = np.exp(-options_params.individual_params[0]["r"]*options_params.individual_params[0]["T"])
        self.define_uncertainty_model(options_params)
        # self._define_payoff_function(options_params.option_type, options_params.strike_prices)
        self.option_type = options_params.option_type

    def define_uncertainty_model(self, option_params: OptionParams) -> None:
        self.all_variables = option_params.individual_params
        self.dimension = len(self.all_variables)
        if self.dimension == 1:
            lower_bound = self.all_variables[0]["low"]
            upper_bound = self.all_variables[0]["high"]
            mu = self.all_variables[0]["mu"]
            std = self.all_variables[0]["std"]
            self.uncertainty_model = LogNormalDistribution(
                num_qubits=self.num_uncertainty_qubits,
                mu=mu,
                sigma=std**2,
                bounds=(lower_bound, upper_bound),
            )
        else:
            self.num_dist_qubits = [self.num_uncertainty_qubits] * self.dimension
            lower_bound = np.array([var["low"] for var in self.all_variables])
            upper_bound = np.array([var["high"] for var in self.all_variables])
            lower_bound = [np.min(lower_bound)] * self.dimension
            upper_bound = [np.max(upper_bound)] * self.dimension
            mu = np.array([var["mu"] for var in self.all_variables])
            if option_params.cov is None:
                cov = np.diag([var["sigma"] ** 2 for var in self.all_variables])
            else:
                cov = option_params.cov
            self.uncertainty_model = LogNormalDistribution(
                num_qubits=self.num_dist_qubits,
                mu=mu,
                sigma=cov,
                bounds=list(zip(lower_bound, upper_bound)),
            )

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def _define_payoff_function(
        self, option_type: str, strike_price: list[float], c_approx=0.125
    ) -> None:
        if option_type == "call":
            strike_price = strike_price[0]
            self._define_basic_call_options(strike_price, c_approx)
        elif option_type == "basket call":
            strike_price = strike_price[0]
            self._define_basket_call_options(strike_price, c_approx)
        elif option_type == "spread call":
            strike_price = strike_price[0]
            self._define_spread_call_options(strike_price, c_approx)
        elif option_type == "call-on-max":
            strike_price = strike_price[0]
            self._define_call_on_max_options(strike_price, c_approx)
        elif option_type == "call-on-min":
            strike_price = strike_price[0]
            self._define_call_on_min_options(strike_price, c_approx)
        elif option_type == "best-of-call":
            self._define_best_of_call_options(strike_price, c_approx)
        else:
            raise Exception("Option type not defined!")

    def _define_basic_call_options(self, strike_price: float, c_approx=0.1) -> None:
        params = self.options_params.individual_params[0]

        self.high = params["high"]
        self.low = params["low"]

        breakpoints = [params["low"], strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = params["high"] - strike_price

        call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(params["low"], params["high"]),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        self.strike_price = strike_price

        num_qubits = call_objective.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.append(self.uncertainty_model, range(self.uncertainty_model.num_qubits))
        circuit.append(call_objective, range(call_objective.num_qubits))

        self.objective = call_objective
        self.objective_index = self.uncertainty_model.num_qubits
        self.option = circuit
        self.post_processor = self.objective.post_processing

    def _define_basket_call_options(self, strike_price: float, c_approx=0.1) -> None:
        self.low = self.lower_bound[0]
        self.high = self.upper_bound[0]
        adder = DraperQFTAdder(self.num_uncertainty_qubits, kind="half", name="Adder")
        self.strike_price = strike_price
        step_ = (self.high - self.low) / (2 ** (self.num_uncertainty_qubits) - 1)

        # setup piecewise linear objective fcuntion
        breakpoints = [2 * self.low, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]

        f_min = 0
        f_max = 2 * self.high - strike_price + step_
        # print("payoff function range: {}".format([0, f_max]))

        basket_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits + 1,
            slopes,
            offsets,
            domain=(2 * self.low, 2 * self.high + step_),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
            breakpoints=breakpoints,
        )
        firstVar = QuantumRegister(self.num_uncertainty_qubits, "first")
        secondVar = QuantumRegister(self.num_uncertainty_qubits, "second")
        adder_carry = QuantumRegister(1, "adder_carry")
        objective = QuantumRegister(1, "objective")
        obj_ancilla = AncillaRegister(basket_objective.num_ancillas, "ancilla")

        objective_index = self.uncertainty_model.num_qubits + 1

        basket_option = QuantumCircuit(
            firstVar, secondVar, adder_carry, objective, obj_ancilla
        )
        basket_option.append(self.uncertainty_model, firstVar[:] + secondVar[:])
        basket_option.append(adder, firstVar[:] + secondVar[:] + [adder_carry[0]])
        basket_option.append(
            basket_objective,
            secondVar[:] + [adder_carry[0]] + [objective[0]] + obj_ancilla[:],
        )

        self.objective = basket_objective
        self.objective_index = objective_index
        self.option = basket_option
        self.post_processor = self.objective.post_processing

    def _define_spread_call_options(self, strike_price: float, c_approx=0.01) -> None:

        params = self.options_params.individual_params[0]
        num_qubits_for_each_dimension = self.num_uncertainty_qubits + 1
        # low_ = params['low']
        high_ = params["high"]
        self.strike_price = strike_price
        step = high_ / (2**self.num_uncertainty_qubits - 1)

        # setup piecewise linear objective fcuntion
        breakpoints = [-(2**self.num_uncertainty_qubits) * step, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]

        f_min = 0
        f_max = high_ - strike_price
        spread_objective = LinearAmplitudeFunction(
            num_qubits_for_each_dimension,
            slopes,
            offsets,
            domain=(-(2**self.num_uncertainty_qubits) * step, high_),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
            breakpoints=breakpoints,
        )

        firstRegister = QuantumRegister(self.num_uncertainty_qubits, "first")
        secondRegister = QuantumRegister(self.num_uncertainty_qubits, "second")
        objectiveRegister = QuantumRegister(1, "obj")
        carryRegister = QuantumRegister(1, "carry")
        optionAncillaRegister = AncillaRegister(
            spread_objective.num_ancillas, "optionAncilla"
        )

        subtract_circuit = subtractorDraper(self.num_uncertainty_qubits)
        # subtract_circuit = subtractorVBE(num_qubits_for_each_dimension, num_qubits_for_each_dimension)

        spread_option = QuantumCircuit(
            carryRegister,
            firstRegister,
            secondRegister,
            objectiveRegister,
            optionAncillaRegister,
        )
        spread_option.append(
            self.uncertainty_model, firstRegister[:] + secondRegister[:]
        )
        spread_option.append(
            subtract_circuit, firstRegister[:] + secondRegister[:] + carryRegister[:]
        )
        spread_option.x(carryRegister[:])

        spread_option.append(
            spread_objective,
            secondRegister[:]
            + carryRegister[:]
            + objectiveRegister[:]
            + optionAncillaRegister[:],
        )
        objective_index = self.num_uncertainty_qubits * 2 + 1

        self.objective = spread_objective
        self.objective_index = objective_index
        self.option = spread_option
        self.post_processor = self.objective.post_processing

    def _define_call_on_max_options(self, strike_price: float, c_approx=0.1) -> None:
        # params = self.options_params.individual_params[0]
        self.high = self.upper_bound[0]
        self.low = self.lower_bound[0]
        self.strike_price = strike_price

        breakpoints = [self.low, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = self.high - strike_price

        european_call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(self.low, self.high),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        call_objective = european_call_objective.to_gate()
        controlled_objective = call_objective.control(1)

        bit_length = self.num_uncertainty_qubits + 1

        carry_register = QuantumRegister(1, name="c")
        second_carry_register = QuantumRegister(1, name="c'")
        first_number_register = QuantumRegister(bit_length, name="a")
        second_number_register = QuantumRegister(bit_length, name="b")
        objective_register = QuantumRegister(1, name="objective")
        ancilla_register = QuantumRegister(3, name="ancilla")
        second_ancilla_register = QuantumRegister(
            self.num_uncertainty_qubits, name="objective_ancilla"
        )

        adder = DraperQFTAdder(bit_length, kind="half")
        circuit = QuantumCircuit(
            first_number_register,
            second_number_register,
            carry_register,
            second_carry_register,
            objective_register,
            ancilla_register,
            second_ancilla_register,
        )
        circuit.append(
            self.uncertainty_model,
            first_number_register[:-1] + second_number_register[:-1],
        )
        qcomp = QComp(bit_length, bit_length)
        circuit.append(
            qcomp,
            first_number_register[:]
            + second_number_register[:]
            + carry_register[:]
            + ancilla_register[:],
        )

        circuit.append(
            adder,
            first_number_register[:]
            + second_number_register[:]
            + second_carry_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[0]]
            + second_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[1]]
            + first_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[2]]
            + first_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )

        self.objective = european_call_objective
        self.objective_index = self.uncertainty_model.num_qubits + 4
        self.option = circuit
        self.post_processor = self.objective.post_processing

    def _define_call_on_min_options(self, strike_price: float, c_approx=0.1) -> None:

        self.high = self.upper_bound[0]
        self.low = self.lower_bound[0]
        self.strike_price = strike_price

        breakpoints = [self.low, strike_price]
        slopes = [-1, 0]
        offsets = [strike_price - self.low, 0]
        f_min = 0
        f_max = strike_price - self.low

        european_call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(self.low, self.high),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        call_objective = european_call_objective.to_gate()
        controlled_objective = call_objective.control(1)

        bit_length = self.num_uncertainty_qubits + 1

        carry_register = QuantumRegister(1, name="c")
        second_carry_register = QuantumRegister(1, name="c'")
        first_number_register = QuantumRegister(bit_length, name="a")
        second_number_register = QuantumRegister(bit_length, name="b")
        objective_register = QuantumRegister(1, name="objective")
        ancilla_register = QuantumRegister(3, name="ancilla")
        second_ancilla_register = QuantumRegister(
            self.num_uncertainty_qubits, name="objective_ancilla"
        )

        adder = DraperQFTAdder(bit_length, kind="half")
        circuit = QuantumCircuit(
            first_number_register,
            second_number_register,
            carry_register,
            second_carry_register,
            objective_register,
            ancilla_register,
            second_ancilla_register,
        )
        circuit.append(
            self.uncertainty_model,
            first_number_register[:-1] + second_number_register[:-1],
        )
        qcomp = QComp(bit_length, bit_length)
        circuit.append(
            qcomp,
            first_number_register[:]
            + second_number_register[:]
            + carry_register[:]
            + ancilla_register[:],
        )
        circuit.append(
            adder,
            first_number_register[:]
            + second_number_register[:]
            + second_carry_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[0]]
            + first_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[1]]
            + second_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )
        circuit.append(
            controlled_objective,
            [ancilla_register[2]]
            + second_number_register[:-1]
            + objective_register[:]
            + second_ancilla_register[:],
        )

        self.objective = european_call_objective
        self.objective_index = self.uncertainty_model.num_qubits + 4
        self.option = circuit
        self.post_processor = self.objective.post_processing

    def _define_best_of_call_options(self, strike_prices: float, c_approx=0.01) -> None:
        optionConstructor = BestOfCallPricer(self.num_uncertainty_qubits)
        circuit, post_processor = optionConstructor.construct_circuit(
            strike_prices,
            self.uncertainty_model,
            self.upper_bound[0],
            self.lower_bound[0],
            c_approx=c_approx,
        )
        self.option = circuit
        self.objective_index = circuit.num_qubits - 1
        self.post_processor = post_processor

    def create_state_prep_circuit(self) -> QuantumCircuit:
        num_qubits = self.objective.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.append(self.uncertainty_model, range(self.uncertainty_model.num_qubits))
        circuit.append(self.objective, range(self.objective.num_qubits))
        self.objective_index = self.uncertainty_model.num_qubits
        self.option = circuit
        return circuit

    def create_estimation_problem(self, epsilon=0.01) -> EstimationProblem:
        scaling_param = np.sqrt(epsilon)
        self._define_payoff_function(
            self.options_params.option_type,
            self.options_params.strike_prices,
            c_approx=scaling_param,
        )
        self.option_type = self.options_params.option_type
        # check if pre-requisites are satisfied
        if self.objective_index is None:
            raise Exception("Objective index not defined yet!")
        if self.option is None:
            raise Exception("Option not defined yet!")
        if self.post_processor is None:
            raise Exception("Post processor not defined yet!")

        # create estimation problem
        self.problem = EstimationProblem(
            state_preparation=self.option,
            objective_qubits=[self.objective_index],
            post_processing=self.post_processor,
        )
        return self.problem

    def run(
        self, epsilon=0.01, alpha=0.05, shots=100, method="MIQAE"
    ) -> AmplitudeEstimatorResult:
        # construct amplitude estimation
        if method == "IQAE":
            ae = IterativeAmplitudeEstimation(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=Sampler(run_options={"shots": shots}),
            )
            self.result = ae.estimate(self.problem)
        elif method == "MIQAE":
            ae = ModifiedIterativeAmplitudeEstimation(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=Sampler(run_options={"shots": shots}),
            )
            self.result = ae.estimate(self.problem, shots=shots)
        return self.result

    def process_results(self) -> Tuple[float, np.ndarray]:
        conf_int = np.array(self.result.confidence_interval_processed)
        estimated_value = self.result.estimation_processed
        # apply discount
        estimated_value *= self.discount_factor
        conf_int *= self.discount_factor
        return estimated_value, conf_int

    def compute_exact_expectation(self) -> float:
        if self.option_type == "call":
            exact_value = self._compute_exact_call_expectation()
        elif self.option_type == "basket call":
            exact_value = self._compute_exact_basket_call_expectation()
        elif self.option_type == "spread call":
            exact_value = self._compute_exact_spread_call_expectation()
        elif self.option_type == "call-on-max":
            exact_value = self._compute_exact_call_on_max_expectation()
        elif self.option_type == "call-on-min":
            exact_value = self._compute_exact_call_on_min_expectation()
        elif self.option_type == "best-of-call":
            exact_value = self._compute_exact_best_of_call_expectation()
        else:
            raise Exception("Option type not defined!")
        # apply discount
        exact_value *= self.discount_factor
        return exact_value

    def _compute_exact_call_expectation(self) -> float:
        payoff = np.maximum(self.uncertainty_model.values - self.strike_price, 0)
        expected_value = np.dot(self.uncertainty_model.probabilities, payoff)
        return expected_value

    def _compute_exact_basket_call_expectation(self) -> float:
        sum_values = np.sum(self.uncertainty_model.values, axis=1)
        expected_value = np.dot(
            self.uncertainty_model.probabilities[sum_values >= self.strike_price],
            sum_values[sum_values >= self.strike_price] - self.strike_price,
        )
        return expected_value

    def _compute_exact_spread_call_expectation(self) -> float:
        diff_values = np.array([v[0] - v[1] for v in self.uncertainty_model.values])

        exact_value = np.dot(
            self.uncertainty_model.probabilities[diff_values >= self.strike_price],
            diff_values[diff_values >= self.strike_price] - self.strike_price,
        )

        return exact_value

    def _compute_exact_call_on_max_expectation(self) -> float:
        exact_value = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            exact_value += self.uncertainty_model.probabilities[i] * max(
                0,
                max(
                    self.uncertainty_model.values[i][0],
                    self.uncertainty_model.values[i][1],
                )
                - self.strike_price,
            )
        return exact_value

    def _compute_exact_call_on_min_expectation(self) -> float:
        exact_value = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            exact_value += self.uncertainty_model.probabilities[i] * max(
                0,
                self.strike_price
                - min(
                    self.uncertainty_model.values[i][0],
                    self.uncertainty_model.values[i][1],
                ),
            )
        return exact_value

    def _compute_exact_best_of_call_expectation(self) -> float:
        strike_price_1, strike_price_2 = self.strike_price
        curr_exact_expectation = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            curr_exact_expectation += self.uncertainty_model.probabilities[i] * max(
                0,
                self.uncertainty_model.values[i][0] - strike_price_1,
                self.uncertainty_model.values[i][1] - strike_price_2,
            )
        return curr_exact_expectation

    def estimate_expectation(
        self, epsilon=0.01, alpha=0.05, shots=100
    ) -> Tuple[float, np.ndarray]:
        self.create_estimation_problem(epsilon)
        self.run(epsilon, alpha, shots)
        return self.process_results()

    def get_num_oracle_calls(self) -> int:
        return self.result.num_oracle_queries


class BestOfCallPricer:
    def __init__(self, num_uncertainty_qubits: int):
        self.num_uncertainty_qubits = num_uncertainty_qubits

    def _map_strike_price_int(
        self, strike_price: float, high: float, low: float, num_uncertainty_qubits: int
    ) -> float:
        return int(
            np.ceil(
                (strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1)
            )
        )

    def _map_strike_price_float(
        self, strike_price: float, high: float, low: float, num_uncertainty_qubits: int
    ) -> float:
        return (strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1)

    def objective_function(
        self,
        num_uncertainty_qubits: int,
        strike_price: float,
        high: float,
        low: float,
        c_approx=0.01,
    ) -> Tuple[Gate, Callable, float]:
        model_register = QuantumRegister(num_uncertainty_qubits, "model")
        objective_register = QuantumRegister(1, "objective")

        mapped_strike_price_float = self._map_strike_price_float(
            strike_price, high, low, num_uncertainty_qubits
        )
        mapped_strike_price_int = self._map_strike_price_int(
            strike_price, high, low, num_uncertainty_qubits
        )

        circuit = QuantumCircuit(model_register, objective_register)
        step = (np.pi / 8) / (2**num_uncertainty_qubits - 1) * c_approx

        offset = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_int - 1)) / 2
        offset_angle = step * offset * 2

        circuit.ry(-step * mapped_strike_price_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit.cry(step * 2 ** (i + 1), model_register[i], objective_register[0])

        def post_processor(prob_of_one):
            coeff = (prob_of_one - 0.5) / (step) + offset
            fmax = high - strike_price
            og_range = 2**num_uncertainty_qubits - 1 - mapped_strike_price_float
            return coeff / og_range * (fmax)

        return circuit.to_gate(label="F"), post_processor, offset_angle

    def objective_function_two_strike(
        self,
        num_uncertainty_qubits: int,
        strike_price_1: float,
        strike_price_2: float,
        high: float,
        low: float,
        c_approx=0.01,
    ) -> Tuple[Gate, Gate, Callable, float]:

        # map strike prices
        mapped_strike_price_1_float = self._map_strike_price_float(
            strike_price_1, high, low, num_uncertainty_qubits
        )
        mapped_strike_price_1_int = self._map_strike_price_int(
            strike_price_1, high, low, num_uncertainty_qubits
        )

        mapped_strike_price_2_float = self._map_strike_price_float(
            strike_price_2, high, low, num_uncertainty_qubits
        )
        mapped_strike_price_2_int = self._map_strike_price_int(
            strike_price_2, high, low, num_uncertainty_qubits
        )

        step = (np.pi / 8) / (2**num_uncertainty_qubits - 1) * c_approx
        # determine offset
        offset_1 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_1_int - 1)) / 2
        offset_2 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_2_int - 1)) / 2

        offset = np.mean([offset_1, offset_2])
        offset_angle = step * offset * 2

        # create first payoff function
        model_register = QuantumRegister(num_uncertainty_qubits, "model")
        objective_register = QuantumRegister(1, "objective")

        circuit_one = QuantumCircuit(model_register, objective_register)
        circuit_one.ry(-step * mapped_strike_price_1_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit_one.cry(
                step * 2 ** (i + 1), model_register[i], objective_register[0]
            )

        # Create second_payoff_function
        circuit_two = QuantumCircuit(model_register, objective_register)
        circuit_two.ry(-step * mapped_strike_price_2_float * 2, objective_register[0])
        for i in range(num_uncertainty_qubits):
            circuit_two.cry(
                step * 2 ** (i + 1), model_register[i], objective_register[0]
            )

        def post_processor(prob_of_one):
            coeff = (prob_of_one - 0.5) / (step) + offset
            fmax = high - min(strike_price_1, strike_price_2)
            og_range = (
                2**num_uncertainty_qubits
                - 1
                - min(mapped_strike_price_1_float, mapped_strike_price_2_float)
            )
            return coeff / og_range * (fmax)

        return (
            circuit_one.to_gate(label="F1"),
            circuit_two.to_gate(label="F2"),
            post_processor,
            offset_angle,
        )

    def construct_circuit(
        self,
        strike_prices: List[float],
        uncertainty_model: LogNormalDistribution,
        high: float,
        low: float,
        c_approx=0.01,
    ) -> Tuple[QuantumCircuit, Callable]:
        first_var_register = QuantumRegister(self.num_uncertainty_qubits, "var1")
        first_ancilla_register = QuantumRegister(
            self.num_uncertainty_qubits, "ancilla1"
        )
        first_carry_register = QuantumRegister(1, "carry")

        second_var_register = QuantumRegister(self.num_uncertainty_qubits, "var2")
        second_ancilla_register = QuantumRegister(
            self.num_uncertainty_qubits, "ancilla2"
        )
        second_carry_register = QuantumRegister(1, "carry2")
        second_carry_register_copy = QuantumRegister(1, "carry2_copy")

        comparator_carry = QuantumRegister(1, "comparator_carry")
        comparator_ancilla = QuantumRegister(3, "comparator_ancilla")

        objective_register = QuantumRegister(1, "objective")
        circuit = QuantumCircuit(
            first_var_register,
            first_ancilla_register,
            first_carry_register,
            second_var_register,
            second_ancilla_register,
            second_carry_register,
            second_carry_register_copy,
            comparator_carry,
            comparator_ancilla,
            objective_register,
        )

        objective_1, objective_2, post_processor, offset_angle = (
            self.objective_function_two_strike(
                self.num_uncertainty_qubits,
                strike_prices[0],
                strike_prices[1],
                high,
                low,
                c_approx,
            )
        )

        comparator = QComp(
            self.num_uncertainty_qubits + 1, self.num_uncertainty_qubits + 1
        )
        subtractor = subtractorDraper(self.num_uncertainty_qubits)

        controlled_objective_1 = objective_1.control(num_ctrl_qubits=2, ctrl_state="10")
        controlled_objective_2 = objective_2.control(num_ctrl_qubits=2, ctrl_state="10")

        circuit.append(
            uncertainty_model, first_var_register[:] + second_var_register[:]
        )

        # set sin(x) to sin(x+pi/4)
        circuit.ry(np.pi / 2, objective_register)
        # offset
        circuit.ry(-offset_angle, objective_register[0])

        # load strike prices
        loadNumber(
            circuit,
            first_ancilla_register,
            self._map_strike_price_int(
                strike_prices[0], high, low, self.num_uncertainty_qubits
            ),
        )
        loadNumber(
            circuit,
            second_ancilla_register,
            self._map_strike_price_int(
                strike_prices[1], high, low, self.num_uncertainty_qubits
            ),
        )
        circuit.append(
            subtractor,
            first_var_register[:]
            + first_ancilla_register[:]
            + [first_carry_register[0]],
        )
        circuit.append(
            subtractor,
            second_var_register[:]
            + second_ancilla_register[:]
            + [second_carry_register[0]],
        )
        circuit.cx(second_carry_register[0], second_carry_register_copy[0])
        circuit.append(
            comparator,
            first_ancilla_register[:]
            + first_carry_register[:]
            + second_ancilla_register[:]
            + second_carry_register[:]
            + comparator_carry[:]
            + comparator_ancilla[:],
        )

        circuit.append(
            controlled_objective_2,
            second_carry_register_copy[:]
            + [comparator_ancilla[0]]
            + second_var_register[:]
            + objective_register[:],
        )
        circuit.append(
            controlled_objective_1,
            first_carry_register[:]
            + [comparator_ancilla[1]]
            + first_var_register[:]
            + objective_register[:],
        )
        circuit.append(
            controlled_objective_1,
            first_carry_register[:]
            + [comparator_ancilla[2]]
            + first_var_register[:]
            + objective_register[:],
        )
        return circuit, post_processor

    def create_estimation_problem(
        self, circuit: QuantumCircuit, objective_qubit: int, post_processor: Callable
    ) -> EstimationProblem:
        self.problem = EstimationProblem(
            state_preparation=circuit,
            objective_qubits=[objective_qubit],
            post_processing=post_processor,
        )
        return self.problem
