# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Modified Iterative Quantum Amplitude Estimation Algorithm."""

from typing import Optional, Union, List, Tuple, Dict, cast
import numpy as np
import math
from scipy.stats import beta
import warnings
from collections import defaultdict

from qiskit import Aer, ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.primitives import BaseSampler

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem
from ..exceptions import AlgorithmError


class ModifiedIterativeAmplitudeEstimation(AmplitudeEstimator):
    r"""The Iterative Amplitude Estimation algorithm.

    This class implements the Iterative Quantum Amplitude Estimation (IQAE) algorithm, proposed
    in [1]. The output of the algorithm is an estimate that,
    with at least probability :math:`1 - \alpha`, differs by epsilon to the target value, where
    both alpha and epsilon can be specified.

    It differs from the original QAE algorithm proposed by Brassard [2] in that it does not rely on
    Quantum Phase Estimation, but is only based on Grover's algorithm. IQAE iteratively
    applies carefully selected Grover iterations to find an estimate for the target amplitude.

    References:
        [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        confint_method: str = "beta",
        min_ratio: float = 2,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        The output of the algorithm is an estimate for the amplitude `a`, that with at least
        probability 1 - alpha has an error of epsilon. The number of A operator calls scales
        linearly in 1/epsilon (up to a logarithmic factor).

        Args:
            epsilon_target: Target precision for estimation target `a`, has values between 0 and 0.5
            alpha: Confidence level, the target probability is 1 - alpha, has values between 0 and 1
            confint_method: Statistical method used to estimate the confidence intervals in
                each iteration, can be 'chernoff' for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals (default)
            min_ratio: Minimal q-ratio (:math:`K_{i+1} / K_i`) for FindNextK
            quantum_instance: Quantum Instance or Backend

        Raises:
            AlgorithmError: if the method to compute the confidence intervals is not supported
            ValueError: If the target epsilon is not in (0, 0.5]
            ValueError: If alpha is not in (0, 1)
            ValueError: If confint_method is not supported
        """
        # validate ranges of input arguments
        if not 0 < epsilon_target <= 0.5:
            raise ValueError(f"The target epsilon must be in (0, 0.5], but is {epsilon_target}.")

        if not 0 < alpha < 1:
            raise ValueError(f"The confidence level alpha must be in (0, 1), but is {alpha}")

        if confint_method not in {"chernoff", "beta"}:
            raise ValueError(
                "The confidence interval method must be chernoff or beta, but "
                f"is {confint_method}."
            )

        super().__init__()

        # set quantum instance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.quantum_instance = quantum_instance

        # store parameters
        self._epsilon = epsilon_target
        self._alpha = alpha
        self._min_ratio = min_ratio
        self._confint_method = confint_method
        self._sampler = sampler

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler.

        Returns:
            The sampler used to run this algorithm.
        """
        return self._sampler
    
    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set the sampler.

        Args:
            sampler: The sampler used to run this algorithm.
        """
        self._sampler = sampler
    
    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, Backend]
    ) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    @property
    def epsilon_target(self) -> float:
        """Returns the target precision ``epsilon_target`` of the algorithm.

        Returns:
            The target precision (which is half the width of the confidence interval).
        """
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args:
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon

    def _find_next_k(
        self,
        k_prev: int,
        theta_interval: Tuple[float, float],
    ) -> int:
        """Find the largest integer k_next, such that the interval (4 * k_next + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k: The current power of the Q operator.
            upper_half_circle: Boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi].
            theta_interval: The current confidence interval for the angle theta,
                i.e. (theta_lower, theta_upper).
            min_ratio: Minimal ratio K/K_next allowed in the algorithm.

        Returns:
            The next power k, and boolean flag for the extrapolated interval.

        Raises:
            AlgorithmError: if min_ratio is smaller or equal to 1
        """

        # initialize variables
        theta_l, theta_u = theta_interval
        K_prev = 2 * k_prev + 1
        K = int(1 / (theta_u-theta_l))
        K -= (K + 1) % 2 # subtract 1 if even
        
        while K >= 3 * K_prev:
            R_u = np.ceil(K * theta_u) - 1
            R_l = int(K * theta_l)
            
            if (K * theta_u) - R_u < self._epsilon / 1000:
                # print((K * theta_u) - R_u)
                R_u -= 1
            if R_u == R_l:
                return (K - 1) // 2 # integer is guaranteed, but cast to int
            K -= 2
        
        return k_prev
    
    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int = 0, measurement: bool = False
    ) -> QuantumCircuit:
        r"""Construct the circuit :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE  circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit implementing :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    def _good_state_probability(
        self,
        problem: EstimationProblem,
        counts_or_statevector: Union[Dict[str, int], np.ndarray],
        num_state_qubits: int,
    ) -> Union[Tuple[int, float], float]:
        """Get the probability to measure '1' in the last qubit.

        Args:
            problem: The estimation problem, used to obtain the number of objective qubits and
                the ``is_good_state`` function.
            counts_or_statevector: Either a counts-dictionary (with one measured qubit only!) or
                the statevector returned from the statevector_simulator.
            num_state_qubits: The number of state qubits.

        Returns:
            If a dict is given, return (#one-counts, #one-counts/#all-counts),
            otherwise Pr(measure '1' in the last qubit).
        """
        if isinstance(counts_or_statevector, dict):
            one_counts = 0
            for state, counts in counts_or_statevector.items():
                if problem.is_good_state(state):
                    one_counts += counts

            return int(one_counts), one_counts / sum(counts_or_statevector.values())
        else:
            statevector = counts_or_statevector
            num_qubits = int(np.log2(len(statevector)))  # the total number of qubits

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                # consider only state qubits and revert bit order
                bitstr = bin(i)[2:].zfill(num_qubits)[-num_state_qubits:][::-1]
                objectives = [bitstr[index] for index in problem.objective_qubits]
                if problem.is_good_state(objectives):
                    prob = prob + np.abs(amplitude) ** 2

            return prob

    def estimate(
        self, estimation_problem: EstimationProblem,
        shots: int,
        ground_truth: float=None, 
        min_ratio: float=2.0, 
        state: dict={},
        nmax_only=False,
        verbose=False
    ) -> "ModifiedIterativeAmplitudeEstimationResult":
        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
        ratios = []  # list of multiplication factors (called 'q' in paper)
        theta_intervals = [[0, 1]]  # a priori knowledge of (theta / 2 / pi) TODO: fix
        a_intervals = [[0.0, 1.0]]  # a priori knowledge of the confidence interval of the estimate
        num_oracle_queries = 0
        num_shots = [] # number of shots taken per iteration
        
        self.sampler_shots = shots

        # maximum number of rounds
#         T = int(np.log(3 * np.pi / 4 / self._epsilon, 3))
        K_max = np.pi / 4 / self._epsilon

        if state == {}:
            state = defaultdict(dict)
    
        if 'K_max' not in state: state['K_max'] = []
        state['K_max'].append(K_max)
                
#         if verbose:
#             print('T:', T)
#             print()
        if self._quantum_instance is None and self._sampler is None:
            raise ValueError("A quantum instance or sampler must be provided.")
        
        # for statevector we can directly return the probability to measure 1
        # note, that no iterations here are necessary
        if self._quantum_instance and self._quantum_instance.is_statevector:
            # simulate circuit
            circuit = self.construct_circuit(estimation_problem, k=0, measurement=False)
            ret = self._quantum_instance.execute(circuit)

            # get statevector
            statevector = ret.get_statevector(circuit)

            # calculate the probability of measuring '1'
            num_qubits = circuit.num_qubits - circuit.num_ancillas
            prob = self._good_state_probability(estimation_problem, statevector, num_qubits)
            prob = cast(float, prob)  # tell MyPy it's a float and not Tuple[int, float ]

            a_confidence_interval = [prob, prob]  # type: List[float]
            a_intervals.append(a_confidence_interval)

            theta_i_interval = [
                np.arccos(1 - 2 * a_i) / 2 / np.pi for a_i in a_confidence_interval  # type: ignore
            ]
            theta_intervals.append(theta_i_interval)
            num_oracle_queries = 0  # no Q-oracle call, only a single one to A

        else:
            num_iterations = 0  # keep track of the number of iterations
            
            # constant for N_i^max
            SIN_CONST = 2 / np.square(np.sin(np.pi / 21)) / np.square(np.sin(8 * np.pi / 21))
            
            k = 0
            
            # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
            while theta_intervals[-1][1] - theta_intervals[-1][0] > 4 * self._epsilon / np.pi:
                num_iterations += 1
                
                k = powers[num_iterations - 1]
                K = 2*k+1
                alpha_i = 2 * self._alpha / 3 * K / K_max # confidence level for this iteration
                shots_i_max = int(SIN_CONST * np.log(2 / alpha_i))
                
                one_counts_total = 0
                
                round_shots = 0
                
                while powers[num_iterations - 1] == k:

                    # Early/late condition by ChebAE
                    if self._confint_method == "beta":
                        nu = 8
                        # If early...
                        if self._check_early(shots, alpha_i, nu, theta_intervals, self._epsilon):
                            N = min(round_shots + shots, shots_i_max)

                            shots = N - round_shots
                            
                            # if self._sampler:
                            #     self._sampler_shots = N - round_shots
                            if self._quantum_instance:
                                self._quantum_instance._run_config.shots = N - round_shots
                            
                            round_shots = N
                        # If late...
                        else:
                            # if self._sampler:
                            #     self._sampler_shots = 1
                            if self._quantum_instance:
                                self._quantum_instance._run_config.shots = 1

                            N = min(round_shots + 1, shots_i_max)                            
                            round_shots = N

                    # Standard Chernoff version                         
                    else: 
                        N = min(round_shots + shots, shots_i_max)

                        shots = N - round_shots
                        # if self._sampler:
                        #     self._sampler_shots = N - round_shots
                        if self._quantum_instance:
                            self._quantum_instance._run_config.shots = N - round_shots
                        
                        round_shots = N

                    if verbose:
                        print()
                        print('shots_i_max:', shots_i_max)
                        print('N:', N)
                        print('N - round_shots:', shots)
                
                    ## run measurements for Q^k A|0> 
                    
                    # if self._sampler:
                    #     circuit = self.construct_circuit(estimation_problem, k, measurement=True)
                    #     ret = self._sampler.run([circuit], shots=self._sampler_shots).result()
                    #     print(ret)
                    #     quasi_dist = ret.quasi_dists[0]
                    #     print(quasi_dist)
                    #     counts = {}
                    #     for key, value in quasi_dist.items():
                    #         counts[bin(int(key))[2:].zfill(circuit.num_qubits)] = int(value*self._sampler_shots)
                        
                    #     # calculate the probability of measuring '1', 'prob' is a_i in the paper
                    #     num_qubits = circuit.num_qubits - circuit.num_ancillas
                    #     # type: ignore
                    #     one_counts, _ = self._good_state_probability(
                    #         estimation_problem, counts, num_qubits
                    #     )
                        
                    if self._quantum_instance:
                        circuit = self.construct_circuit(estimation_problem, k, measurement=True)
                        ret = self._quantum_instance.execute(circuit)

                        # get the counts and store them
                        counts = ret.get_counts(circuit) # TODO: is this sum of 1s measured across all shots?

                        # calculate the probability of measuring '1', 'prob' is a_i in the paper
                        num_qubits = circuit.num_qubits - circuit.num_ancillas
                        # type: ignore
                        one_counts, _ = self._good_state_probability(
                            estimation_problem, counts, num_qubits
                        )
                    
                    else:
                        theta = 0.5 * np.arccos(1 - 2*ground_truth) #k0/N
                        a_est = np.sin((2*k+1)*theta)**2
                        
                        one_counts = np.random.binomial(1, a_est, size=shots).sum()
                    
                    
                    one_counts_total += one_counts
                    prob = one_counts_total / N

                    if verbose:
                        print('one_counts:', one_counts)
                        print('prob:', prob)
                    
                    ##
                
                    # compute a_min_i, a_max_i
                    if self._confint_method == "chernoff":
                        a_i_min, a_i_max = _chernoff_confint(prob, N, alpha_i)
                    else:  # 'beta'
                        a_i_min, a_i_max = _clopper_pearson_confint(
                            one_counts_total, N, alpha_i
                        )
                    
                    R_i = int(K * theta_intervals[-1][0])
                    if verbose:
                        print("R_i:", R_i)
                    q_i = (R_i % 4) + 1

                    # compute theta_i_min, theta_i_max
                    if q_i % 2 == 1:
                        theta_i_min = np.arcsin(np.sqrt(a_i_min))
                        theta_i_max = np.arcsin(np.sqrt(a_i_max)) 
                    elif q_i % 2 == 0:
                        theta_i_min = -np.arcsin(np.sqrt(a_i_max)) + np.pi/2
                        theta_i_max = -np.arcsin(np.sqrt(a_i_min)) + np.pi/2
                    else:
                        raise ValueError('Invalid quartile computed')

                    if not True:
                        print('equal R:', R_equal)
                        print(f'q_i: {q_i}, theta_i_min: {theta_i_min}, theta_i_max: {theta_i_max}')
                        
                    # fixing units of theta_i ci and ensuring it lies in [0, 1]
                    theta_i_min /= (np.pi / 2)
                    theta_i_max /= (np.pi / 2)
                    # theta_i_min -= (q_i - 1)
                    # theta_i_max -= (q_i - 1)
                    
                    # compute theta_u, theta_l of this iteration
                    theta_l = (R_i + theta_i_min) / K
                    theta_u = (R_i + theta_i_max) / K
                        
                    theta_intervals.append([theta_l, theta_u])

                    # compute a_u_i, a_l_i
                    a_l = float(np.square(np.sin(np.pi * theta_l / 2)))
                    a_u = float(np.square(np.sin(np.pi * theta_u / 2)))
                    a_intervals.append([a_l, a_u])

                    if verbose:
                        print('a i interval:', [a_i_min, a_i_max])
                        print('theta i interval:', [theta_i_min, theta_i_max])
                        print('theta interval:', [theta_l, theta_u])
                        print('a interval:', [a_l, a_u])
                    
                    if theta_intervals[-1][1] - theta_intervals[-1][0] < 4 * self._epsilon / np.pi:
                        break
    
                    # get the next k
                    if math.isnan(1 / (theta_intervals[-1][1]-theta_intervals[-1][0])):
                        print(f'confint: {self._confint_method}')
                        print('a i interval:', [a_i_min, a_i_max])
                        print('theta i interval:', [theta_i_min, theta_i_max])
                        print('theta interval:', [theta_l, theta_u])
                        print('a interval:', [a_l, a_u])
                        print(f"ISNAN: {theta_intervals[-1]}\n")
                    k = self._find_next_k(
                        powers[-1],
                        theta_intervals[-1],
                    )
                    
                    if verbose:
                        print('k_i:', k)
            
                # after inner loop terminates
                # store the variables
                
                powers.append(k)
                ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))
                num_shots.append(N) # TODO: change name later

                if verbose:
                    print('  k_i:', k)

                # track number of Q-oracle calls
                num_oracle_queries += N * K
                
#                 # bookkeeping
                state['round_shots'][k] = N
                state['n_queries'][k] = N * K
                
                if verbose:
                    print('round_shots:', round_shots) # look at this, changing between iterations
                
                if verbose:
                    print()
                    
                
                
        # get the latest confidence interval for the estimate of a
        confidence_interval = tuple(a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = ModifiedIterativeAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.post_processing = estimation_problem.post_processing
        result.num_oracle_queries = num_oracle_queries

        result.estimation = estimation
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = estimation_problem.post_processing(estimation)
        confidence_interval = tuple(
            estimation_problem.post_processing(x) for x in confidence_interval
        )
        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (confidence_interval[1] - confidence_interval[0]) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals
        result.powers = powers
        result.ratios = ratios
        
        state['ks'] = powers

        return result

    def _check_early(self, shots, alpha, nu, theta_intervals, eps):
        import math 
        # Get max eps threshold 
        eps_max = -math.inf 
        for x in range(1, shots):
            lower = beta.ppf(1 - alpha / 2, x + 1, shots - x)
            upper = beta.ppf(alpha / 2, x, shots - x + 1)
            if np.abs(upper - lower) > eps_max:
                eps_max = np.abs(upper - lower)
        num = theta_intervals[-1][1] - theta_intervals[-1][0]
        den = np.abs(math.sin(theta_intervals[-1][1] - math.sin(theta_intervals[-1][0])))
        return eps_max * num / den > eps * nu 
        


class ModifiedIterativeAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``IterativeAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._alpha = None
        self._epsilon_target = None
        self._epsilon_estimated = None
        self._epsilon_estimated_processed = None
        self._estimate_intervals = None
        self._theta_intervals = None
        self._powers = None
        self._ratios = None
        self._confidence_interval_processed = None

    @property
    def alpha(self) -> float:
        r"""Return the confidence level :math:`\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        r"""Set the confidence level :math:`\alpha`."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width of the confidence interval."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Set the target half-width of the confidence interval."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the estimated half-width of the confidence interval."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Set the estimated half-width of the confidence interval."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the post-processed estimated half-width of the confidence interval."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Set the post-processed estimated half-width of the confidence interval."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> List[List[float]]:
        """Return the confidence intervals for the estimate in each iteration."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: List[List[float]]) -> None:
        """Set the confidence intervals for the estimate in each iteration."""
        self._estimate_intervals = value

    @property
    def theta_intervals(self) -> List[List[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: List[List[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value

    @property
    def powers(self) -> List[int]:
        """Return the powers of the Grover operator in each iteration."""
        return self._powers

    @powers.setter
    def powers(self, value: List[int]) -> None:
        """Set the powers of the Grover operator in each iteration."""
        self._powers = value

    @property
    def ratios(self) -> List[float]:
        r"""Return the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        return self._ratios

    @ratios.setter
    def ratios(self, value: List[float]) -> None:
        r"""Set the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        self._ratios = value

    @property
    def confidence_interval_processed(self) -> Tuple[float, float]:
        """Return the post-processed confidence interval."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: Tuple[float, float]) -> None:
        """Set the post-processed confidence interval."""
        self._confidence_interval_processed = value


def _chernoff_confint(
    value: float, shots: int, alpha_i: float
) -> Tuple[float, float]:
    """Compute the Chernoff confidence interval for `shots` i.i.d. Bernoulli trials.

    The confidence interval is

        [value - eps, value + eps], where eps = sqrt(3 * log(2 * T/ alpha) / shots)

    but at most [0, 1].

    Args:
        value: The current estimate.
        shots: The number of shots.
        T: The maximum number of rounds, used to compute epsilon_a.
        alpha: The confidence level, used to compute epsilon_a.

    Returns:
        The Chernoff confidence interval.
    """
    
    # TODO: rename the parameters
    eps = np.sqrt(1 / (2 * shots) * np.log(2 / alpha_i))
    lower = np.maximum(0, value - eps)
    upper = np.minimum(1, value + eps)
    return lower, upper


def _clopper_pearson_confint(counts: int, shots: int, alpha: float) -> Tuple[float, float]:
    """Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.

    Args:
        counts: The number of positive counts.
        shots: The number of shots.
        alpha: The confidence level for the confidence interval.

    Returns:
        The Clopper-Pearson confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    if counts != 0:
        lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

    # if counts == shots, the beta quantile returns nan
    if counts != shots:
        upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

    return lower, upper
