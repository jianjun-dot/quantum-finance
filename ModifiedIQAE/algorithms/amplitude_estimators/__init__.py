# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Amplitude Estimators package."""

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .ae import AmplitudeEstimation, AmplitudeEstimationResult
# from .fae import FasterAmplitudeEstimation, FasterAmplitudeEstimationResult
from .iae import IterativeAmplitudeEstimation, IterativeAmplitudeEstimationResult
from .mod_iae import ModifiedIterativeAmplitudeEstimation, ModifiedIterativeAmplitudeEstimationResult
# from .noq_iae import NoQuantumIterativeAmplitudeEstimation, NoQuantumIterativeAmplitudeEstimationResult
# from .mlae import MaximumLikelihoodAmplitudeEstimation, MaximumLikelihoodAmplitudeEstimationResult
from .estimation_problem import EstimationProblem

__all__ = [
    "AmplitudeEstimator",
    "AmplitudeEstimatorResult",
    "AmplitudeEstimation",
    "AmplitudeEstimationResult",
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationResult",
    "ModifiedIterativeAmplitudeEstimation",
    "ModifiedIterativeAmplitudeEstimationResult",
    "EstimationProblem",
]
