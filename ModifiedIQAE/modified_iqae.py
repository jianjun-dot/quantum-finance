#!/usr/bin/env python
# coding: utf-8

# Modified IQAE tests

import numpy as np
import matplotlib.pyplot as plt

import os
import yaml
import csv
import datetime
import itertools
from tqdm import tqdm

from random import sample, seed
from collections import defaultdict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.algorithms import amplitude_estimators, EstimationProblem
from qiskit.algorithms import IterativeAmplitudeEstimation as BaseIterativeAmplitudeEstimation

from algorithms import IterativeAmplitudeEstimation, ModifiedIterativeAmplitudeEstimation
from operators import *
from plots import make_plots


with open('./config.yaml') as f:
    config = yaml.safe_load(f)

# add default arguments if not already there
defaults = {
    'experiment_name': '{}_modified-iqae',
    'results_path': './results/{}',
    'simulator': 'aer_simulator',
    'compare': True,
    'noise': 0.0,
    'plots': True,
    'verbose': False,
}

now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
defaults['experiment_name'] = defaults['experiment_name'].format(now)
defaults['results_path'] = defaults['results_path'].format(defaults['experiment_name'])

for key in defaults:
    if key not in config:
        config[key] = defaults[key]

# process parameters from config

runs = config['runs']
shots = config['shots']
N = config['a_resolution']
step = config['a_step']
alpha = config['alpha']
epsilons = config['epsilons']
confint_method = config['confint_method']
noise = config['noise']
experiment_name = config['experiment_name']
results_path =  config['results_path']
simulator = config['simulator']
compare = config['compare']
plots = config['plots']
verbose = config['verbose']


if isinstance(step, int):
    amplitudes = np.arange((N // step)+1) * step
else:
    amplitudes = step

epsilons = [float(eps) for eps in epsilons]

if confint_method == 'all':
    methods = ['chernoff', 'beta']
else:
    methods = [confint_method]

algs = ['miae']
if compare:
    algs.append('iae')

results_per_round_path = os.path.join(results_path, 'per_round')

if not os.path.exists('./results'):
    os.mkdir('./results')
if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists(results_per_round_path):
    os.mkdir(results_per_round_path)


# Compare Modified IQAE to No-Quantum IQAE
# define the estimation problem and oracle function

def make_problem(n, marked):

    def good_state(state):
        bin_marked = [(n-len(bin(s))+2)*'0'+bin(s)[2:] for s in marked]
        return (state in bin_marked)

    problem = EstimationProblem(
        state_preparation=A(n),        # A operator
        grover_operator=Q(n, marked),  # Q operator
        objective_qubits=range(n),
        is_good_state=good_state       # the "good" state Psi1 is identified as measuring |1> in qubit 0
    )

    return problem

# process the per-round info
def process_state(state, verbose=False):
    if verbose:
        for k,v in state.items():
            print(k)
            print(v)
    if len(state) == 0: return [],[],[]
    round_shots = state['round_shots']
    queries = state['n_queries']
    removed = False

    k_i = [k for k in round_shots]
    queries_i = [queries[k] for k in k_i]
    shots_i = ([shots_at_k0] if removed else []) + [round_shots[k] for k in k_i]

    return shots_i, queries_i, k_i

def experiment(alg: str, shots: int, M: int, N: int,
               alpha: float, epsilon: float, confint_method: str,
               noise: float, runs: int, simulator: str):

    AE = None
    if alg == 'miae':
        AE = ModifiedIterativeAmplitudeEstimation(epsilon_target=epsilon,
                                                  alpha=alpha,
                                                  confint_method=confint_method,
                                                  quantum_instance=simulator)

    else: # alg == 'iae'
        AE = IterativeAmplitudeEstimation(epsilon_target=epsilon,
                                          alpha=alpha,
                                          confint_method=confint_method,
                                          quantum_instance=simulator)


    n = int(np.log2(N))
    marked = sample(range(N), M)
    problem = make_problem(n, marked)
    ae_results = []
    successes = []

    for _ in range(runs):

        # setup
        state = defaultdict(dict)
        scaling = 1 # figure out what to do here

        if noise > 0 and simulator == 'classical':
            noisy_M = np.random.normal(M, 0.05)
            noisy_M = min(max(noisy_M, 0), N)
        else:
            N = max(128, N * scaling) # replace with the correct formula
            n = int(np.log2(N))
            sampled_noise = np.random.uniform(-noise if M > N else 0, noise if M < N else 0)
            scaled_noisy_M = int(max(0, min(N, (M + sampled_noise) * scaling)))

            marked = sample(range(N), scaled_noisy_M)
            problem = make_problem(n, marked)

            # scale back to original resolution
            noisy_M = scaled_noisy_M / scaling
            N = config['a_resolution']
            n = int(np.log2(N))

        # run estimation
        ae_result = AE.estimate(problem,
                                shots=shots,
                                ground_truth=noisy_M/N if noise > 0 else M/N,
                                state=state,
                                verbose=verbose)

        ae_results.append(ae_result)
        successes.append(ae_result.confidence_interval_processed[0] <= noisy_M/N if noise > 0 else M/N <= ae_result.confidence_interval_processed[1])

        if verbose: print()

        # round-specific logs
        round_shots, round_queries, round_k = process_state(state)

        round_results_csv = os.path.join(results_per_round_path, f'{M/N}_{confint_method}_{epsilon:.1e}_{alg}_round-results.csv')

        with open(round_results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round_shots'] + round_shots)
            writer.writerow(['round_queries'] + round_queries)
            writer.writerow(['round_k'] + round_k)

    # experiment-specific logs
    half_ci_widths = np.array([(res.confidence_interval_processed[1] - res.confidence_interval_processed[0]) / 2 for res in ae_results])
    queries = np.array([res.num_oracle_queries for res in ae_results])
    estimations = np.array([res.estimation for res in ae_results])
    successes = np.array(successes)

    results_csv = os.path.join(results_path, f'{M/N}_{confint_method}_{alg}_results.csv')

    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epsilon, 1-successes.mean()] + queries.tolist())

    if verbose:
        print(f'Results using algorithm \'{alg}\', amplitude {M}/{N} = {M/N}', end='')
        print(f' (with noise: {noisy_M/N})' if noise > 0 else '', end=' ')
        print(f'averaged over {runs} runs:')
        print(f'\tEstimation: {round(estimations.mean(), 10)}')
        print(f'\tTotal queries: {round(queries.mean())}')
        print(f'\tCI width: {half_ci_widths.mean()}')


if __name__ == '__main__':
    experiment_combos = list(itertools.product(algs, amplitudes, epsilons, methods))
    print('Experiment configuration size:', len(experiment_combos))
    print('# runs per configuration:', runs)
    print('Total runs:', len(experiment_combos) * runs)

    # re-save the config into the file
    with open(os.path.join(results_path, 'config.yaml'), 'w') as out:
        yaml.dump(config, out)

    for alg, M, epsilon, method in tqdm(experiment_combos):
        experiment(alg, shots, M, N, alpha, epsilon, method, noise, runs, simulator)

    print('Generating plots...')
    if plots:
        make_plots(results_path)

    print('Done')
