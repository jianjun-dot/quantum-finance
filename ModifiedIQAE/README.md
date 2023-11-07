# Modified Iterative Quantum Amplitude Estimation is Asymptotically Optimal

Install dependencies and run `python modified_iqae.py`. Edit `config.yaml` to modify experiment parameters. Running the script saves a folder that contains the input config file, a `.csv` file with estimations, and optionally the plots included in the paper.

---
## Config parameters
All of the following keys must be present in the config with associated values.
- **runs** (`int`): number of runs for the experiment.
- **shots** (`int`): $N_{\text{shots}}$ in the paper, number of circuit trials per quantum execution. Not applicable to the `'classical'` backend.
- **a_resolution** (`int`): the resolution of the amplitude, which is the number of qubits used to express the amplitude in the circuit. Only accepts powers of 2.
- **a_step** (`int`, `List[int]`): the size of the step taken between tested amplitudes in units of `a_resolution`. An integer `a_step` tests all amplitudes on $[0,1]$. A list of step values tests those specific amplitudes.
- **alpha** (`int`): $\alpha$ as defined in the paper, representing the confidence level or maximum failure rate of the algorithm.
- **epsilons** (`List[float]`): sequence of maximum error tolerances $\epsilon$.
- **confint_method** {`'chernoff`, `'beta'`, `'all'`}:

### Optional parameters
Unless specified otherwise in the config, these keys will use their default values.
- **experiment_name** (`str`): Default: `<datetime>_modified-iqae`.
- **results_path** (`str`): Default: `./results/<experiment_name>`.
- **simulator** (`str`): specifies which [Qiskit backend](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html) to use. Can also pass in the string `'classical'` to simulate circuit measurements using a Bernoulli distribution. Default: `aer_simulator`.
- **compare** (`bool`): whether to add the original (Qiskit builtin) IQAE to the experiment results. Default: `True`.
- **noise** (`float`): adds a value to each tested amplitude $a$. This value is sampled from the uniform distribution on the closed interval $[a-$ `noise` $, a+$ `noise` $]$. If necessary, the endpoints are shifted to fit within the interval $[0,1]$. Default: `0.0`.
- **plots** (`bool`): whether to save plots. Default: `True`.
- **verbose** (`bool`): prints intermediate algorithm values if `True`. Default: `False`. *Use with caution for high numbers of experiments.*
---

## Cite this work
If you find our work useful, please consider using the following citation:
```
@misc{fukuzawa2022,
  title = {Modified {{Iterative Quantum Amplitude Estimation}} Is {{Asymptotically Optimal}}},
  author = {Fukuzawa, Shion and Ho, Christopher and Irani, Sandy and Zion, Jasen},
  year = {2022},
  month = aug,
  number = {arXiv:2208.14612},
  eprint = {2208.14612},
  eprinttype = {arxiv},
  primaryclass = {quant-ph},
  keywords = {Quantum Physics},
}

```
