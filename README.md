# quantum-finance
Package for `qfinance` used in https://arxiv.org/abs/2301.09241

# Installation

Navigate to the root directory of this package. Use `pip` to install:
```bash
pip install .
```

The required libraries should be installed with the comment. If it is not, you can install it with the `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you are using conda, you can create a new `conda` environment with the `environment.yml`:
```bash
conda env create -f environment.yml
```

# Basic Usage
Basic usage is demonstrated in the `test_package.ipynb` notebook.
The source code is under the `qfinance/` directory

# Benchmarking
Scripts used for generating graphs in the paper is under `scripts/` directory. It is much more performant to define everything within a single python file.

# Modified Iterative Quantum Amplitude Estimation
This algorithm use the Modified IQAE by Fukuzawa et al., obtained from (https://github.com/shifubear/Modified-IQAE). I restructured the code to work with the newest version of Qiskit, and cut down the code to streamline with our `qfinance` application. Since there is no `pip` installation options for MIQAE, I have included the streamlined code within our `qfinance` for easy installation. I stress that the development of the logic for MIQAE is from Fukuzawa et al.. For citation of the MIQAE work, refer to https://github.com/shifubear/Modified-IQAE .
