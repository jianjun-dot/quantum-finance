# quantum-finance
Package for `qfinance`.

# Installation
Navigate to the root directory of this package. Use `pip` to install:
```
pip install .
```

# Modified Iterative Quantum Amplitude Estimation
This algorithm use the Modified IQAE by Fukuzawa et al., obtained from (https://github.com/shifubear/Modified-IQAE). I restructured the code to work with the newest version of Qiskit, and cut down the code to streamline with our `qfinance` application. Since there is no `pip` installation options for MIQAE, I have included the streamlined code within our `qfinance` for easy installation. I stress that the development of the logic for MIQAE is from Fukuzawa et al.. For citation of the MIQAE work, refer to https://github.com/shifubear/Modified-IQAE .
