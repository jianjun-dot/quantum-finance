#!/bin/bash

# if [ $CONDA_DEFAULT_ENV != "qstochastic" ]; then
#     eval "$(conda shell.bash hook)" # this is needed to activate conda
#     conda activate qstochastic
# fi

# if [ $# -eq 1 ]; then 
#     nohup python -u $1 &
# elif [ $# -eq 2 ]; then
#     nohup python -u $1 > $2 &
# fi

if [ $CONDA_DEFAULT_ENV != "qfinance" ]; then
    eval "$(conda shell.bash hook)" # this is needed to activate conda
    conda activate qfinance
fi

# Default number of cores
num_cores=1

# Parse command line options
while getopts ":n:" opt; do
  case ${opt} in
    n )
      num_cores=$OPTARG
      ;;
    \? )
      echo "Usage: cmd [-n num_cores] python_script [output_file]"
      ;;
    : )
      echo "Option -$OPTARG requires an argument"
      ;;
  esac
done

# Shift to access remaining arguments
shift $((OPTIND -1))

# Check for python script and optionally output file
if [ $# -eq 1 ]; then 
    export NUM_CORES=$num_cores
    nohup python -u $1 &
elif [ $# -eq 2 ]; then
    export NUM_CORES=$num_cores
    nohup python -u $1 > $2 &
else
    echo "Usage: cmd [-n num_cores] python_script [output_file]"
fi
