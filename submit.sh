#!/bin/bash

if [ $CONDA_DEFAULT_ENV != "qfinance" ]; then
    eval "$(conda shell.bash hook)" # this is needed to activate conda
    conda activate qfinance
fi

# Default number of cores
num_cores=1
memory_profiling=0
time_profiling=0

# default output log directory
OUTPUT_DIR="logs"

# Parse command line options
while getopts ":n:mt" opt; do
  case ${opt} in
    n )
      num_cores=$OPTARG
      ;;
    m )
      memory_profiling=1
      ;;
    t )
      time_profiling=1
      ;;
    \? )
      echo "Usage: cmd [-n num_cores] [-m][-t] python_script [output_file]"
      ;;
    : )
      echo "Option -$OPTARG requires an argument"
      ;;
  esac
done

# Shift to access remaining arguments
shift $((OPTIND -1))

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

python_cmd="python -u"
if [ $memory_profiling -eq 1 ]; then
    python_cmd="python -m memory_profiler"
fi
if [ $time_profiling -eq 1 ]; then
    python_cmd="kernprof -l"
fi

# Run the Python script with or without memory profiling
if [ $# -eq 1 ]; then 
    DEFAULT_OUTPUT_PATH="$OUTPUT_DIR/$(basename $1).out"
    export NUM_CORES=$num_cores
    nohup $python_cmd $1 > "$DEFAULT_OUTPUT_PATH" &
elif [ $# -eq 2 ]; then
    FULL_OUTPUT_PATH="$OUTPUT_DIR/$2"
    export NUM_CORES=$num_cores
    nohup $python_cmd $1 > "$FULL_OUTPUT_PATH" &
else
    echo "Usage: cmd [-n num_cores] [-m] python_script [output_file]"
fi
