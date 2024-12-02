# FastKG

## Installation
    
To install, first clone the repository and move into the folder. Then run: `pip install -e .`

# CPU/GPU Testing
To test fb15k dataset:

    cd ./tests
    python trans_e.py

# Speedup Comparison
The `./tests/comparison` compares the speedup with TorchKGE. To run comparison, do the following:

1. `pip install torchkge`
2. `cd ./tests/comparison`
3. `python trans_e_fastkg.py`
4. `python trans_e_torchkge.py`
