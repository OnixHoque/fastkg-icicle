# FastKG - A sparse implementation of translational KG embedding models

## Installation
    
To install, first clone the repository and move into the folder. Then run: `pip install -e .`

# CPU/GPU Testing
To test fb15k dataset:

    cd ./tests
    python trans_e.py

# Speedup Comparison
FastKG uses SpMM implementation to speed up KG embedding trainining. The `./tests/comparison` compares the speedup with TorchKGE. To run comparison, do the following:

1. `pip install torchkge`
2. `cd ./tests/comparison`
3. `python trans_e_fastkg.py`
4. `python trans_e_torchkge.py`

Note: FastKG is typically 30% faster than TorchKGE with PyTorch SpMM. We observed up to 5x speedup when a high-performance SpMM such as iSpLib is used.