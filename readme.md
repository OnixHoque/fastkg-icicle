# FastKG

## Installation
    
To install, first clone the repository and move into the folder `fastkg-icicle`. Then run: `pip install -e .`

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

# Custom Dataset
FastKG has in-built FB15k and PPOD dataset. To use your own dataset, modify line 35 of the `./tests/trans_e.py`. Replace 
    
    df = load_fb15k('train')

with the following snippet:

    from fastkg.utils import read_csv
    df = read_csv('your/tsv/train/file')

The tsv train file should have three columns separated by `\t` and sequenced in head-rel-tail order.
