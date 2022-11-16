# GR4J JAX implementation
DARE, 2022
Contact: Joshua Simmons

This repository contains a GR4J implementation in JAX.

The model is described in the following paper:


To run the tests (`01_model_test.ipynb`) you will need to have the [RRMPG repository ](https://github.com/kratzert/RRMPG) package downloaded and available in a base directory. We will use the numpy RRMPG implmenetation, test scaffold and associated data as a reference to test the JAX implementation. 

Per the [RRMPG docs](https://rrmpg.readthedocs.io/en/latest/getting_started.html#installing-the-rrmpg-package), there is no pip package avalaible, **so you will need to manually download from the [github repo](https://github.com/kratzert/RRMPG). Place the unzipped `RRMPG-master` folder in the base directory and rename `RRMPG_master`** (as python imports will complain about "-").

Otherwise feel free to use the implementation directly from the `gr4j_jax.py` file, importing the function `run_gr4j_jax`.

## run_gr4j_jax function

`run_gr4j_jax` takes as input:

    - inputs: (n_timesteps,2) a jax array with values for rain at each timestep in the first column and evaporation (at each timestep) in the second column. e.g., you could create this from numpy arrays `rain` and `et` using: `inputs = jnp.stack([rain,et],axis=1)`.
    - params_dict: an dictionary with the parameters of the model:
        - s_init: s storage init
        - r_init: r storage init
        - X1
        - X2S
        - X3
        - X4 

and outputs:

    - qsim: (n_timesteps,) a jax array with simulated discharge at each timestep
    - s: (n_timesteps,) a jax array with the s storage at each timestep
    - r: (n_timesteps,) a jax array with the r storage at each timestep


## Directory structure

- `gr4j_jax.py`: contains the JAX implementation of the GR4J model
- `01_model_test.ipynb`: notebook to test the JAX implementation using the tests from the RRMPG numpy implementation
- `tests.py`: contains the scaffold to run the tests adapted from RRMPG
- `RRMPG_master`: folder where the downloaded RRMPG package should be placed



