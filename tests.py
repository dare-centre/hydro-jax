import unittest
import numpy as np
import jax.numpy as jnp
import os
import pandas as pd

# Numpy implementation from RRMPG
from RRMPG_master.rrmpg.models.gr4j import GR4J
# jax implementation
from gr4j_jax import run_gr4j_jax

# tvp implementation
from tvp_gr4j_main.gr4j_jax import run_gr4j as run_gr4j_tvp

from abc import abstractmethod

###############################################################################
###############################################################################

class TestGR4J_base(unittest.TestCase):
    """
    This is a class to wrap around the different implementations of GR4J
    and test these. The tests are based on the RRMPG implementation and 
    relies on this package having beein downloaded for access to the data
    from the reference excel implmentation.

    Base class to be extended for each implementation, this provides the
    data and parameters to be run.
    """
    
    def __init__(self):
        # parameters are taken from the excel sheet
        self.params = {
            'X1': np.exp(5.76865628090826),
            'X2': np.sinh(1.61742503661094),
            'X3': np.exp(4.24316129943456),
            'X4': np.exp(-0.117506799276908)+0.5
        }
        self.zeros_data = {
            'prec': np.zeros(100, dtype=np.float64),
            'etp': np.random.uniform(0,3,100).astype(np.float64),
        }
        self.zeros_init = {
            's_init': np.float64(0.0),
            'r_init': np.float64(0.0)
        }
        # test on rrmpg excel data
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(
            test_dir, 'RRMPG_master',
            'test','data',
            'gr4j_example_data.csv'
        )
        data = pd.read_csv(data_file, sep=',')

        self.excel_data = {
            'prec': data.prec.values.astype(np.float64),
            'etp': data.etp.values.astype(np.float64)
        }
        self.excel_init = {
            's_init': np.float64(0.6),
            'r_init': np.float64(0.7)
        }
        self.excel_qsim = data.qsim_excel
    
    @abstractmethod
    def test_simulate_zero_rain(self):
        ''' Placeholder function for the zero data'''
        raise ImportError(
            f"{type(self).__name__}: test_simulate_zero_rain function not implemented"
        )

    @abstractmethod
    def test_simulate_compare_against_excel(self):
        ''' Placeholder function for the excel data'''
        raise ImportError(
            f"{type(self).__name__}: test_simulate_compare_against_excel function not implemented"
        )

###############################################################################
###############################################################################
  

class TestGR4J_jax(TestGR4J_base):
    """
    Test the jax implementation of the GR4J Model.
    https://github.com/dare-centre/hydro-jax
    """
    
    def __init__(self):
        super().__init__()
        self.model = run_gr4j_jax

    def test_simulate_zero_rain(self):
        params_in = {**self.params, **self.zeros_init}
        covariates = jnp.stack([
            self.zeros_data['prec'],
            self.zeros_data['etp']
        ], axis=1)
        qsim, _, _ = self.model(
            covariates,
            params_in
            )
        # self.assertEqual(np.sum(np.asarray(qsim)), 0)
        return np.sum(np.asarray(qsim)) == 0, np.sum(np.asarray(qsim))
        
    def test_simulate_compare_against_excel(self):
        params_in = {**self.params, **self.excel_init}
        covariates = jnp.stack([
            self.excel_data['prec'],
            self.excel_data['etp']
        ], axis=1)
        qsim, _, _ = self.model(
            covariates,
            params_in
            )
            
        # self.assertTrue(np.allclose(np.asarray(qsim).flatten(), data.qsim_excel))
        return qsim, np.allclose(np.asarray(qsim).flatten(), self.excel_qsim), np.asarray(qsim).flatten() - self.excel_qsim

###############################################################################
###############################################################################

class TestGR4J_np(TestGR4J_base):
    """
    Test the numpy implementation of the GR4J Model from RRMPG.
    https://github.com/kratzert/RRMPG
    """
    
    def __init__(self):
        super().__init__()
        # convert params to lower case for rrmpg and init model
        self.params = {_.lower(): v for _, v in self.params.items()}
        self.model = GR4J(params=self.params)

    def test_simulate_zero_rain(self):
        qsim = self.model.simulate(
            self.zeros_data['prec'],
            self.zeros_data['etp'],
            self.zeros_init['s_init'],
            self.zeros_init['r_init']
        )
        # self.assertEqual(np.sum(np.asarray(qsim)), 0)
        return np.sum(np.asarray(qsim)) == 0, np.sum(np.asarray(qsim))
        
    def test_simulate_compare_against_excel(self):
        qsim = self.model.simulate(
            self.excel_data['prec'],
            self.excel_data['etp'],
            self.excel_init['s_init'],
            self.excel_init['r_init'],
        )

        # self.assertTrue(np.allclose(np.asarray(qsim).flatten(), data.qsim_excel))
        return qsim, np.allclose(np.asarray(qsim).flatten(), self.excel_qsim), np.asarray(qsim).flatten() - self.excel_qsim

###############################################################################
###############################################################################
