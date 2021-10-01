import unittest
import os
import sys
import numpy as np
import clim_snr.stat
import clim_snr.stat_model_generator

absolute_tolerance=10**(-7)


class TestStat(unittest.TestCase):
    """Test Class for Siegert Data generation

    """
    @classmethod
    def setUpClass(self):
        """
        Generates edge cases for testing the snr functions
        """

        self.no_signal_ensemble_variance = np.random.uniform(0,10)

        self.experiment_no_signal = clim_snr.stat_model_generator.gen_ensemble_no_signal(n_samples = 1000, n_ensemble_members = 100, noise_variance=self.no_signal_ensemble_variance)
        self.experiment_no_noise  = clim_snr.stat_model_generator.gen_ensemble_no_noise(n_samples = 1000, n_ensemble_members = 100 )

        self.experiment_no_signal_signal = clim_snr.stat.calc_signal_var(self.experiment_no_signal["forecast"]) 
        self.experiment_no_signal_noise  = clim_snr.stat.calc_noise_var(self.experiment_no_signal["forecast"])
        self.experiment_no_signal_total  = clim_snr.stat.calc_total_var(self.experiment_no_signal["forecast"])

        self.experiment_no_noise_signal = clim_snr.stat.calc_signal_var(self.experiment_no_noise["forecast"]) 
        self.experiment_no_noise_noise  = clim_snr.stat.calc_noise_var(self.experiment_no_noise["forecast"])
        self.experiment_no_noise_total  = clim_snr.stat.calc_total_var(self.experiment_no_noise["forecast"])

    def test_calc_noise_var(self):
        np.testing.assert_allclose(self.experiment_no_noise_noise, 0., atol = absolute_tolerance)
        np.testing.assert_allclose(self.experiment_no_signal_noise,self.no_signal_ensemble_variance, atol=absolute_tolerance)
 

    def test_calc_signal_var(self):
        np.testing.assert_allclose(self.experiment_no_signal_signal,0.0, atol = absolute_tolerance)
        
    def test_calc_total_var(self):
        np.testing.assert_allclose(self.experiment_no_signal_signal + self.experiment_no_signal_noise, self.experiment_no_signal_total, atol = absolute_tolerance)
        np.testing.assert_allclose(self.experiment_no_noise_signal + self.experiment_no_noise_noise, self.experiment_no_noise_total, atol = absolute_tolerance)
      



if __name__ == '__main__':
    unittest.main()