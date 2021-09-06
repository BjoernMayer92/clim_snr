import numpy as np
import xarray as xr


def calc_noise_var(ensemble, ensemble_dim = "ensemble", sample_dim = "sample"):
    """Calculates the noise variance of a given ensemble forecast

    Args:
        ensemble (xarray_obj): Xarray object over which the signal variance is calculated. Must have at least the dimensions defined by ensemble_dim and sample_dim
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".

    Returns:
        xarray_obj: Xarray object containing the noise variance. Array or Dataset has lost ensemble and sample dimension 
    """


    ensemble_mean = ensemble.mean(dim=ensemble_dim)
    ensemble_clim = ensemble.mean(dim=(ensemble_dim, sample_dim))


    noise = xr.ufuncs.square(ensemble - ensemble_mean).mean(dim=(sample_dim, ensemble_dim), skipna=True)

    return noise


def calc_total_var(ensemble, ensemble_dim, sample_dim):
    """Calculates the total variance of a given ensemble forecast

    Args:
        ensemble (xarray_obj): Xarray object over which the signal variance is calculated. Must have at least the dimensions defined by ensemble_dim and sample_dim
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".

    Returns:
        xarray_obj: Xarray object containing the signal signal. Array or Dataset has lost ensemble and sample dimension 
    """
    ensemble_clim = ensemble.mean(dim=(ensemble_dim, sample_dim))
    total_var = xr.ufuncs.square(ensemble - ensemble_clim).mean(dim=(ensemble_dim, sample_dim))

    return total_var


def calc_signal_var(ensemble, ensemble_dim = "ensemble", sample_dim ="sample"):
    """Calculates the signal variance of a given ensemble forecast

    Args:
        ensemble (xarray_obj): Xarray object over which the signal variance is calculated. Must have at least the dimensions defined by ensemble_dim and sample_dim
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".

    Returns:
        xarray_obj: Xarray object containing the total variance. Array or Dataset has lost ensemble and sample dimension 
    """


    n_ensemble_members = ensemble.count(dim=ensemble_dim)
    ensemble_clim = ensemble.mean(dim=(ensemble_dim, sample_dim))
    residual_anomalies_squared = xr.ufuncs.square(ensemble_mean - ensemble_clim )
    n_total = n_ensemble_members.sum(dim = sample_dim)

    signal = (n_ensemble_members * residual_anomalies_squared).sum(dim=sample_dim)/n_total

    return signal 

    
def calc_sn_decomposition(ensemble, ensemble_dim="ensemble", sample_dim="sample"):
    """Calculates the signal and noise decomposition variance of a given ensemble forecast

    Args:
        ensemble (xarray_obj): Xarray object over which the signal variance is calculated. Must have at least the dimensions defined by ensemble_dim and sample_dim
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".

    Returns:
        xarray_obj: Xarray object containing the signal. Array or Dataset has lost ensemble and sample dimension 
    """
    assert ensemble_dim in ensemble.dims
    assert sample_dim in ensemble.dims
    
    noise_var = calc_noise_var(ensemble, ensemble_dim = ensemble_dim, sample_dim = sample_dim)
    signal_var = calc_signal_var(ensemble, ensemble_dim = ensemble_dim, sample_dim = sample_dim)
    total_var = calc_total_var(ensemble, ensemble_dim = ensemble_dim, sample_dim = sample_dim)

    noise_var = noise_var.rename("noise_var")
    signal_var = signal_var.rename("signal_var")
    total_var = total_var.rename("total_var")

    xr.testing.assert_allclose(snr_decomposition["noise_var"] + snr_decomposition["signal_var"],snr_decomposition["total_var"])

    return xr.merge([noise_var, signal_var, total_var])