import numpy as np
import xarray as xr

def siegert(n_samples, n_ensemble_members, mu_y = 0 , mu_x= 0 , beta = 1, sigma_s = 1, sigma_e = 1, sigma_eta = 1, output="xarray"):
    """Creates ensemble data with the method described in https://doi.org/10.1175/JCLI-D-15-0196.1

    Args:
        n_samples (int): Number of samples. usually the time dimension of an ensemble forecast
        n_ensemble_members (Union[int, list, numpy.ndarray]): Number of ensemble members for each sample. If it is an integer the number of ensemble members is constant. List and Arrays allow different number of ensemble members for each sample.
        mu_y (int, optional): Mean state of the truth timeseries. Defaults to 0.
        mu_x (int, optional): Mean state of the ensemble timeseries. Defaults to 0.
        beta (int, optional): Coupling parameter between truth and ensemble. Defaults to 1.
        sigma_s (int, optional): Standard deviation of the truth signal. Defaults to 1.
        sigma_e (int, optional): Standard deviation of the truth noise. Defaults to 1.
        sigma_eta (int, optional): Standard deviation of the ensemble noise. Defaults to 1.
        output (str, optional): Whether output is given as xarray or numpy array. Defaults to "xarray".

    Returns:
        [type]: [description]
    """



    if(type(n_ensemble_members)==int):
        n_ensemble_members = np.full(n_samples, n_ensemble_members)
    if(type(n_ensemble_members)==list) or (type(n_ensemble_members) == np.ndarray):
        assert n_samples == len(n_ensemble_members)


    signal_truth    = np.random.normal(loc = 0, scale = sigma_s, size = n_samples)
    signal_ensemble = beta * signal_truth

    noise_truth     = np.random.normal(loc = 0, scale = sigma_e, size = n_samples)

    truth = mu_y +  signal_truth + noise_truth
    
    noise_ensemble = np.full( [n_samples, max(n_ensemble_members)], np.nan)
    for sample in range(n_samples):
        tmp_n_ensemble = n_ensemble_members[sample]
        tmp_noise_ensemble = np.random.normal(loc = 0, scale = sigma_eta, size = [tmp_n_ensemble ])
        noise_ensemble[sample, 0 : tmp_n_ensemble ] = tmp_noise_ensemble

    ensemble = mu_x + signal_ensemble[..., None] + noise_ensemble
    
    if output=="xarray":
        truth = xr.DataArray(truth, dims = ["sample"], coords = {"sample": range(n_samples)})
        ensemble = xr.DataArray(ensemble, dims = ["sample","ensemble"], coords = {"sample": range(n_samples), "ensemble": range(max(n_ensemble_members))})

    return truth, ensemble