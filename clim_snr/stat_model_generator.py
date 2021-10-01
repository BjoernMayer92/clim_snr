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



def gen_ensemble_no_noise(n_samples, n_ensemble_members, mean_state = 0, sigma_s = 1., sample_dim = "sample", ensemble_dim ="ensemble"):
    """Generates an Ensemble forecast without Noise (every ensemble is equal to the truth)

    Args:
        n_samples (integer): Number of samples generated
        n_ensemble_members (integer): Number of ensemble members
        mean_state (int, optional): Mean state of the system. Defaults to 0.
        sigma_s (float, optional): Standard deviation of the signal. Defaults to 1.
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".

    Returns:
        [xarray Dataset]: Forecast Experiment
    """
    
    truth = np.random.normal(loc = mean_state, scale = sigma_s, size = n_samples)


    ensemble = [truth for member in range(n_ensemble_members)]
    ensemble = np.stack(ensemble)
    
    reference = xr.DataArray(truth, dims = [sample_dim], coords = {sample_dim: range(n_samples)} )
    forecast = xr.DataArray(ensemble, dims = [ensemble_dim, sample_dim], coords = {sample_dim: range(n_samples), ensemble_dim:range(n_ensemble_members)})
    
    forecast  = forecast.rename("forecast")
    reference = reference.rename("reference")

    return xr.merge([forecast, reference])






def cal_equidistant_increment_for_variance(var, n_ensemble_members):
    """ Calculates the increment for an even sized ensemble with a given variance

    Args:
        var (float): Variance that the final ensemble should have
        n_ensemble_members (integer): Number of ensemble members for the given ensemble. Must be even

    Returns:
        [float]: Increment for an theoretical ensemble with zero mean and variance given by the input variance
    """

    assert n_ensemble_members % 2 == 0, "Number of ensemble members must be even"
    return np.sqrt(var* 6/((n_ensemble_members/2+1)*(n_ensemble_members+1)))

def gen_equidistant_values(increment, n_ensemble_members):
    """ Generates equidistant values for a theoretical ensemble given an increment. 

    Args:
        increment (float): Increment as the distance between to ensemble members
        n_ensemble_members (integer): number of ensemble members

    Returns:
        [float list]: Values for the ensemble members
    """

    assert n_ensemble_members % 2 == 0, "Number of ensemble members must be even"
    
    pos_integers = np.arange(-n_ensemble_members/2,0)
    neg_integers = np.arange(1, n_ensemble_members/2+1)

    integer_list = np.concatenate([neg_integers,pos_integers])

    values = [i*increment for i in integer_list]
    return values

def gen_equidistant_ensemble(var, n_ensemble_members):
    """ Generates equidistant ensemble members for a target variance

    Args:
        var (float): Target variance for the ensemble
        n_ensemble_members (integer): Number of ensemble members

    Returns:
        [float list]: Values for all ensemble members
    """
    assert n_ensemble_members % 2 == 0, "Number of ensemble members must be even"
    

    increment = cal_equidistant_increment_for_variance(var, n_ensemble_members)
    values = gen_equidistant_values(increment, n_ensemble_members)
    return values


def gen_ensemble_no_signal(n_samples, n_ensemble_members, mean_state = 0, noise_variance = 1, sample_dim = "sample", ensemble_dim ="ensemble"):
    """Generates an Ensemble forecast experiment without a Signal

    Args:
        n_samples (integer): Number of samples generated
        n_ensemble_members (integer): Number of ensemble members
        mean_state (int, optional): Mean state of the system. Defaults to 0.
        noise_variance (float, optional): Variance of the ensemble forecast. Defaults to 1.
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".
        ensemble_dim (str, optional): Name of the ensemble dimension. Defaults to "ensemble".

    Returns:
        [xarray Dataset]: Forecast Experiment
    """

    assert n_ensemble_members %2 == 0, "Number of Ensemble Members must be even for the no signal case"    
    
    truth = np.ones(n_samples)*mean_state

    ensemble = mean_state + np.array(gen_equidistant_ensemble(var = noise_variance, n_ensemble_members = n_ensemble_members))

    ensemble = [ensemble for sample in range(n_samples)]

    ensemble = np.stack(ensemble)
    
    reference = xr.DataArray(truth, dims = [sample_dim], coords = {sample_dim: range(n_samples)} )
    forecast = xr.DataArray(ensemble, dims = [sample_dim, ensemble_dim], coords = {sample_dim: range(n_samples), ensemble_dim:range(n_ensemble_members)})

    reference = reference.rename("reference")
    forecast  = forecast.rename("forecast")
    return xr.merge([forecast, reference])

