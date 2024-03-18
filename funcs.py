import numpy as np
from scipy import special

def erf_diff(freq_top, freq_bot, mu_f, sig_f):
    """
    Computes the integral boundaries for a gaussian.
    Parameters
    ----------
    freq_top : float, in MHz
        The highest frequency in the dynamic spectrum

    freq_bot : float, in MHz
        The lowest frequency in the dynamic spectrum

    mu_f : float, in MHz
        Mean frequency of gaussian burst

    sig_f : float, in MHz
        Standard deviation of gaussian burst
    """
    erf_d = special.erf(((freq_top-mu_f)/(np.sqrt(2)*sig_f)).to_base_units().magnitude) - special.erf(((freq_bot-mu_f)/(np.sqrt(2)*sig_f)).to_base_units().magnitude)
    return erf_d