import numpy as np
import xarray as xr
import pint
import pint_xarray
from scipy import special

from backClasses import *
from funcs import *

class MetaFRB(Units):
    """
    Designed to be a parent class. Loads FRB metadata from a parameter file.
    ...
    Attributes
    ----------
    DM: pint.UnitRegistry().Quantity
        dispersion measure given in DM_UNIT standard
    SNR: float
        gives the signal-to-noise ratio (SNR) of the burst to underlying noise when averaged over all frequency channels - assumes properly dedispersed spectrum (no delay time in burst arrival time)
    MU_TIME: pint.UnitRegistry().Quantity
        arrival time of the burst - mean of the gaussian burst in time using TIME_UNIT
    MU_FREQ: pint.UnitRegistry().Quantity
        center of the burst in frequency using FREQ_UNIT
    SIG_TIME: pint.UnitRegistry().Quantity
        standard deviation of the burst in time using TIME_UNIT convention - pulse width
    SIG_FREQ: pint.UnitRegistry().Quantity
        standard deviation of the burst in frequency using FREQ_UNIT convention - frequency spread
    """
    def __init__(self, params:dict):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about a gaussian FRB
        """
        TIME_UNIT = self.quant(1, params['TIME_UNIT']) # time units - typically ms
        FREQ_UNIT = self.quant(1, params['FREQ_UNIT']) # frequency units - typically MHz

        self.DM = params['DM'] * self.DM_UNIT
        self.SNR = params['SNR']        
        self.MU_TIME = params['MU_TIME'] * TIME_UNIT
        self.MU_FREQ = params['MU_FREQ'] * FREQ_UNIT
        self.SIG_TIME = params['SIG_TIME'] * TIME_UNIT
        self.SIG_FREQ = params['SIG_FREQ'] * FREQ_UNIT

class InsertFRB(MetaFRB, DynamicFrame):
    """
    Class called to insert a generated FRB into a dynamic spectrum. The dynamic spectrum is built according the the DynamicFrame parent class.
    ...
    Attributes
    ----------
    burst: xr.DataArray
        the dispersed gaussian FRB within the subframe
    """
    def __init__(self, instrument_params:dict, burst_params:dict):
        """
        Parameters
        ----------
        instrument_params : dict
            dictionary including metadata about the dynamic spectrum parameters
        burst_params: dict
            dictionary including metadata about a gaussian FRB
        """
        MetaFRB.__init__(self, params=burst_params)
        DynamicFrame.__init__(self, params=instrument_params)
    
        dt_max = self.dt_max(DM=self.DM)

        subframe = self.dynamic_frame.copy().pint.sel(time=slice(self.MU_TIME-5*self.SIG_TIME,self.MU_TIME+5*self.SIG_TIME+dt_max), #  subsection of the dynamic spectrum - given by the FRB mean in time and frequency plus five standard deviations along both axes plus the maximum time delay for the FRB DM
                                                      freq=slice(self.MU_FREQ-5*self.SIG_FREQ,self.MU_FREQ+5*self.SIG_FREQ))

        t, f = np.meshgrid(subframe.coords['time'], subframe.coords['freq'])
        time_mesh = subframe.copy(data=t.T) * self.TIME_UNIT # gives the position along the time vector as a function of time and frequency
        freq_mesh = subframe.copy(data=f.T) * self.FREQ_UNIT # gives the position along the frequency vector as a function of time and frequency
        shft_mesh = (self.DM/self.K)/(freq_mesh)**2 # gives the time delay caused by dispersion as a function of time and frequency

        erf_d = erf_diff(freq_top=self.FREQ_UPPER, # calculates the difference between error functions - integral of a gaussian with upper and lower bounds (needed to compute the burst amplitude)
                         freq_bot=self.FREQ_LOWER,
                         mu_f=self.MU_FREQ,
                         sig_f=self.SIG_FREQ)
        
        burst_amp = np.sqrt((2*self.NCHAN)/np.pi) * (self.CHAN_BW * self.SNR) * (self.sig_noise/self.SIG_FREQ) * erf_d**-1 # amplitude of the gaussian FRB needed to produce the appropriate SNR value
        
        shift_t = (self.DM/self.K)/(self.FREQ_UPPER)**2 # the time delay in the burst at the highest frequency channel relative to 'infinite' frequency

        # the following values (gamma_f/t & comp_1-5) are subparts of the full function describing a dispersed gaussion FRB
        gamma_f = -.5*(1/self.SIG_FREQ)**2
        gamma_t = -.5*(1/self.SIG_TIME)**2
        comp_1 = (freq_mesh-self.MU_FREQ)**2
        comp_2 = (shft_mesh**2+2*shft_mesh*(self.MU_TIME-shift_t))
        comp_3 = (-2*time_mesh*shft_mesh)
        comp_4 = (time_mesh**2+time_mesh*(2*shift_t-2*self.MU_TIME))
        comp_5 = (shift_t-self.MU_TIME)**2
        
        self.burst = xr.DataArray(data=(burst_amp * np.exp(gamma_f*comp_1 + gamma_t*(comp_2+comp_3+comp_4+comp_5))),
                                  coords=subframe.coords,
                                  dims=subframe.dims).reindex_like(self.dynamic_frame, fill_value=0)

class SimulateSadTrombone(DynamicSpectrum):
    """
    Child class of DynamicSpectrum with a method to inject FRBs using InsertFRB.
    ...
    Attributes
    ----------
    down_drift_prob: float
        the probability that the sad trombone islands will decrease in frequency with time.
    Methods
    -------
    inject_frb(params)
        inserts a generated FRB into the dynamic spectrum using the InsertFRB class.
    ...
    NOTE: this class is a mess, but I don't currently have time to fix it.
    Currently, I am setting the distributions to set values and ranges.
    It would be nice to have a better way to determine the pdf's used to generate variations in bursts within a 'sad trombone' cluster.
    """
    down_drift_prob = 0.95
    def __init__(self, params:dict, spectrum_input:False):
        DynamicSpectrum.__init__(self, params=params, spectrum_input=spectrum_input)

        DM_0 = 500 * self.DM_UNIT
        SNR_MAX = 20

        drift_sign = np.random.choice([-1,1], p=[self.down_drift_prob, 1-self.down_drift_prob])
        n_islands = np.random.randint(low=1, high=6)
        delta_freq = drift_sign * np.random.random(size=n_islands) * self.quant(100, 'MHz')
        delta_time = np.random.chisquare(df=3, size=n_islands)/3 * self.quant(10, 'ms')
        delta_dm = (np.random.random(size=n_islands) - .5) * 20 * self.quant(1, 'parsec') / self.quant(1, 'cm')**3
        snr_vec = np.random.random(size=n_islands) * SNR_MAX

        Delta_freq = np.sum(delta_freq)
        Delta_time = np.sum(delta_time)+self.dt_max(DM=DM_0)

        sig_time_vec = np.random.chisquare(df=3, size=n_islands)/3 * (self.quant(3, 'ms')/self.quant(1,'sec')).to_base_units().magnitude
        sig_freq_vec = np.random.chisquare(df=3, size=n_islands)/3 * (self.quant(75, 'MHz')/self.quant(1,'MHz')).to_base_units().magnitude

        mu_time_0 = np.random.random(size=1) * (self.TIME_UPPER-Delta_time).to_base_units()
        if drift_sign > 0:
            mu_freq_0 = (self.FREQ_UPPER-Delta_freq).to_base_units()
        else:
            mu_freq_0 = (self.FREQ_LOWER-Delta_freq).to_base_units()

        dm_vec = ((DM_0+delta_dm)/self.DM_UNIT).to_base_units().magnitude
        mu_time_vec = ((mu_time_0 + delta_time)/self.quant(1,'sec')).to_base_units().magnitude
        mu_freq_vec = ((mu_freq_0 + delta_freq)/self.quant(1,'MHz')).to_base_units().magnitude
        
        for n in range(n_islands):
            print(f'Inserting burst {n+1} of {n_islands+1}')
            params = {'TIME_UNIT':'sec', 'FREQ_UNIT':'MHz', 'DM':dm_vec[n], 'SNR':snr_vec[n], 'MU_TIME':mu_time_vec[n], 'MU_FREQ':mu_freq_vec[n], 'SIG_TIME':sig_time_vec[n], 'SIG_FREQ':sig_freq_vec[n]}
            print(params)
            self.inject_frb(params=params)

    def inject_frb(self, params):
        """
        Inject a FRB into the dynamic spectrum.
        Parameters
        ----------
        params: dict
            dictionary including metadata about a gaussian FRB
        """
        injection = InsertFRB(instrument_params=self.params, burst_params=params)
        self.dynamic_frame = self.dynamic_frame + injection.burst