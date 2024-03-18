import numpy as np
import xarray as xr
import pint

from numpy.lib.stride_tricks import as_strided

from funcs import *

class Units():
    """
    Designed to be a parent class. Initializes the pint unit registry and sets some constants.
    ...
    Attributes
    ----------
    unr : pint.UnitRegistry()
        initializes the unit registry across child classes
    quant : pint.UnitRegistry().Quantity
        used to set unit values throughotu child classes
    K : pint.UnitRegistry().Quantity
        value of the dispersion constant in units of DM / (MHz^2 * second)
    DM_UNIT : pint.UnitRegistry().Quantity
        sets the DM unit to parsec / (cm^3)
    """
    unr = pint.UnitRegistry()
    quant = unr.Quantity
    K = 2.41e-4 * (quant(1, 'parsec') / quant(1, 'cm')**3) * 1/(quant(1, 'MHz')**2 * quant(1, 's'))
    DM_UNIT = quant(1, 'parsec') / quant(1, 'cm')**3

class Meta(Units):
    """
    Designed to be a parent class. Loads dynamic spectrum metadata from a parameter file following .fits file standard.
    ...
    Attributes
    ----------
    FREQ_UNIT : pint.UnitRegistry().Quantity
        frequency units - typically MHz
    OBSCF : pint.UnitRegistry().Quantity
        center frequency in FREQ_UNITS
    OBSBW : pint.UnitRegistry().Quantity
        total bandwidth in FREQ_UNITS
    CHAN_BW: pint.UnitRegistry().Quantity
        bandwidth of individual frequency chanels in FREQ_UNITS
    NCHAN: int
        number of frequency channels
    FREQ_LOWER: pint.UnitRegistry().Quantity
        lowest frequency of dynamic spectrum in FREQ_UNITS
    FREQ_UPPER: pint.UnitRegistry().Quantity
        highest frequency of dynamic spectrum in FREQ_UNITS

    TIME_UNIT: pint.UnitRegistry().Quantity
        time units - typically seconds
    NSBLK: int
        length of each data row in .fits file format - multiplies NROWS to give total time samples
    NROWS: int
        number of rows in .fits file format - multiplies NSBLK to give total time samples
    TBIN: pint.UnitRegistry().Quantity
        sample rate of data in TIME_UNITS - time bin size
    NSAMP: int
        number of time samples in dynamic spectrum - NSBLK * NROWS
    TIME_LOWER: pint.UnitRegistry().Quantity
        start time in TIME_UNITS - defalt is zero
    TIME_UPPER: pint.UnitRegistry().Quantity
        end time in TIME_UNITS - NSAMP * TBIN
    """
    def __init__(self, params:dict):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        """

        # frequency information
        self.FREQ_UNIT = self.quant(1, params['TUNIT13'])
        self.OBSCF = params['OBSFREQ'] * self.FREQ_UNIT
        self.OBSBW = params['OBSBW'] * self.FREQ_UNIT    
        self.CHAN_BW = params['CHAN_BW'] * self.FREQ_UNIT

        self.NCHAN = int(params['NCHAN'])
        
        self.FREQ_LOWER = self.OBSCF - .5*self.OBSBW
        self.FREQ_UPPER = self.OBSCF + .5*self.OBSBW
        
        # time information
        self.TIME_UNIT = self.quant(1, params['TUNIT1'])
        self.NSBLK = params['NSBLK']
        self.NROWS = params['NAXIS2']
        self.TBIN = params['TBIN'] * self.TIME_UNIT

        self.NSAMP = self.NSBLK*self.NROWS  

        self.TIME_LOWER = 0
        self.TIME_UPPER = self.NSAMP*self.TBIN

class DynamicFrame(Meta):
    """
    Designed to be a parent class. Loads dynamic spectrum metadata from a parameter file following .fits file standard.
    ...
    Attributes
    ----------
    sig_noise : float
        standard deviation of gaussian noise for dynamic spectrum
    time_vec : xr.DataArray
        vector from TIME_LOWER to TIME_UPPER at the sample rate (NSAMP) in TIME_UNITS
    freq_vec : xr.DataArray
        vector from FREQ_LOWER to FREQ_UPPER over the number of frequency channels (NCHAN) in FREQ_UNITS
    coords : dict
        sets labels for time and frequency coordinates for dynamic spectrum
    units: dict
        sets units for time and frequency coordinates for dynamic spectrum
    dynamic_frame: xr.DataArray
        dynamic spectrum composed of the time and frequency vectors with intensity uniformly 0.0
    ...
    Methods
    -------
    dt_max(DM)
        returns the maximum time delay in the dynamic spectrum for a given dispersion measure (DM)
    """
    sig_noise = 1
    def __init__(self, params:dict):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        """
        Meta.__init__(self, params)

        self.time_vec = xr.DataArray(np.linspace(self.TIME_LOWER,self.TIME_UPPER,self.NSAMP), dims='time')
        self.freq_vec = xr.DataArray(np.linspace(self.FREQ_LOWER,self.FREQ_UPPER,self.NCHAN), dims='freq')

        self.coords = {'time':self.time_vec,'freq':self.freq_vec}
        self.units = {'time':str(self.TIME_UNIT.units),'freq':str(self.FREQ_UNIT.units)}
        
        self.dynamic_frame = xr.DataArray(name='intensity',
                                          data=0.0,
                                          dims=list(self.coords.keys()),
                                          coords=self.coords).pint.quantify(self.units)
        
    def dt_max(self, DM):
        """
        Returns the maximum time delay (in seconds) in the dynamic spectrum for a given dispersion measure (DM)
        Parameters
        ----------
        DM: pint.UnitRegistry().Quantity
            dispersion measure in DM_UNITS
        Return
        ----------
        dt_max: pint.UnitRegistry().Quantity
            the maximum time delay in the lowest frequency channel of the dynamic spectrum in TIME_UNIT convention
        """
        dt_max = DM/self.K*((self.FREQ_LOWER)**-2 - (self.FREQ_UPPER)**-2)
        return dt_max
        
class DynamicBackground(DynamicFrame):
    """
    Designed to be a parent class. Sets the dynamic spectrum background (dynamic_frame) to white noise or loads in real data.
    """
    def __init__(self, params:dict, background_spectrum:False):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        background_spectrum : xr.DataArray
            real dynamic spectrum data to be added (default is False)
        """
        DynamicFrame.__init__(self, params)
        
        if background_spectrum:
            self.dynamic_frame.data = background_spectrum

        else:
            self.dynamic_frame += np.random.normal(loc=0, scale=self.sig_noise, size=self.dynamic_frame.shape)

class DynamicSpectrum(DynamicBackground):
    """
    Designed to be a child class. Is the final version of the dynamic spectrum.
    ...
    Attributes
    ----------
    frame_DM: pint.UnitRegistry()
        current amount the frame has been dispersed by relative to normal (default is 0)
    ...
    Methods
    -------
    disperse(DM)
        disperses the dynamic spectrum to the given dm - must be input according to DM_UNIT convention
    """
    def __init__(self, params:dict, spectrum_input:False):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        spectrum_input : xr.DataArray
            real dynamic spectrum data to be added (default is False)
        """
        DynamicBackground.__init__(self, params=params, background_spectrum=spectrum_input)
        self.frame_DM = 0 * self.DM_UNIT

    def disperse(self, dm):
        """
        Disperses the entire dynamic spectrum and saves new relative DM to frame_DM
        ...
        Parameters
        ----------
        dm : float
            dispersion measure to disperse the dynamic spectrum by - must be consistent with DM_UNIT convention
        ...
        NOTE: this does not currently use FFT rolling methods --> this should be improved in an update at some point
        """
        DM = dm * self.DM_UNIT
        dt_vec = DM/self.K*((self.freq_vec)**-2 - (self.FREQ_UPPER)**-2) # dt_vec gives the time delay corresponding to given dm along the frequency vector
        dn_vec = (dt_vec / self.TBIN).astype('int') # dn_vec is a transformtion of dt_vec into index units (int)

        arr = self.dynamic_frame.data.T # arr is the dynamic spectrum intensity values turned into a np.ndarray and transposed
        arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy() # arr_roll turnes the array into rollers (i wrote this a long time ago & don't understand it completely anymore) - need `copy`
        strd_0, strd_1 = arr_roll.strides # strides of an array tells us how many bytes we have to skip in memory to move to the next position along a certain axis
        
        dispersed = as_strided(arr_roll, (*arr.shape, arr.shape[1]), (strd_0 ,strd_1, strd_1))[np.arange(arr.shape[0]), (arr.shape[1]-dn_vec)%arr.shape[1]] # as_strided manipulates the internal data structure of ndarray according to strides

        self.dynamic_frame = xr.DataArray(data=dispersed.T, coords=self.dynamic_frame.coords, dims=self.dynamic_frame.dims) # take the transpose of the dispersed array and insert it back into the dynamic spectrum

        self.frame_DM = self.frame_DM + DM # add the new dispersion value to the prior frame_DM to give the current amount of dispersion applied to the dynamic spectrum
