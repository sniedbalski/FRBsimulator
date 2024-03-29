import numpy as np
import xarray as xr 

from numpy.lib.stride_tricks import as_strided

from funcs import *
from unitClasses import *
from realClasses import *

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

    STDV_NOISE : float
        standard deviation of gaussian noise for dynamic spectrum
    """
    def __init__(self, params:dict):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        """

        self.params = params

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

        self.STDV_NOISE = params['STDV_NOISE']

class DynamicFrame(Meta):
    """
    Designed to be a parent class. Loads dynamic spectrum metadata from a parameter file following .fits file standard.
    ...
    Attributes
    ----------
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
   
    """
    def __init__(self, params:dict, dynamic_input=0.0):
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
                                          data=dynamic_input,
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
        
class DynamicSpectrum(DynamicFrame):
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
    dt_max(DM)
        returns the maximum time delay in the dynamic spectrum for a given dispersion measure (DM)
    disperse(DM)
        disperses the dynamic spectrum to the given dm - must be input according to DM_UNIT convention
    """
    def __init__(self, params:dict, dynamic_input=0):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        spectrum_input : xr.DataArray
            real dynamic spectrum data to be added (default is False)
        """
        DynamicFrame.__init__(self, params=params, dynamic_input=dynamic_input)
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
