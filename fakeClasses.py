import numpy as np
import xarray as xr 

from numpy.lib.stride_tricks import as_strided

from funcs import *
from unitClasses import *
from realClasses import *

class FakeMeta(Units):
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

class FakeData(FakeMeta):
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
    def __init__(self, params:dict):
        """
        Parameters
        ----------
        params : dict
            dictionary including metadata about the dynamic spectrum parameters
        """
        FakeMeta.__init__(self, params)

        self.time_vec = xr.DataArray(np.linspace(self.TIME_LOWER,self.TIME_UPPER,self.NSAMP), dims='time')
        self.freq_vec = xr.DataArray(np.linspace(self.FREQ_LOWER,self.FREQ_UPPER,self.NCHAN), dims='freq')

        self.coords = {'time':self.time_vec,'freq':self.freq_vec}
        self.units = {'time':str(self.TIME_UNIT.units),'freq':str(self.FREQ_UNIT.units)}
        
        self.dynamic_frame = xr.DataArray(name='intensity',
                                          data=np.random.normal(loc=0, scale=self.STDV_NOISE, size=self.dynamic_frame.shape),
                                          dims=list(self.coords.keys()),
                                          coords=self.coords).pint.quantify(self.units)
        
class FakeExtracted(FakeData):
    def __init__(self, params:str):
        FakeData.__init__(self,params=params)

        self.data = self.dynamic_frame.data