from astropy.io import fits
import numpy as np
import xarray as xr

from unitClasses import *
from funcs import *

class FitsMeta(Units):
    def __init__(self, path:str):
        # separate file path and file name
        self.path = path
        self.name = self.path.split("/")[-1]

        # open headers
        self.head_0 = fits.open(self.path)[0].header
        self.head_1 = fits.open(self.path)[1].header
        
        # frequency information
        self.FREQ_UNIT = self.quant(1, self.head_1['TUNIT13'])
        self.OBSCF = self.head_0['OBSFREQ'] * self.FREQ_UNIT
        self.OBSBW = self.head_0['OBSBW'] * self.FREQ_UNIT   
        self.CHAN_BW = self.head_1['CHAN_BW'] * self.FREQ_UNIT
        self.NCHAN = int(self.head_1['NCHAN'])
        
        self.FREQ_LOWER = self.OBSCF - .5*self.OBSBW
        self.FREQ_UPPER = self.OBSCF + .5*self.OBSBW
        
        # time information
        self.TIME_UNIT = self.quant(1, self.head_1['TUNIT1'])
        self.NSBLK = self.head_1['NSBLK']
        self.NROWS = self.head_1['NAXIS2']
        self.TBIN = self.head_1['TBIN'] * self.TIME_UNIT

        self.NSAMP = self.NSBLK*self.NROWS
        self.TIME_LOWER = 0
        self.TIME_UPPER = self.NSAMP*self.TBIN
        
class FitsData(FitsMeta):
    cutoff = .1
    def __init__(self, path:str):
        FitsMeta.__init__(self, path=path)
        
        data = fits.open(self.path)[1].data
        dynamic = np.reshape(np.array([x[-1][:,0,:,0] for x in data]),(self.NSAMP, self.NCHAN)).astype('float')
        
        self.time_vec = xr.DataArray(np.linspace(self.TIME_LOWER,self.TIME_UPPER,self.NSAMP), dims='time')
        self.freq_vec = xr.DataArray(np.linspace(self.FREQ_LOWER,self.FREQ_UPPER,self.NCHAN), dims='freq')
        
        self.coords = {'time':self.time_vec,'freq':self.freq_vec}
        self.units = {'time':str(self.TIME_UNIT.units),'freq':str(self.FREQ_UNIT.units)}
        
        self.dynamic_frame = xr.DataArray(name='intensity',
                                          data=dynamic,
                                          dims=list(self.coords.keys()),
                                          coords=self.coords).pint.quantify(self.units)
        
        self.FREQ_UPPER = self.FREQ_UPPER-self.cutoff*self.OBSBW
        self.FREQ_LOWER = self.FREQ_LOWER+self.cutoff*self.OBSBW

        self.dynamic_frame = self.dynamic_frame.pint.sel(freq=slice(self.FREQ_LOWER,
                                                                    self.FREQ_UPPER))
        
        self.spectrum = self.dynamic_frame.mean(dim='time')
        
        self.OBSBW = self.FREQ_UPPER-self.FREQ_LOWER
        self.NCHAN = self.dynamic_frame.freq.shape[0]

class CleanData(FitsData):
    def __init__(self,path:str):
        FitsData.__init__(self, path=path)

        self.spikewidth = int((self.quant(3, 'MHz')/self.quant(self.CHAN_BW, self.FREQ_UNIT)).to_base_units())

        self.flatten_spectrum()
        self.get_flags()
        self.flatten_time()
        self.cleanup()

    def smooth_spikes(self,spectrum:xr.DataArray,width:float=1):

        return spectrum.rolling(freq=width*self.spikewidth,
                                min_periods=1,
                                center=True)
    
    def get_flags(self):
        spectrum_flat = (self.spectrum-self.bandpass)/self.bandpass
        
        threshold = spectrum_flat.quantile(q=.8, dim='freq').drop('quantile')
        
        self.freq_flags = spectrum_flat<threshold

    def flatten_spectrum(self):

        smooth_spectrum = self.smooth_spikes(self.smooth_spikes(self.spectrum).min()).mean()
        
        difference = self.smooth_spikes(np.abs(smooth_spectrum-self.smooth_spikes(smooth_spectrum).mean()),width=2).mean()
        
        threshold = difference.quantile(q=.7, dim='freq').drop('quantile')

        interp_spectrum = smooth_spectrum.where(difference<threshold).interpolate_na(dim='freq',
                                                                                     method='linear',
                                                                                     fill_value='extrapolate')
        
        bandpass = np.minimum(smooth_spectrum,interp_spectrum)
        self.bandpass = bandpass.where(cond=(0.0!=bandpass), other=1e-6)
        
        self.dynamic_frame = ((self.dynamic_frame-self.bandpass)/self.bandpass)
                
    def flatten_time(self):
        freq_average = (self.dynamic_frame.where(self.freq_flags)).mean(dim='freq', skipna=True)
        
        polyfit = freq_average.polyfit(dim='time', deg=10)
        
        self.time_fit = xr.polyval(coord=freq_average.time, coeffs=polyfit.polyfit_coefficients)
        
        self.dynamic_frame = (self.dynamic_frame-self.time_fit)
        
    def cleanup(self):
        self.flat_spectrum = self.dynamic_frame.mean(dim='time')
        
        self.sig_noise = self.dynamic_frame.where(self.freq_flags).std()

class FitsExtracted(CleanData):
    def __init__(self, path:str):
        CleanData.__init__(self,path=path)

        self.params = {
            'TUNIT13':self.head_1['TUNIT13'],
            'OBSFREQ':self.head_0['OBSFREQ'],
            'OBSBW':self.OBSBW.to('MHz').magnitude,
            'CHAN_BW':self.head_1['CHAN_BW'],
            'NCHAN':self.NCHAN,
            'TUNIT1':self.head_1['TUNIT1'],
            'NSBLK':self.head_1['NSBLK'],
            'NAXIS2':self.head_1['NAXIS2'],
            'TBIN':self.head_1['TBIN'],
            'STDV_NOISE':self.sig_noise}

        self.data = self.dynamic_frame.data


