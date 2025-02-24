import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


class wind_distribution:
    
    def __init__(self,shape=None,scale=None,azimuth_edges=None,probability=None):
        if shape == 1:
            self.dist = stats.expon(scale=scale)
        elif shape == 2:
            self.dist = stats.rayleigh(scale=scale)
        else:
            self.dist = stats.weibull_min(shape,scale=scale)

        self.direction = {'azimuth_edges':azimuth_edges,
                          'probability':probability} # probability corresponds to bin centers and therefore must be len(azimuth_edges)-1
    
    def pdf(self,x):
        return self.dist.pdf(x)
    
    def cdf(self,x):
        return self.dist.cdf(x)
    
    def ppf(self,x):
        return self.dist.ppf(x)
    
    def rvs(self,size=1):
        return self.dist.rvs(size=size)
    
    def wind_speed_curve(self,x,hours_per_year=8760,plot=True):
        F = self.cdf(x)

        if plot:
            fig,ax = plt.subplots()
            ax.plot(F*hours_per_year,x)
            ax.set_xlabel('Hours per year')
            ax.set_ylabel('Wind speed')
        else:
            fig = None
            ax = None

        return F,fig,ax

    def fit(self,data,plot=False,type='weibull',hist_kwargs={},fit_kwargs={},handles=None):
        # fits a distribution to the wind data

        if type.lower() == 'weibull':
            self.shape,_,self.scale = stats.weibull_min.fit(data,floc=0) # don't fit location parameter
            self.dist = stats.weibull_min(self.shape,scale=self.scale)
        elif type.lower() == 'rayleigh':
            _,self.scale = stats.rayleigh.fit(data,floc=0) # don't fit location parameter
            self.dist = stats.rayleigh(scale=self.scale)
        else:
            raise ValueError('Invalid distribution type. Must be "weibull" or "rayleigh"')
        
        if plot:
            if handles is None:
                fig,ax = plt.subplots()
            else:
                fig,ax = handles
            ax.hist(data,label='Data',density=True,**hist_kwargs)
            XL = ax.get_xlim()
            x = np.linspace(XL[0],XL[1],1000)
            ax.plot(x,self.pdf(x),label=f'{type} fit',**fit_kwargs)
            ax.set_xlabel('Wind speed')
            ax.set_ylabel('Density')
            ax.legend()
            return fig,ax
        else:
            return None,None

    def scale(self):
        return self.dist.args[2]
    
    def shape(self):
        return self.dist.args[0]
    
        