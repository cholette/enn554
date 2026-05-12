import numpy as np
import pvlib
import matplotlib.pyplot as plt
from enn554.constants import kB, electron_charge
from scipy.optimize import fsolve
tol = 1e-10

class simple_equivalent_circuit:
    def __init__(self,I0,Isc=None,Voc=None,T=25+273.15,I=1000):

        self.I0 = I0
        
        if (Isc is None) and (Voc is None):
            err = Voc- kB*T/electron_charge * np.log(Isc/I0+1)
            assert np.abs(err)<tol, "Isc and Voc both supplied and are inconsistent."
        elif Isc is None:
            Isc = self.current(Voc,T)
        elif Voc is None:
            Voc = kB*T/electron_charge * np.log(Isc/I0+1)

        self.Isc = Isc
        self.Voc = Voc

        self.T = T
        self.G = I
    
    def set_irradiance(self,I):
        
        self.Isc *= I/self.G
        self.Voc = kB*self.T/electron_charge * np.log(self.Isc/self.I0+1)
        self.G = I
    
    def current(self,V):
        return self.Isc - self.I0*(np.exp( electron_charge*V / kB / self.T) - 1)

    def plot_iv(self,ax=None,mplkwds={}):
        
        
        if ax is None:
            fig,ax = plt.subplots()

        v = np.linspace(0,self.Voc)
        ax.plot(v,self.current(v),**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')

        return ax

    def plot_power(self,ax=None,mplkwds={}):
        
        
        if ax is None:
            fig,ax = plt.subplots()

        v = np.linspace(0,self.Voc)
        ax.plot(v,self.current(v)*v,**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Power (W)')

        return ax

class series_parallel_equivalent_circuit:
    def __init__(self,I0,Rp,Rs,Isc=None,Voc=None,T=25+273.15,I=1000):
        self.I0 = I0
        self.Rp = Rp
        self.Rs = Rs
        self.T = T
        self.G = I

        assert (Isc is None) ^ (Voc is None), "Specify ONE of Isc or Voc (not both)."

        if Isc is not None:
            self.Isc = Isc
            x0 = kB*T/electron_charge * np.log(Isc/I0+1)
            Vdoc, info, ier, msg = fsolve(lambda x: self.current(x),x0=x0,full_output=True)
            self.Voc = self.voltage(Vdoc)

        elif Voc is not None:
            self.Voc = Voc
            self.Isc = self.I0*(np.exp( electron_charge*Voc / kB / self.T) - 1)  + (Voc/self.Rp)
        
    def current(self,Vd):
        return self.Isc - self.I0*(np.exp( electron_charge*Vd / kB / self.T) - 1) -(Vd/self.Rp)
    
    def voltage(self,Vd):
        return Vd - self.current(Vd)*self.Rs

    def plot_iv(self,ax=None,mplkwds={}):
        
        
        if ax is None:
            fig,ax = plt.subplots()

        vd = np.linspace(0,self.Voc)
        i,v = self.current(vd),self.voltage(vd)
        ax.plot(v,i,**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')

        return ax


    