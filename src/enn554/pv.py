import numpy as np
import pvlib
import matplotlib.pyplot as plt
from enn554.constants import kB, electron_charge, eV2J
from scipy.optimize import fsolve, root_scalar
tol = 1e-10

class SimpleEquivalentCircuit:
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

class SeriesParallelEquivalentCircuit:
    def __init__(self,I0,Rsh,Rs,Isc=None,Voc=None,nI=1.0,Ns=1,T=25+273.15,G=1000):
        self.I0 = I0
        self.Rsh = Rsh
        self.Rs = Rs
        self.T = T
        self.G = G
        self.nI = nI
        self.a = nI*T*kB/electron_charge
        self.Ns = Ns

        assert (Isc is None) ^ (Voc is None), "Specify ONE of Isc or Voc (not both)."

        if Isc is not None:
            V0 = Isc*self.Rs
            fcn = lambda IL: Isc - (IL - self.I0*(np.exp( V0/self.a) - 1) - V0/self.Rsh)
            self.IL = fsolve(fcn,x0=Isc)[0]

            Vd0 = kB*T/electron_charge * np.log(Isc/I0+1) # for simple circuit
            Vdoc = fsolve(lambda x: self.current(Vd=x),x0=Vd0)[0]
            self.Voc = self.voltage(Vdoc)

        elif Voc is not None:
            self.Voc = Voc
            self.IL = self.I0*(np.exp( Voc / self.a) - 1)  + (Voc/self.Rsh)

    def current(self,Vd=None,V=None):
        assert (Vd is None) ^ (V is None), "Specify ONE of Vd or V (not both)."
        if V is not None:
            Vc = V/self.Ns
            fcn = lambda I: I - (self.IL - self.I0*(np.exp( (Vc+I*self.Rs) / self.a) - 1) - (Vc+I*self.Rs)/self.Rsh)
            I0 = self.IL - self.I0*(np.exp( Vc/ self.Ns /self.a) - 1) - Vc/self.Rsh
            I= fsolve(fcn,I0)[0]
        else: # Vd supplied directly
            I = self.IL - self.I0*(np.exp( Vd / self.a) - 1) -(Vd/self.Rsh)

        return I
    
    def voltage(self,Vd=None,I=None):
        assert (Vd is None) ^ (I is None), "Specify ONE of Vd or I (not both)."

        if I is not None:
            fcn = lambda V: I - (self.IL - self.I0*(np.exp( (V+I*self.Rs) / self.a) - 1) - (V+I*self.Rs)/self.Rsh)
            V = self.Ns*fsolve(fcn,x0=self.Voc/self.Ns)[0]
        else:
            V = self.Ns*(Vd - self.current(Vd)*self.Rs)

        return V

    def plot_iv(self,ax=None,mplkwds={}):
        
        if ax is None:
            fig,ax = plt.subplots()

        voltage = np.linspace(0,self.Voc,1000)
        # i,v = self.current(vd),self.voltage(vd)
        i = [self.current(V=v) for v in voltage]
        ax.plot(voltage,i,**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')

        return ax

class DeSotoModel:
    """
    De Soto single-diode PV module model.

    Reference: De Soto et al. (2006), Solar Energy, 80(1), 78-88.

    The module I-V equation is:
        I = IL - I0*(exp((V + I*Rs)/a) - 1) - (V + I*Rs)/Rsh

    Five parameters: IL, I0, a (= n*Ns*kT/q), Rs, Rsh.
    Extracted at STC from 6 datasheet values; translated to arbitrary (G, T).
    """

    def __init__(self,Ns,Isc_ref,Voc_ref,Imp_ref,Vmp_ref,
                delta_Isc,delta_Voc,Eg0=1.121,G_ref=1000.0,T_ref_C = 25.0):
        """
        Parameters
        ----------
        Ns : int
            Number of cells in series.
        Isc_ref : float
            Short-circuit current at STC [A].
        Voc_ref : float
            Open-circuit voltage at STC [V].
        Imp_ref : float
            MPP current at STC [A].
        Vmp_ref : float
            MPP voltage at STC [V].
        delta_Isc : float
            Temperature coefficient of Isc [A/K].
        delta_Voc : float
            Temperature coefficient of Voc [V/K].
        Eg0 : float
            Zero-temperature bandgap [eV]. Default: 1.121 (c-Si).
        G_ref : float
            Reference irradiance [W/m²]. Default: 1000.
        T_ref_C : float
            Reference temperature [°C]. Default: 25.
        """
        self.Ns = Ns
        self.Isc_ref = Isc_ref
        self.Voc_ref = Voc_ref
        self.Imp_ref = Imp_ref
        self.Vmp_ref = Vmp_ref
        self.mu_Isc = delta_Isc
        self.mu_Voc = delta_Voc
        self.Eg0 = Eg0
        self.G_ref = G_ref
        self.T_ref = T_ref_C + 273.15  # [K]

        self.IL_ref, self.I0_ref, self.a_ref, self.Rs_ref, self.Rsh_ref = self.fit_parameters()

    def _Eg(self, T):
        """Temperature-dependent bandgap [eV]."""
        return self.Eg0 * (1.0 - 0.0002677 * (T - self.T_ref))

    def _residuals(self,p,dT=10):
        IL, I0, a, Rs, Rsh = p
        Isc, Voc = self.Isc_ref, self.Voc_ref
        Imp, Vmp = self.Imp_ref, self.Vmp_ref

        T = self.T_ref
        T2 = T + dT

        e_sc = np.exp(Isc * Rs / a)
        e_oc = np.exp(Voc / a)
        e_mp = np.exp((Vmp + Imp * Rs) / a)

        # (1) Short circuit current
        r1 = Isc - IL + I0*(e_sc - 1.0) + Isc*Rs/Rsh

        # (2) Open circuit voltage
        r2 = IL - I0*(e_oc-1.0) - Voc/Rsh

        # (3) MPP
        r3 = Imp - IL + I0*(e_mp-1.0) + (Vmp+Imp*Rs)/Rsh

        # (4) MPP derivative mus vanish
        num = I0/a * e_mp + 1.0/Rsh
        den = 1.0 + I0*Rs/a * e_mp + Rs/Rsh
        r4 = Imp - Vmp * (num/den)

        # (5) Temperature derivative
        Voc2 = Voc + self.mu_Voc*dT
        IL2 = IL + self.mu_Isc*dT
        a2 = a/T * T2
        I02 = I0*(T2/T)**3 * np.exp(1.0/kB*(self._Eg(T)*eV2J/T - self._Eg(T2)*eV2J/T2) )
        e_oc2 = np.exp(Voc2 / a2)
        r5 = IL2 - I02*(e_oc2-1.0) - Voc2/Rsh
        return [r1,r2,r3,r4,r5]        

    def _iv_equation(self,I,V,G=1000,T_C=25.0):
        IL, I0, a, Rs, Rsh = self.translate(G, T_C)
        return I - IL + I0*(np.exp((V+I*Rs)/a)-1.0) + (V+I*Rs)/Rsh

    def current(self, V, G, T_C):
        """
        Compute current at a given voltage, irradiance, and temperature [A].

        Solves the implicit diode equation numerically via fsolve.
        """
        IL, I0, a, Rs, Rsh = self.translate(G, T_C)

        def f(I):
            return (IL - I0 * (np.exp((V + I * Rs) / a) - 1.0)
                    - (V + I * Rs) / Rsh - I)
        
        iguess = IL - I0*(np.exp(V/a)-1.0) + V/Rsh

        return float(fsolve(f, x0=iguess)[0])

    def voltage(self, I, G, T_C):
        """
        Compute terminal voltage at a given current, irradiance, and temperature [V].

        Solves the implicit diode equation numerically via fsolve.
        """
        IL, I0, a, Rs, Rsh = self.translate(G, T_C)

        def f(V):
            return I - (IL - I0 * (np.exp((V + I * Rs) / a) - 1.0) - (V + I * Rs) / Rsh)

        return float(fsolve(f, x0=self.Voc(G=G, T_C=T_C) * 0.9)[0])

    def fit_parameters(self):
        T = self.T_ref
        Isc, Voc = self.Isc_ref, self.Voc_ref
        Imp, Vmp = self.Imp_ref, self.Vmp_ref

        # Initial guesses (n~1.3 typical for c-Si)
        a0   = 1.3 * self.Ns * kB * T / electron_charge
        IL0  = Isc
        I0_0 = Isc * np.exp(-Voc / a0)
        Rs0  = 0.5 * (Voc - Vmp) / Imp
        Rsh0 = 3.0 * Vmp / (Isc - Imp)

        sol, _, ier, msg = fsolve(self._residuals, [IL0, I0_0, a0, Rs0, Rsh0],full_output=True)

        if ier != 1:
            raise RuntimeError(f"STC parameter extraction failed: {msg}")

        IL, I0, a, Rs, Rsh = sol
        if not (IL > 0 and I0 > 0 and a > 0 and Rs >= 0 and Rsh > 0):
            raise ValueError("Extracted parameters have invalid signs — check inputs.")

        return IL, I0, a, Rs, Rsh
    
    def translate(self, G, T_C):
        """
        Translate STC parameters to arbitrary (G, T) conditions.

        Translation equations (De Soto 2006, Eqs. 11-14):
          IL  = G/G_ref * (IL_ref + mu_Isc*(T - T_ref))
          a   = a_ref * T/T_ref
          I0  = I0_ref * (T/T_ref)^3 * exp(Ns*Eg_ref/a_ref - Ns*Eg(T)/a)
          Rsh = Rsh_ref * G_ref/G
          Rs  = Rs_ref  (constant)

        Parameters
        ----------
        G : float
            Irradiance [W/m²].
        T_C : float
            Cell temperature [°C].

        Returns
        -------
        IL, I0, a, Rs, Rsh : floats
        """
        T = T_C + 273.15
        G_rat = G / self.G_ref

        IL  = G_rat * (self.IL_ref + self.mu_Isc * (T - self.T_ref))
        a   = self.a_ref * T / self.T_ref
        I0  = self.I0_ref * (T / self.T_ref)**3 * np.exp(
                  self.Ns * self.Eg0 / self.a_ref - self.Ns * self._Eg(T) / a)
        Rsh = self.Rsh_ref * self.G_ref / G
        Rs  = self.Rs_ref

        return IL, I0, a, Rs, Rsh

    def Voc(self,G=1000,T_C=25.0):
        IL, I0, a, Rs, Rsh = self.translate(G, T_C)
        Voc0 = a * np.log(IL/I0 + 1.0)
        return float(fsolve(lambda v: self._iv_equation(0.0,v,G=G,T_C=T_C), x0=Voc0)[0])

    def iv_curve(self, G, T_C, n_points=200):
        """
        Compute the full I-V (and P-V) curve.

        Parameters
        ----------
        G : float
            Irradiance [W/m²].
        T_C : float
            Cell temperature [°C].
        n_points : int
            Number of voltage points.

        Returns
        -------
        V, I, P : 1-D arrays  [V, A, W]
        """
        IL, I0, a, Rs, Rsh = self.translate(G, T_C)
        
        V = np.linspace(0.0, self.Voc(G=G,T_C=T_C) * 0.9999, n_points)
        I = np.array([self.current(v, G, T_C) for v in V])

        return V, I, V * I
    
    def plot_power_curve(self,G,T_C,n_points=200,ax=None,mplkwds={}):
        """
        Plot the O-V curve.

        Parameters
        ----------
        G : float
            Irradiance [W/m²].
        T_C : float
            Cell temperature [°C].
        n_points : int
            Number of voltage points.

        Returns
        -------
        V, I, P : 1-D arrays  [V, A, W]
        """

        if ax is None:
            fig,ax = plt.subplots()

        V, _, P = self.iv_curve(G, T_C)
        ax.plot(V,P,label=f'T={T_C:.1f}C, G={G:.1f}W/m2',**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Power (W)')
        ax.legend()

        return ax

    def plot_iv(self,G=1000,T_C=25.0,ax=None,mplkwds={}):
        
        if ax is None:
            fig,ax = plt.subplots()

        V, I, _ = self.iv_curve(G, T_C)
        
        ax.plot(V,I,label=f'T={T_C:.1f}C, G={G:.1f}W/m2',**mplkwds)
        ax.set_ylim((0,None))
        ax.set_xlim((0,None))
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')
        ax.legend()

        return ax

    def mpp(self, G, T_C):
        """
        Find the maximum power point.

        Returns
        -------
        Vmp, Imp, Pmp : floats
        """
        V, I, P = self.iv_curve(G, T_C)
        idx = np.argmax(P)
        return float(V[idx]), float(I[idx]), float(P[idx])

    def summary(self):
        """Print the extracted STC parameters."""
        n = self.a_ref / (self.Ns * k_B * self.T_ref / q_e)
        print("Extracted STC parameters:")
        print(f"  IL   = {self.IL_ref:.4f} A  (≈ Isc)")
        print(f"  I0   = {self.I0_ref:.3e} A")
        print(f"  a    = {self.a_ref:.4f} V   (ideality n = {n:.3f})")
        print(f"  Rs   = {self.Rs:.4f} Ω")
        print(f"  Rsh  = {self.Rsh_ref:.2f} Ω")