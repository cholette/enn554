# %%
# References:
#   [1] R. Forristall, “Heat Transfer Analysis and Modeling of a Parabolic Trough Solar Receiver 
#       Implemented in Engineering Equation Solver,” National Renewable Energy Lab. (NREL), 
#       Technical Report TP-550-34169.
#   [2] V. Dudley et al., “Test results: SEGS LS-2 solar collector,” SAND94-1884, 70756, 
#       ON: DE95010682, Dec. 1994. doi: 10.2172/70756.

 
import egb351.parabolic_trough as pt
from egb351.math import cosd, acosd, atand, tand, sind
import numpy as np
import egb351.sun as sun
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


plant = pt.parabolic_trough_1D_transient()
plant.import_material_parameters('data/parabolic_trough_parameters.csv')
# %% 
to_kelvin = lambda t: t+273.15
to_celcius = lambda t: t-273.15

T_ambient = 22 # C
γ_axis = 0 # north-south orientation, +ve west
T_sky = T_ambient - 8 # According to [1] 
L_aperture = 12*16*4.06 # According to [1, p. 133]. 
DNI = 950 # W/m2
massflow = 10 # kg/s
T = [250, 250,T_ambient]
T = [to_kelvin(t) for t in T]
wind_speed = 6.7
IAM = lambda θ: cosd(θ) + 0.000884*θ - 0.00005369*θ**2 # incidence angle modifier [2, p. 12], theta in degrees.
ε_soiling = 0.95
N_elements = 10


θz,γs = 20,37.8 # azimuth zero along axis pointing towards equator, +ve west
α = 90-θz
β,γ = pt.one_axis_tracking_angles(θz,γs,γ_axis,output='tilt_azimuth')
rotation = pt.one_axis_tracking_angles(θz,γs,γ_axis,output='rotation_about_axis_azimuth')

θ = pt.incidence_angle(θz,γs,β,γ)
θ2 = pt.incidence_angle2(θz,γs,γ_axis)

# %%
pc = plant.collector
epsilons = [pc[k] for k in pc.keys() if "epsilon_" in k]
η_opt = ε_soiling * np.prod(epsilons) * pc['rho_nominal'] * IAM(θ)
Q_sun = plant.solar_heating_rates(DNI,η_opt)
# %% 
T0 = np.array([T for ii in range(N_elements)])
ambient = {'T_ambient':T_ambient+273.15,
            'T_sky':T_sky+273.15,
            'wind_speed': wind_speed,
            'DNI':DNI,
            'η_opt':η_opt}


# %% Solve ODEs
dt = 1.0
tf = 1000
N_hce = plant.collector['number_of_hce']
L_hce = plant.collector['hce_length']
element_edges = np.linspace(0,L_hce*N_hce,N_elements+1)
zl = element_edges[0:-1]
zu = element_edges[1::]
n_brackets = (zu[0]-zl[0])/L_hce
t = np.arange(0,tf,dt)
sol = odeint(lambda T,time: plant.pt_odes(T,ambient,massflow,N_elements,n_brackets),T0.flatten(),t)

# %% Process solution
Ts = np.zeros((N_elements,3,len(t)))
for ii in range(sol.shape[0]):
    Ts[:,:,ii] = sol[ii,:].reshape(N_elements,3)
    Ts[:,:,ii] = plant._apply_boundary_conditions(Ts[:,:,ii])

cmap = cm.viridis
norm = mcolors.Normalize(vmin=min(zl), vmax=max(zl))
zn = norm(zl)
colors = cmap(zn)

fig,ax = plt.subplots(nrows=3,figsize=(5,10))
for ii,zz in enumerate(zl):

    ax[0].plot(t,to_celcius(Ts[ii,0,:].T),label=fr'$z \in \left[{zz:.0f},{zu[ii]:.0f}\right)$',color=colors[ii])
    ax[1].plot(t,to_celcius(Ts[ii,1,:].T),label=fr'$z \in \left[{zz:.0f},{zu[ii]:.0f}\right)$',color=colors[ii])
    ax[2].plot(t,to_celcius(Ts[ii,2,:].T),label=fr'$z \in \left[{zz:.0f},{zu[ii]:.0f}\right)$',color=colors[ii])

ax[0].set_title('HTF')
ax[1].set_title('Absorber')
ax[1].set_ylabel('Temperature (C)')
ax[2].set_title('Envelope')
ax[2].set_xlabel('Time (s)')
ax[2].legend()
fig.tight_layout()

# %%
T = Ts[:,:,-1]
Q_htf,Q_env,Q_conv_surr,Q_rad_surr,Q_bracket = [],[],[],[],[]
for ii in range(Ts.shape[0]):
    Q_htf.append(plant.absorber_to_htf_convection(T[ii,:],massflow))
    Q_env.append(plant.absorber_to_envelope_radiation(T[ii,:]))
    Q_conv_surr.append(plant.envelope_to_surroundings_convection(T[ii,:],T_ambient+273.15,wind_speed))
    Q_rad_surr.append(plant.envelope_to_surroundings_radiation(T[ii,:],T_sky+273.15))
    Q_bracket.append(plant.bracket_conduction(n_brackets,T_ambient+273.15,T[ii,1]-10,wind_speed))
# %%
