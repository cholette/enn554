import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timezone, timedelta
from .math import cosd, sind, acosd
from IPython.display import display, Markdown
from calendar import isleap
from ipywidgets import interact
import ipywidgets as widgets
from .constants import h,c,kB,σ

def get_day_of_year(d,ignore_leapday=False):
    """Return the day of year for a given date.

    Parameters
    ----------
    d : datetime
        The date to convert.
    ignore_leapday : bool, optional
        If True and the year is a leap year, subtract 1 from the day of year
        for dates after February 28. This normalises leap years to a 365-day
        calendar, which is required by correlations (e.g. declination) that
        assume a 365-day year. Default is False.

    Returns
    -------
    int
        Day of year (1–365, or 1–366 for leap years when ignore_leapday is False).
    """
    doy = d.timetuple().tm_yday # day of year
    if ignore_leapday and isleap(d.year):
        if d.month > 2:
            doy -= 1
    return doy

def sun_solid_angle(R_sun,D_sun_earth):
    """Compute the solid angle subtended by the sun as seen from Earth.

    Parameters
    ----------
    R_sun : float
        Radius of the sun (m).
    D_sun_earth : float
        Distance from the centre of the sun to the observer (m).

    Returns
    -------
    float
        Solid angle in steradians (sr).
    """
    return 2*np.pi*(1-np.sqrt( (D_sun_earth**2-R_sun**2)/D_sun_earth**2 )) # [sr]

def black_body_spectral_radiance(λ,T,display_eqn=False):
    """Compute the spectral radiance of a black body using Planck's law.

    Parameters
    ----------
    λ : float or array-like
        Wavelength in micrometres (µm).
    T : float
        Temperature of the black body in kelvin (K).
    display_eqn : bool, optional
        If True, render the Planck equation as a LaTeX expression in the
        Jupyter notebook output. Default is False.

    Returns
    -------
    float or ndarray
        Spectral radiance in W m⁻² sr⁻¹ µm⁻¹.
    """
    C1 = 2*h*c**2 * 1e30 * 1e-6 # 1e30 for m -> µm then 1e6 for W/m3 -> W/m2/µm
    C2 = h*c/kB * 1e6 # 1e6 for m -> µm
    if display_eqn:
        display(Markdown( rf""" $E(\lambda,T) = \frac{{{2*h*c**2 * 1e30 * 1e-6:.4e}}}{{\lambda^5 \left[exp\left(\frac{{{C2:.4e}}}{{\lambda T}}\right) -1\right]}}\qquad [W/m^2 \, sr \, \mu m]$
                    """))
    return C1 / (λ**5 * (np.exp(C2/λ/T)-1) ) # [W/m2/µm/sr]

def black_body_spectral_exitance(λ,T):
    """Compute the spectral exitance of a convex black body surface.

    Integrates the spectral radiance over the hemisphere using the
    Lambertian relation M = π·L, which is exact for a convex surface.

    Parameters
    ----------
    λ : float or array-like
        Wavelength in micrometres (µm).
    T : float
        Temperature of the black body in kelvin (K).

    Returns
    -------
    float or ndarray
        Spectral exitance in W m⁻² µm⁻¹.

    """
    # assumes that the body is convex
    return np.pi * black_body_spectral_radiance(λ,T) # [W/m2/µm]
    
def declination(doy):
    """Compute the solar declination angle for a given day of year.

    Uses the Spencer approximation: δ = 23.45 · sin(360/365 · (doy − 81)).

    Parameters
    ----------
    doy : int or float
        Day of year (1–365). Leap days should be normalised to 365 days
        before calling this function (see ``get_day_of_year`` with
        ``ignore_leapday=True``).

    Returns
    -------
    float
        Solar declination angle in degrees. Ranges from −23.45° (winter
        solstice in the northern hemisphere) to +23.45° (summer solstice).
    """
    return 23.45*sind(360/365.0*(doy-81)) # δ = 23.45*sind(360/365.0*(284+doy))

def equation_of_time(doy):
    """Compute the equation of time correction for a given day of year.

    The equation of time accounts for the difference between apparent solar
    time and mean solar time caused by the eccentricity of Earth's orbit and
    the obliquity of the ecliptic.

    Parameters
    ----------
    doy : int or float
        Day of year (1–365).

    Returns
    -------
    float
        Time correction in minutes. Positive values mean the sun is ahead of
        the clock; negative values mean it is behind.
    """
    B = 360.0*(doy-81)/364.0
    E = 9.87*sind(2*B)-7.53*cosd(B)-1.5*sind(B)
    return E # minutes

def solar_time(t,L_tz,L_loc,ignore_leapday=False):
    """Convert standard clock time to local apparent solar time.

    Applies two corrections to the clock time: a longitude correction for
    the offset between the observer's meridian and the standard time-zone
    meridian (4 minutes per degree), and the equation of time correction for
    orbital effects.

    Parameters
    ----------
    t : datetime
        Standard clock time (timezone-naive or timezone-aware).
    L_tz : float
        Longitude of the standard time-zone meridian in degrees (positive east
        of Greenwich). For example, AEST (UTC+10) uses 150°.
    L_loc : float
        Longitude of the observer's location in degrees (positive east of
        Greenwich).
    ignore_leapday : bool, optional
        Passed to ``get_day_of_year``. Default is False.

    Returns
    -------
    datetime
        Local apparent solar time as a datetime object.
    """
    doy = get_day_of_year(t,ignore_leapday=ignore_leapday)
    st = t + timedelta(minutes = 4*(L_tz-L_loc) + equation_of_time(doy))
    return st

def compute_solar_angles(standard_clock_time: datetime,φ,L,L_tz,force_south_as_zero=False,ignore_leapday=False):
    """Compute solar zenith and azimuth angles for a given time and location.

    Converts standard clock time to solar time, then calculates the hour angle,
    declination, zenith angle, and azimuth angle. In the southern hemisphere the
    azimuth reference is automatically switched to north unless overridden.

    Parameters
    ----------
    standard_clock_time : datetime
        Local standard clock time (not daylight saving time).
    φ : float
        Observer latitude in degrees (positive north, negative south).
    L : float
        Observer longitude in degrees (positive east of Greenwich).
    L_tz : float
        Longitude of the standard time-zone meridian in degrees. For example,
        AEST (UTC+10) uses 150°.
    force_south_as_zero : bool, optional
        If True, always measure azimuth from south regardless of hemisphere.
        Default is False, which uses south as zero in the northern hemisphere
        and north as zero in the southern hemisphere.
    ignore_leapday : bool, optional
        Normalise leap years to 365 days for angle correlations. Default is False.

    Returns
    -------
    dict with keys:
        ``zenith`` : float
            Solar zenith angle in degrees (0° = overhead, 90° = horizon).
        ``azimuth`` : float
            Solar azimuth angle in degrees. Positive values are west of the
            reference direction; negative values are east.
        ``declination`` : float
            Solar declination angle in degrees.
        ``solar_time`` : float
            Local apparent solar time as a decimal hour.
        ``azimuth_zero`` : str
            Reference direction for azimuth (``"South"`` or ``"North"``).
    """
    azimuth_zero = "South"
    doy = get_day_of_year(standard_clock_time,ignore_leapday=ignore_leapday)
    sdt = solar_time(standard_clock_time,L_tz,L,ignore_leapday=ignore_leapday)
    t_solar = sdt.hour + sdt.minute/60 + sdt.second/3600 + sdt.microsecond/1e6/3600
    ω = 15*(t_solar-12.0)
    δ = declination(doy)
    θ = acosd( sind(δ)*sind(φ)+cosd(δ)*cosd(φ)*cosd(ω) )
    γ = np.sign(ω) * np.abs( acosd( (cosd(θ)*sind(φ) - sind(δ))/(sind(θ)*cosd(φ)) ) )

    if (φ < 0) and (not force_south_as_zero): # southern hemisphere and convention is not forced to use south as zero
        azimuth_zero = "North" # set azimuth zero to N
        γ = _set_azimuth_zero_to_north(γ)

    return {"zenith":θ, 
            "azimuth":γ,
            "declination":δ, 
            "solar_time": t_solar,
            "azimuth_zero":azimuth_zero}

def _set_azimuth_zero_to_north(γ):
    """Convert a south-referenced azimuth angle to a north-referenced azimuth angle.

    Parameters
    ----------
    γ : float
        Azimuth angle in degrees measured from south (positive west, negative east).

    Returns
    -------
    float
        Azimuth angle in degrees measured from north (positive west, negative east).
    """
    if γ<0: # east of north
        γ = -180 - γ
    else: # west of north
        γ = 180 - γ

    return γ

def _all_sun_angles(dt0, hour_grid, doy_grid,φ,L,L_tz,force_south_as_zero=False,ignore_leapday=False,location_name=""):
    """Compute solar zenith and azimuth angles over a grid of days and hours.

    Parameters
    ----------
    dt0 : datetime
        Reference datetime corresponding to day-of-year offset 0 in *doy_grid*.
    hour_grid : array-like of float
        Hours of the day (decimal) at which to evaluate sun angles.
    doy_grid : array-like of int
        Day-of-year offsets from *dt0* at which to evaluate sun angles.
    φ, L, L_tz : float
        Observer latitude, longitude, and time-zone meridian longitude (degrees).
    force_south_as_zero : bool, optional
        See ``compute_solar_angles``. Default is False.
    ignore_leapday : bool, optional
        See ``compute_solar_angles``. Default is False.
    location_name : str, optional
        Unused; reserved for labelling purposes.

    Returns
    -------
    zeniths : ndarray, shape (len(doy_grid), len(hour_grid))
        Solar zenith angles in degrees.
    azimuths : ndarray, shape (len(doy_grid), len(hour_grid))
        Solar azimuth angles in degrees.
    azimuth_zero : str
        Reference direction for azimuth (``"South"`` or ``"North"``).
    """
    zeniths = np.zeros( (len(doy_grid),len(hour_grid)) )
    azimuths = np.zeros( (len(doy_grid),len(hour_grid)) )
    for ii,d in enumerate(doy_grid):
        for jj,h in enumerate(hour_grid):
            dt = dt0 + timedelta(days=d) + timedelta(hours=h)
            sun_angles = compute_solar_angles(dt,φ,L,L_tz,
                                              force_south_as_zero=force_south_as_zero,
                                              ignore_leapday=ignore_leapday)
            zeniths[ii,jj] = sun_angles['zenith']
            azimuths[ii,jj] = sun_angles['azimuth']
    
    return zeniths, azimuths, sun_angles['azimuth_zero']

def sun_path_diagram(dt0, hour_grid, doy_grid,φ,L,L_tz,force_south_as_zero=False,ignore_leapday=False,location_name=""):
    """Plot a 2-D sun path diagram (altitude vs. azimuth) for multiple days.

    Each day in *doy_grid* is drawn as a separate arc coloured by date using the
    ``"twilight"`` colormap.

    Parameters
    ----------
    dt0 : datetime
        Reference datetime corresponding to day-of-year offset 0 in *doy_grid*.
    hour_grid : array-like of float
        Hours of the day (decimal) at which to evaluate sun position.
    doy_grid : array-like of int
        Day-of-year offsets from *dt0* for which to draw arcs.
    φ : float
        Observer latitude in degrees (positive north).
    L : float
        Observer longitude in degrees (positive east of Greenwich).
    L_tz : float
        Longitude of the standard time-zone meridian in degrees.
    force_south_as_zero : bool, optional
        See ``compute_solar_angles``. Default is False.
    ignore_leapday : bool, optional
        See ``compute_solar_angles``. Default is False.
    location_name : str, optional
        Name shown in the plot title. Default is an empty string.

    Returns
    -------
    dict with keys:
        ``figure_handle`` : matplotlib.figure.Figure
        ``axis_handle`` : matplotlib.axes.Axes
        ``data`` : dict
            ``zenith``, ``azimuth`` – ndarrays of shape
            (len(doy_grid), len(hour_grid)); ``azimuth_zero`` – str.
    """
    zeniths,azimuths,az_zero = _all_sun_angles(dt0, hour_grid, doy_grid,φ,L,L_tz,
                                       force_south_as_zero=force_south_as_zero,
                                       ignore_leapday=ignore_leapday,
                                       location_name=location_name)

    # Create a colormap
    cmap = plt.get_cmap("twilight")

    # Normalize the color values to be between 0 and 1
    norm = plt.Normalize(doy_grid.min(), doy_grid.max())

    fig,ax = plt.subplots()
    for ii,d in enumerate(doy_grid):
        ax.plot(azimuths[ii,:],90-zeniths[ii,:], color=cmap(norm(d)))
    ax.set_ylim((0,90))
    ax.set_xlabel(f"Azimuth Angle (degrees, + is West of {az_zero})")
    ax.set_ylabel("Altitude angle (degrees)")
    ax.set_title(f"Sun path for {location_name} ({φ:.2f},{L:.2f})")

    # Add a color bar to show the color scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)


    # Customize the colorbar labels to show date format
    def date_formatter(x, pos):
        return mdates.num2date(x).strftime('%b %d')
    # cbar.set_ticks([doy_grid.min(),121,242,doy_grid.max()])
    cbar.set_ticks(doy_grid)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(date_formatter))
    
    return {    'figure_handle':fig, 
                'axis_handle':ax,
                'data': {'zenith':zeniths,
                         'azimuth':azimuths,
                         'azimuth_zero': az_zero}}

def sun_path_3d(dt0, hour_grid, doy_grid,φ,L,L_tz,
                     ignore_leapday=False,
                     location_name="",
                     convention='sam',
                     radius=1.0):
    """Create an interactive 3-D sun path visualisation in a Jupyter notebook.

    Renders solar arcs on a hemisphere using ``ipywidgets`` sliders to control
    the selected date, hour, and viewing angle. The sun position is represented
    as a yellow sphere on the selected arc.

    Parameters
    ----------
    dt0 : datetime
        Reference datetime corresponding to day-of-year offset 0 in *doy_grid*.
    hour_grid : array-like of float
        Hours of the day (decimal) at which to evaluate sun position.
    doy_grid : array-like of int
        Day-of-year offsets from *dt0* for which to draw arcs.
    φ : float
        Observer latitude in degrees (positive north).
    L : float
        Observer longitude in degrees (positive east of Greenwich).
    L_tz : float
        Longitude of the standard time-zone meridian in degrees.
    ignore_leapday : bool, optional
        See ``compute_solar_angles``. Default is False.
    location_name : str, optional
        Reserved for labelling; not currently used in the plot. Default is ``""``.
    convention : str, optional
        Coordinate convention for ``solar_vector_from_angles``. Either
        ``"sam"`` (x-east, y-north, z-zenith) or ``"soltrace"``
        (x-west, y-zenith, z-north). Default is ``"sam"``.
    radius : float, optional
        Radius of the hemisphere on which sun vectors are plotted. Default is 1.0.

    Returns
    -------
    ipywidgets.interactive
        The interactive widget object returned by ``ipywidgets.interact``.
    """
    
    θz,γs,az_zero = _all_sun_angles(dt0, hour_grid, doy_grid,φ,L,L_tz,
                                       force_south_as_zero=True,
                                       ignore_leapday=ignore_leapday,
                                       location_name=location_name)
    
    sun_vecs = np.zeros((len(doy_grid),len(hour_grid),3))
    for ii,_ in enumerate(doy_grid):
        for jj,_ in enumerate(hour_grid):
            sun_vecs[ii,jj,:] = solar_vector_from_angles(γs[ii,jj],θz[ii,jj],convention=convention)
    sun_vecs = radius*sun_vecs

    # doys = [f"Arc {i+1}" for i in range(len(doy_grid))]
    doys = [(dt0+timedelta(days=dt)).strftime('%d-%b') for dt in doy_grid]
    hods = [f"{h:.1f}" for h in hour_grid]
    def _plot_sun(doy,hod,angle=0,elev=30):
        di = doys.index(doy)
        hi = hods.index(hod)
        mag = 1.5*np.sum(sun_vecs**2,axis=2).max()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # sun arcs for each day
        for ii in range(sun_vecs.shape[0]):
            hours_above_zero = np.where(sun_vecs[ii,:,2]>=0)[0]
            ax.plot(sun_vecs[ii,hours_above_zero,0],
                    sun_vecs[ii,hours_above_zero,1],
                    sun_vecs[ii,hours_above_zero,2],
                    color='gray',
                    alpha=0.5,
                    linewidth=0.5)
    
        # highlight selected arc
        hours_above_zero = np.where(sun_vecs[di,:,2]>=0)[0]
        ax.plot(    sun_vecs[di,hours_above_zero,0],
                    sun_vecs[di,hours_above_zero,1],
                    sun_vecs[di,hours_above_zero,2],
                    color='black',
                    linewidth=0.5)
        
        # plot sun
        ax.scatter( sun_vecs[di,hi,0],
                    sun_vecs[di,hi,1],
                    sun_vecs[di,hi,2],
                    color='#FFDF00',
                    s=100)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # compass
        x_start,x_end,y_start,y_end,z_start,z_end = -mag,mag,0,0,0.0*mag,0.0*mag
        # ax.plot([x_start, x_end], [y_start, y_end],[0,0], marker=None)
        ax.text(x_start, y_start, z_start, "W", color='black', fontsize=12, ha='center')
        ax.text(x_end, y_end, z_end, "E", color='black', fontsize=12, ha='center')
        dx = x_end - x_start
        dy = y_end - y_start
        dz = z_end - z_start
        ax.quiver(x_start, y_start, z_start, dx, dy, dz, arrow_length_ratio=0.05, color='black')
        ax.quiver(x_end, y_end, z_end, -dx, -dy, -dz, arrow_length_ratio=0.05, color='black')

        x_start,x_end,y_start,y_end,z_start,z_end = 0,0,-mag,mag,0.0*mag,0.0*mag
        # ax.plot([x_start, x_end], [y_start, y_end],[0,0], marker=None)
        ax.text(x_start, y_start, z_start, "S", color='black', fontsize=12, ha='center')
        ax.text(x_end, y_end, z_end, "N", color='black', fontsize=12, ha='center')
        dx = x_end - x_start
        dy = y_end - y_start
        dz = z_end - z_start
        ax.quiver(x_start, y_start, z_start, dx, dy, dz, arrow_length_ratio=0.05, color='black')
        ax.quiver(x_end, y_end, z_end, -dx, -dy, -dz, arrow_length_ratio=0.05, color='black')

        ax.set_zlim((0,mag))
        ax.set_ylim((-mag,mag))
        ax.set_ylim((-mag,mag))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_zticks(())

        ax.view_init(elev=elev, azim=angle)
        plt.draw()
    
    # Create sliders   
    arc_slider = widgets.SelectionSlider(options=doys, value=doys[0], description='Date:')
    point_slider = widgets.SelectionSlider(options=hods, value=hods[0], description='Hour:')
    angle_slider = widgets.FloatSlider(value=0, min=0, max=360, step=1, description='View Azimuth:')
    elev_slider = widgets.FloatSlider(value=30, min=0, max=90, step=1, description='View Elevation:')
    
    # interactive plot
    intp = interact(_plot_sun, 
                    doy=arc_slider, 
                    hod=point_slider,
                    angle=angle_slider,
                    elev=elev_slider)
    return intp

def solar_vector_from_angles(γ_s,θ_zenith,convention='sam'):
    """Convert solar azimuth and zenith angles to a 3-D unit vector.

    The direction conventions match those used by common solar simulation tools.

    Parameters
    ----------
    γ_s : float
        Solar azimuth angle in degrees, measured from south (positive west,
        negative east). Must be a south-referenced value regardless of hemisphere.
    θ_zenith : float
        Solar zenith angle in degrees (0° = directly overhead).
    convention : str, optional
        Coordinate system convention:

        * ``"sam"`` – x points east, y points north, z points toward zenith.
          This matches the SAM 3-D Shade Calculator convention.
        * ``"soltrace"`` – x points west, y points toward zenith, z points north.

        Default is ``"sam"``.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector pointing from the observer toward the sun.

    Raises
    ------
    AssertionError
        If *convention* is not ``"sam"`` or ``"soltrace"``.
    """
    g,z = γ_s,θ_zenith
    assert convention.lower() in ['soltrace','sam'], "Convention must be either soltrace or sam"

    if convention.lower() == 'soltrace':
        # x-west, y-zenith, z-north
        return np.array([sind(z)*sind(g),cosd(z),-sind(z)*cosd(g)])
    elif convention.lower() == 'sam':
        # x-east, y-north, z-zenith. See SAM manual 3D Shade Calculator for more on this
        return np.array([sind(z)*sind(-g),-sind(z)*cosd(-g),cosd(z)])

def angle_of_incidence(θ_z,γ_s,β,γ):
    """Compute the angle of incidence of direct solar radiation on a tilted surface.

    Uses the standard geometric relation for a fixed flat surface:
    cos(AOI) = cos(θ_z)·cos(β) + sin(θ_z)·sin(β)·cos(γ_s − γ).

    Parameters
    ----------
    θ_z : float
        Solar zenith angle in degrees.
    γ_s : float
        Solar azimuth angle in degrees (south-referenced, positive west).
    β : float
        Surface tilt angle from horizontal in degrees (0° = horizontal, 90° = vertical).
    γ : float
        Surface azimuth angle in degrees (south-referenced, positive west). Use 0°
        for a south-facing surface in the northern hemisphere.

    Returns
    -------
    float
        Angle of incidence in radians. Values greater than π/2 indicate that
        the sun is behind the surface.
    """
    return np.arccos(cosd(θ_z)*cosd(β) + sind(θ_z)*sind(β)*cosd(γ_s-γ))

def plane_of_array_irradiance(standard_clock_times,DNI,GHI,φ,L,L_tz,β,γ,
                              ρ=0,model="ISM",θ_z_max=90,
                              clip_aoi_greater_than_90=False,
                              ignore_leapday=False):
    """Compute the time series of plane-of-array (POA) irradiance for a tilted surface.

    Decomposes global horizontal irradiance (GHI) into its direct, diffuse, and
    ground-reflected components and projects each onto the surface plane using the
    isotropic sky model (ISM).

    POA = DNI·cos(AOI) + DHI·(1 + cos β)/2 + GHI·ρ·(1 − cos β)/2

    where DHI = GHI − DNI·cos(θ_z).

    Parameters
    ----------
    standard_clock_times : list of datetime
        Timestamps in standard clock time (not daylight saving time).
    DNI : array-like of float
        Direct normal irradiance time series in W m⁻².
    GHI : array-like of float
        Global horizontal irradiance time series in W m⁻².
    φ : float
        Observer latitude in degrees (positive north).
    L : float
        Observer longitude in degrees (positive east of Greenwich).
    L_tz : float
        Longitude of the standard time-zone meridian in degrees.
    β : float
        Surface tilt from horizontal in degrees.
    γ : float
        Surface azimuth in degrees (south-referenced, positive west).
    ρ : float, optional
        Ground reflectance (albedo). Default is 0 (no ground reflection).
    model : str, optional
        Irradiance decomposition model. Currently only ``"ISM"`` (isotropic sky
        model) is supported. Default is ``"ISM"``.
    θ_z_max : float, optional
        Maximum solar zenith angle in degrees beyond which irradiance is set to
        zero. Default is 90.
    clip_aoi_greater_than_90 : bool, optional
        If True, set POA to zero when the sun is behind the surface (cos AOI < 0).
        Default is False.
    ignore_leapday : bool, optional
        See ``compute_solar_angles``. Default is False.

    Returns
    -------
    list of float
        POA irradiance at each timestamp in W m⁻².

    Raises
    ------
    AssertionError
        If *standard_clock_times*, *DNI*, and *GHI* are not the same length.
    ValueError
        If *model* is not ``"ISM"``.
    """
    
    assert len(DNI) == len(GHI), "times, DNI, and GHI must be the same length"
    assert len(standard_clock_times) == len(GHI), "times, DNI, and GHI must be the same length"
    G_POA = []
    for ii,t in enumerate(standard_clock_times):
        sun_pos = compute_solar_angles(t,φ,L,L_tz,force_south_as_zero=True,ignore_leapday=ignore_leapday)
        θ_z,γ_s = sun_pos['zenith'],sun_pos['azimuth']
        cAOI = cosd(θ_z)*cosd(β) + sind(θ_z)*sind(β)*cosd(γ_s-γ)
        DHI = GHI[ii]-DNI[ii]*cosd(θ_z)

        if model.lower()=="ism":
            if θ_z < θ_z_max: # sun is above horizon
                g = DNI[ii]*cAOI + DHI*0.5*(1+cosd(β)) + ρ*GHI[ii]*0.5*(1-cosd(β))
            else:
                g = 0 
        else:
            raise ValueError("Model must be ISM for now.")

        if clip_aoi_greater_than_90 and cAOI < 0.0:
            g = 0.0
        
        G_POA.append(g)
        
    return G_POA