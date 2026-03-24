import scipy.stats as stats
from scipy.optimize import fsolve,minimize_scalar, root_scalar
from scipy.optimize import minimize,LinearConstraint
from scipy.integrate import quad
import numpy as np
from numpy import log
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import pandas as pd
import csv
from typing import Any

# References
# [1] Wind Energy Explained, 1st ed. John Wiley & Sons, Ltd, 2009. doi: 10.1002/9781119994367.

class merra_wind_speed_data:
    """Container for MERRA-2 reanalysis wind data downloaded from NASA POWER.

    Imports single-point CSV exports from the NASA POWER Data Access Viewer and
    provides methods to extrapolate wind speed to arbitrary hub heights, fit
    wind profile parameters, and export data in SAM-compatible format.

    Typical MERRA-2 columns after import:
        ``WS10M``, ``WS50M`` – wind speed at 10 m and 50 m (m/s)
        ``WD10M``, ``WD50M`` – wind direction at 10 m and 50 m (degrees)
        ``PS``               – surface pressure (kPa in raw data)
        ``T2M``              – air temperature at 2 m (°C)

    Attributes
    ----------
    data_source_url : str
        URL of the NASA POWER data access viewer.
    latitude : float or None
        Site latitude in degrees (positive north).
    longitude : float or None
        Site longitude in degrees (positive east of Greenwich).
    date_range : str or None
        Date range string parsed from the CSV header.
    average_elevation : str or None
        Site elevation string parsed from the CSV header.
    time_reference : {'LST', 'UTC'} or None
        Time reference reported in the NASA POWER header. ``'LST'`` means the
        timestamps are in Local Standard Time (a fixed UTC offset — no DST);
        ``'UTC'`` means timestamps are in Coordinated Universal Time.
    data : pandas.DataFrame or None
        Imported time-series data with a ``Timestamp`` column.
    """

    def __init__(self):
        self.data_source_url = "https://power.larc.nasa.gov/data-access-viewer/"
        self.latitude = None
        self.longitude = None
        self.date_range = None
        self.average_elevation = None
        self.time_reference = None  # 'LST' or 'UTC' as reported in the NASA POWER header
        self.data = None

    def import_data(self,merra_single_point_csv: str):
        """Import a MERRA-2 single-point CSV file exported from NASA POWER.

        Parses the custom NASA POWER header (lines up to and including
        ``-END HEADER-``) to extract site metadata, then reads the remaining
        rows as a time-series DataFrame with a ``Timestamp`` column.

        Parameters
        ----------
        merra_single_point_csv : str or path-like
            Path to the NASA POWER CSV export file.

        Returns
        -------
        None
            Populates ``self.latitude``, ``self.longitude``,
            ``self.average_elevation``, ``self.time_reference``, and
            ``self.data`` in place.
        """
        file = merra_single_point_csv

        # read header
        with open(file,'r') as f:
            reader = csv.reader(f)
            header_count = 0
            for row in reader:
                header_count += 1
                if row and row[0] == '-END HEADER-':
                    break

                if "location" in row[0].lower():
                    r_split = row[0].split()
                    islat = [('latitude' in r.lower()) for r in r_split]
                    if any(islat):
                        self.latitude = float(r_split[islat.index(True)+1])

                    islon = [('longitude' in r.lower()) for r in r_split]
                    if any(islon):
                        self.longitude = float(r_split[islon.index(True)+1])

                if "elevation" in row[0].lower():
                    self.average_elevation = row[0].split('=')[-1]

                # The Dates line ends with "in LST" or "in UTC"
                if "dates" in row[0].lower():
                    tokens = row[0].split()
                    if tokens[-1].upper() in ('LST', 'UTC'):
                        self.time_reference = tokens[-1].upper()

        # read the rest
        df = pd.read_csv(file,skiprows=header_count)
        df['Timestamp'] = pd.to_datetime(dict(
                                year=df['YEAR'],
                                month=df['MO'],
                                day=df['DY'],
                                hour=df['HR'],
                                minute=df.get('minute', 0),  # use 0 if not present
                                second=df.get('second', 0),  # use 0 if not present
                                ))
        
        if self.time_reference == 'UTC':
            print(f"Time reference is UTC.")
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
        else:
            print(f"""Time reference is {self.time_reference}. Assuming timestamps are in local 
                      time with a fixed UTC offset (no DST). If this is from NASA POWER, this is 
                      likley incorrect since LST in the DAV is Local Solar Time.""")

        df = df.drop(['YEAR','MO','DY','HR'],axis=1)
        # NASA POWER names all shortwave irradiance parameters with 'SW'
        # (e.g. ALLSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DNI, CLRSKY_SFC_SW_DWN)
        solar_cols = [c for c in df.columns if 'SW' in c]
        df = df.drop(solar_cols, axis=1)
        df = df[['Timestamp',*df.columns[:-1]]]
        self.data = df

    def utc_to_lst(self, lst_offset: int):
        utc_str = f"{lst_offset:+d}:00"
        print(utc_str)
        self.time_reference = 'LST'
        self.data['Timestamp'] = self.data['Timestamp'].dt.tz_convert(utc_str)

    def add_speed_at_height(self,
                        heights: list[int],
                        z0: float | None = None,
                        alpha: float | None = None,
                        model: str = "power_law",
                        plot: bool = False):
        """Extrapolate wind speed from 50 m to one or more target heights.

        Uses either the power law or the logarithmic wind profile model.
        If the required profile parameter (``alpha`` or ``z0``) is not
        supplied it is fitted automatically from the 10 m and 50 m data
        already present in ``self.data``.

        New columns are added to ``self.data`` using the naming convention
        ``WS{height}M`` and ``WD{height}M``. Wind direction at the new
        height is approximated as the circular mean of the 10 m and 50 m
        directions.

        Parameters
        ----------
        heights : list of int or int
            Target height(s) in metres.
        z0 : float or None, optional
            Surface roughness length in metres (logarithmic model only).
            If None, ``fit_logarithmic`` is called automatically.
        alpha : float or None, optional
            Power law exponent (power law model only).
            If None, ``fit_power_law`` is called automatically.
        model : {'power_law', 'logarithmic'}, optional
            Wind profile model to use. Default is ``'power_law'``.
        plot : bool, optional
            If True, plot the fitted profile against the 10 m vs 50 m data.
            Default is False.

        Returns
        -------
        None
            Modifies ``self.data`` in place.
        """

        assert model.lower() in ['power_law','logarithmic'], 'Invalid model. Must be "power_law" or "logarithmic"'
        
        # ensure it is a list
        if isinstance(heights,int):
            heights = [heights]

        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        wd10,wd50= self.data['WD10M'], self.data['WD50M']
        if model.lower() == 'logarithmic':
            if z0 is None:
                print('z0 not provided. Will try to fit it based on data available from 10m and 50m')
                p = self.fit_logarithmic()
            else:
                p = z0
                            
            for height in heights:
                new_col = f'WS{height}M'
                scale = log(height/p)/log(50/p)
                self.data[new_col] = scale * ws50
                self.data[f'WD{height}M'] = stats.circmean(np.c_[wd10,wd50],low=0,high=360,axis=1)

        else: # model.lower() == 'power_law':
            if alpha is None:
                print('alpha not provided. Will try to fit it based on data available from 10m and 50m')
                p = self.fit_power_law()
            else:
                p = alpha

            for height in heights:
                new_col = f'WS{height}M'                
                scale = (height/50.0)**p
                self.data[new_col] = scale * ws50
                self.data[f'WD{height}M'] = stats.circmean(np.c_[wd10,wd50],low=0,high=360,axis=1)

        if plot:
            fig,ax = plt.subplots()
            ax.plot(ws50,ws10,'o')
            xl = ax.get_xlim()
            x = np.linspace(xl[0],xl[1],1000)
            if model.lower() == 'logarithmic':
                ax.plot(x,log(10.0/p)/log(50.0/p) * x,linewidth=3,label="Logarithmic fit")
            else:
                ax.plot(x,(10.0/50.0)**p *x,linewidth=3,label='Power law fit')
            ax.set_xlabel('Wind speed at 50m [m/s]')
            ax.set_ylabel('Wind speed at 10m [m/s]')
            ax.legend()
    
    def fit_power_law(self):
        """Fit the power law exponent α to the 10 m and 50 m wind speed data.

        Minimises the residual sum of squares between the measured 10 m wind
        speeds and those predicted from 50 m using the power law. Optimisation
        is performed in log-space (over ``log_alpha``) to guarantee α > 0.

        Returns
        -------
        float
            Fitted power law exponent α (dimensionless).
        """
        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        rss = lambda log_alpha: np.sum((ws10 - power_law(ws50,50,10,np.exp(log_alpha)) )**2)
        alpha = np.exp(minimize_scalar(rss)['x'])
        print(f'alpha = {alpha}')
        return alpha
    
    def fit_logarithmic(self):
        """Fit the surface roughness length z₀ to the 10 m and 50 m wind speed data.

        Minimises the residual sum of squares between the measured 10 m wind
        speeds and those predicted from 50 m using the logarithmic profile.
        Optimisation is performed in log-space (over ``log_z0``) to guarantee
        z₀ > 0.

        Returns
        -------
        float
            Fitted surface roughness length z₀ in metres.
        """
        ws10,ws50= self.data['WS10M'], self.data['WS50M']
        rss = lambda log_z0: np.sum((ws10 - logarithmic(ws50,50,10,np.exp(log_z0)) )**2)
        z0 = np.exp(minimize_scalar(rss)['x'])
        print(f'z0 = {z0:.3e}m')
        return z0
    
    def export_to_sam_csv(self,utc_offset: int, file_name: str,
                          date_range: list | None = None):
        """Export the wind data to a SAM-compatible CSV file.

        Writes a single-row header followed by the time-series data in the
        format expected by NREL's System Advisor Model (SAM) wind resource
        CSV format. Column names are converted to SAM conventions, e.g.
        ``WS80M`` → ``wind speed at 80m (m/s)``.

        Two assumptions are made and printed as runtime warnings:

        * Surface pressure (``PS``) is labelled as measured at 10 m even
          though MERRA-2 does not provide a height-resolved pressure field.
        * Air temperature uses the 2 m value (``T2M``) as a proxy for 10 m.

        Unit conversions applied:

        * Pressure: kPa → Pa.

        Parameters
        ----------
        utc_offset : int
            UTC offset in hours for the site (e.g. ``10`` for AEST, ``-7`` for
            MST). NASA POWER LST data uses a fixed offset with no DST, so an
            integer offset is the appropriate representation. Written to both
            ``Site Timezone`` and ``Data Timezone`` header fields.
        file_name : str or path-like
            Output file path for the SAM CSV.
        date_range : two-element array-like of datetime-like, optional
            ``[start, end]`` bounds (inclusive) used to slice ``self.data``
            before export. Compared against the ``Timestamp`` column.
            If ``None`` (default), all rows are exported.

        Returns
        -------
        None
            Writes the file to disk.
        """
        # See SAM CSV Format for Wind for more details on format.

        # header
        row1 = ['Site Timezone',utc_offset,
                'Data Timezone',utc_offset,
                'Latitude',self.latitude,
                'Longitude',self.longitude,
                'Elevation',self.average_elevation]
        row1 = [str(r) for r in row1]
        row1 = ','.join(row1)

        df2 = self.data.copy()

        # optional date range filter
        if date_range is not None:
            mask = (df2['Timestamp'] >= date_range[0]) & (df2['Timestamp'] <= date_range[1])
            df2 = df2.loc[mask]

        # rename wind speed and direction columns
        heights = [name[2:-1] for name in df2.columns if 'WS' in name]
        df2 = df2.rename(columns={f'WS{h}M':f'wind speed at {h}m (m/s)' for h in heights})
        df2 = df2.rename(columns={f'WD{h}M':f'wind direction at {h}m (degrees)' for h in heights})
        df2['PS'] = df2['PS'] * 1000  # convert from kPa to Pa

        print("Warning: Arbitrarily setting pressure data to be at 10m.")
        df2 = df2.rename(columns={'PS':'air pressure at 10m (Pa)'})

        print("Warning: Temperature at 10m is not available in MERRA-2 data. Using temperature at 2m instead.")
        df2 = df2.rename(columns={'T2M':'air temperature at 10m (C)'})

        # add back columns for year, month, day, hour, minute if not there
        def prepend(c,v):
            if c not in df2.columns:
                df2.insert(0,c,v)
        prepend('Minute',df2.Timestamp.dt.minute)
        prepend('Hour',df2.Timestamp.dt.hour)
        prepend('Day',df2.Timestamp.dt.day)
        prepend('Month',df2.Timestamp.dt.month)
        prepend('Year',df2.Timestamp.dt.year)

        df2 = df2.drop(columns='Timestamp')

        with open(file_name,'w',newline='') as f:
            f.write(row1+'\n')
            df2.to_csv(f,header=True,index=False)

def power_law(ws: float | np.ndarray, height: float, new_height: float, alpha: float):
    """Extrapolate wind speed to a new height using the power law profile.

    ws_new = ws · (new_height / height)^alpha

    Assumes neutral atmospheric stability. A typical value of alpha for
    open terrain is 1/7 ≈ 0.143.

    Parameters
    ----------
    ws : float or array-like
        Wind speed at the reference height (m/s).
    height : float
        Reference measurement height (m).
    new_height : float
        Target height (m).
    alpha : float
        Power law wind shear exponent (dimensionless).

    Returns
    -------
    float or ndarray
        Wind speed at *new_height* (m/s).
    """
    return ws * (new_height/height)**alpha

def mean_direction(wd: np.ndarray):
    """Compute the circular mean wind direction.

    Uses the complex-exponential method to correctly handle the 0°/360°
    wraparound: mean = angle(1/N Σᵢ· exp(j · θᵢ · π/180)) · 180/π.

    Returns
    -------
    float
        Mean wind direction in degrees, in the range [0°, 360°).
    """
    # Circular mean of the wind direction
    m = np.angle( np.sum(np.exp(np.pi/180 * wd * 1j)) / len(wd))
    if m > 0:  # the angle returns the principal value between -np.pi and np.pi, but I want 0 to 360
        return m * 180/np.pi
    else: 
        return 360 + m*180/np.pi

def logarithmic(ws: float | np.ndarray, height: float, new_height: float, z0: float):
    """Extrapolate wind speed to a new height using the logarithmic wind profile.

    ws_new = ws · ln(new_height / z0) / ln(height / z0)

    Assumes neutral atmospheric stability and that both *height* and
    *new_height* are well above the roughness sublayer.

    Parameters
    ----------
    ws : float or array-like
        Wind speed at the reference height (m/s).
    height : float
        Reference measurement height (m). Must be greater than *z0*.
    new_height : float
        Target height (m). Must be greater than *z0*.
    z0 : float
        Aerodynamic surface roughness length (m). Typical values range from
        ~0.0002 m (open sea) to ~1 m (urban areas).

    Returns
    -------
    float or ndarray
        Wind speed at *new_height* (m/s).
    """
    return ws * log(new_height/z0)/log(height/z0)
    
def speed_fit(data: np.ndarray, plot: bool = False, type: str = 'Weibull', hist_kwargs: dict = {}, fit_kwargs: dict = {}, handles: tuple | None = None):
    """Fit a Weibull or Rayleigh distribution to wind speed data.

    The location parameter is fixed at zero (``floc=0``) so the distribution
    starts at 0 m/s, which is physically appropriate for wind speeds.

    Parameters
    ----------
    data : array-like
        Measured wind speeds (m/s).
    plot : bool, optional
        If True, plot a histogram of the data alongside the fitted PDF.
        Default is False.
    type : {'Weibull', 'Rayleigh'}, optional
        Distribution family to fit. Case-insensitive. Default is ``'Weibull'``.
    hist_kwargs : dict, optional
        Keyword arguments forwarded to ``matplotlib.axes.Axes.hist``.
    fit_kwargs : dict, optional
        Keyword arguments forwarded to the PDF line plot.
    handles : tuple of (Figure, Axes) or None, optional
        Existing matplotlib figure and axes to plot onto. If None and
        *plot* is True, a new figure is created. Ignored when *plot* is False.

    Returns
    -------
    dist : scipy.stats frozen distribution
        Fitted Weibull or Rayleigh distribution object.
    fig : matplotlib.figure.Figure
        Only returned when *plot* is True.
    ax : matplotlib.axes.Axes
        Only returned when *plot* is True.
    """
    
    assert type.lower() is not None, 'Must provide a distribution type'
    assert type.lower() in ['weibull','rayleigh'], 'Invalid distribution type. Must be "weibull" or "rayleigh'
    if type.lower() == 'weibull':
        shape,loc,scale = stats.weibull_min.fit(data,floc=0) # don't fit location parameter
        dist = stats.weibull_min(shape,scale=scale)
    elif type.lower() == 'rayleigh':
        loc,scale = stats.rayleigh.fit(data,floc=0)
        dist = stats.rayleigh(scale=scale)
    else:
        raise ValueError('Invalid distribution type. Must be "weibull" or "rayleigh')
    
    if (not plot) and (handles is not None):
        print('Not adding to plot since plot = False')
        handles = None

    if plot:
        if handles is None:
            fig,ax = plt.subplots()
        else:
            fig,ax = handles
        ax.hist(data,label='Data',density=True,**hist_kwargs)
        XL = ax.get_xlim()
        x = np.linspace(XL[0],XL[1],1000)
        ax.plot(x,dist.pdf(x),label=f'{type} fit',**fit_kwargs)
        ax.set_xlabel('Wind speed')
        ax.set_ylabel('Density')
        ax.legend()
        return dist,fig,ax
    else:
        return dist

def change_height(dist: Any, height_old: float, height_new: float, alpha: float, type: str = 'weibull'):
    """Scale a wind speed distribution to a new hub height using the power law.

    Only the scale parameter of the distribution is modified; the shape
    parameter (which characterises the spread of wind speeds) is assumed to
    be height-independent.

    new_scale = old_scale · (height_new / height_old)^alpha

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Wind speed distribution at *height_old*. Must be a frozen
        ``weibull_min`` or ``rayleigh`` object.
    height_old : float
        Reference height of *dist* (m).
    height_new : float
        Target height (m).
    alpha : float
        Power law wind shear exponent (dimensionless).
    type : {'weibull', 'rayleigh'}, optional
        Distribution family. Case-insensitive. Default is ``'weibull'``.

    Returns
    -------
    scipy.stats frozen distribution
        New frozen distribution of the same family at *height_new*.

    Raises
    ------
    ValueError
        If *type* is not ``'weibull'`` or ``'rayleigh'``.
    """
    # Modify scale parameter for different height. See notes on wind for details.
    if type.lower() == 'weibull':
        return stats.weibull_min(dist.args[0],scale=dist.kwds['scale']*(height_new/height_old)**alpha)
    elif type.lower() == 'rayleigh':
        return stats.rayleigh(scale=dist.kwds['scale']*(height_new/height_old)**alpha)
    else:
        raise ValueError('Invalid distribution type. Must be "weibull" or "rayleigh')

def wind_rose(direction: np.ndarray, speed: np.ndarray, num_bins: int = 16, s_units: str | None = None):
    """Plot a wind rose (polar frequency histogram) for wind speed and direction.

    Each bar shows the frequency of winds from that direction sector; bar
    segments are coloured by wind speed class. Y-axis tick labels show
    percentage of total observations.

    Parameters
    ----------
    direction : array-like
        Wind directions in degrees. Convention is meteorological (0° = north,
        90° = east, increasing clockwise).
    speed : array-like
        Wind speeds corresponding to each direction measurement (m/s or other
        units — used for display only).
    num_bins : int, optional
        Number of directional sectors. Default is 16 (22.5° per sector).
    s_units : str or None, optional
        Unit label shown in the speed legend (e.g. ``'m/s'``). Default is None.

    Returns
    -------
    WindroseAxes
        The matplotlib-compatible axes object containing the wind rose plot.
    """
    ax = WindroseAxes.from_ax()
    ax.bar(direction,speed,nsector=num_bins,normed=True)
    ax.set_yticklabels([f"{a:.2f}%" for a in ax.get_yticks()])
    ax.legend(title=f'Wind speed {s_units}',loc='best')
    return ax

def power_cdf(dist: Any, power_curve: Any, n_bins: int = 10, power_units: str = 'W'):
    """Compute the cumulative distribution function of turbine power output.

    Discretises the wind speed distribution into *n_bins* equally spaced
    bins up to the 99.9th-percentile wind speed, maps each bin to a power
    value via the turbine power curve, then assembles the CDF of power output.

    Duplicate power values (e.g. multiple wind speed bins that map to rated
    power) are collapsed by retaining only the maximum probability for each
    unique power level.

    The returned arrays are prepended with ``(-1, 0)`` so that the CDF can be
    plotted starting from zero probability (a conventional CDF plotting
    convenience).

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Wind speed probability distribution.
    power_curve : turbine
        Turbine object with a ``get_power(wind_speed, power_units=...)`` method.
    n_bins : int, optional
        Number of wind speed bins for discretisation. Default is 10.
    power_units : {'W', 'kW', 'MW'}, optional
        Units for the returned power values. Default is ``'W'``.

    Returns
    -------
    power : ndarray
        Unique power values prepended with -1, in *power_units*.
    probability : ndarray
        CDF values (cumulative probability) corresponding to each power value,
        prepended with 0.
    """
    wmax = dist.ppf(0.999)
    w = np.linspace(0,wmax,n_bins+1)
    prob_w = dist.cdf(w)
    power_ = power_curve.get_power(w,power_units=power_units)

    power = np.unique(power_)
    prob = np.zeros_like(power)
    for ii,p in enumerate(power):
        prob[ii] = np.max(prob_w[power_== p])

    return np.r_[-1,power],np.r_[0,prob]
class speed_and_direction_dist:
    """Joint wind speed and direction distribution modelled as a directional mixture.

    The full distribution is represented as a mixture of per-sector wind speed
    distributions (one per azimuth bin), each weighted by the probability that
    the wind comes from that sector.

    Azimuth convention: bins span [0°, 360°) with 0° = north, increasing
    clockwise (meteorological convention). The first bin edge is at −dθ/2 so
    that 0° falls at the centre of the first bin.

    Parameters
    ----------
    type : {'weibull', 'rayleigh'}
        Distribution family for each directional sector.
    shapes : list of float
        Weibull shape parameters (k) for each sector. Must be empty for Rayleigh.
    scales : list of float
        Scale parameters (λ) for each sector (m/s).
    n_az_bins : int
        Number of equally spaced azimuth sectors.
    probabilities : list of float
        Probability weight for each sector. Should sum to 1.
    """

    def __init__(self, type: str | None = None, shapes: list[float] = [], scales: list[float] = [], n_az_bins: int | None = None, probabilities: list[float] = []):

        assert type is not None, 'Must provide a distribution type'
        assert type.lower() in ['weibull','rayleigh'], 'Invalid distribution type. Must be weibull or rayleigh'
        if type.lower() == 'weibull':
            self.dists = [stats.weibull_min(shape,scale=scale) for shape,scale in zip(shapes,scales)]
        elif type.lower() == 'rayleigh':
            assert shapes == [], 'Rayleigh distribution does not have shape parameter'
            self.dists = [stats.rayleigh(scale=scale) for scale in scales]
        else:
            raise ValueError('Invalid distribution type. Must be "weibull" or "rayleigh')
        dth = 360/(n_az_bins)
        azbin = np.arange(-dth/2,360-dth/2,dth)
        self.azimuth_bin_edges = azbin # azimuth is [0,360) with east-of-north positive
        self.azimuth_bin_centers = self.azimuth_bin_edges + dth/2
        azbin += dth
        self.probabilities = probabilities
        self.height = None
        self.type = type.lower()

    def set_height(self, height: float):
        """Record the reference height of the wind speed distributions.

        Must be called before ``change_height`` so the scaling factor can be
        computed correctly.

        Parameters
        ----------
        height : float
            Height in metres at which the distributions were fitted.
        """
        self.height = height

    def change_height(self, new_height: float, alpha: float):
        """Scale all directional sector distributions to a new height in place.

        Uses the power law to adjust the scale parameter of each sector's
        distribution. Currently only implemented for Weibull distributions;
        Rayleigh sectors are silently skipped.

        ``set_height`` must be called before this method.

        Parameters
        ----------
        new_height : float
            Target hub height (m).
        alpha : float
            Power law wind shear exponent (dimensionless).

        Returns
        -------
        None
            Modifies ``self.dists`` in place.
        """
        assert self.height is not None, 'Must set height of the original data using set_height()'
        
        p = self.probabilities
        if self.type == 'weibull':
            for ii,d in enumerate(self.dists):
                self.dists[ii] = change_height(d,self.height,new_height,alpha,self.type)

    def rvs(self, N: int = 1):
        """Draw random (direction, speed) samples from the joint distribution.

        First samples a directional sector from the multinomial defined by
        ``self.probabilities``, then draws a wind speed from the corresponding
        sector's distribution.

        Parameters
        ----------
        N : int, optional
            Number of samples to draw. Default is 1.

        Returns
        -------
        directions : ndarray, shape (N,)
            Sampled wind directions in degrees (azimuth bin centres).
        speeds : ndarray, shape (N,)
            Sampled wind speeds in m/s.
        """
        rng = np.random.default_rng()
        direction_bin = rng.choice(len(self.probabilities),size=N,p=self.probabilities,replace=True)
        speeds = np.array([self.dists[ii].rvs() for ii in direction_bin])
        directions = self.azimuth_bin_centers[direction_bin]
        return directions,speeds
    
    def get_params(self):
        """Extract fitted parameters from all directional sector distributions.

        Returns
        -------
        probabilities : ndarray, shape (n_sectors,)
            Sector probability weights.
        scales : ndarray, shape (n_sectors,)
            Scale parameter (λ) of each sector's distribution (m/s).
        shapes : ndarray, shape (n_sectors,)
            Shape parameter (k) of each sector's Weibull distribution.
            For Rayleigh distributions this will be the default args value.
        """

        probs = self.probabilities
        scales,shapes = [],[]
        for d in self.dists:
            shapes.append(d.args[0])
            scales.append(d.kwds['scale'])

        return np.array(probs),np.array(scales),np.array(shapes)

    def pdf(self, x: float | np.ndarray):
        """Evaluate the marginal wind speed probability density function.

        Computes the mixture PDF by summing each sector's PDF weighted by its
        probability: f(x) = Σᵢ pᵢ · fᵢ(x).

        Parameters
        ----------
        x : float or array-like
            Wind speed(s) at which to evaluate the PDF (m/s).

        Returns
        -------
        float or ndarray
            PDF value(s) at *x*.
        """
        return np.sum([self.probabilities[ii]*d.pdf(x) for ii,d in enumerate(self.dists)],axis=0)

    def cdf(self, x: float | np.ndarray):
        """Evaluate the marginal wind speed cumulative distribution function.

        Computes the mixture CDF: F(x) = Σᵢ pᵢ · Fᵢ(x).

        Parameters
        ----------
        x : float or array-like
            Wind speed(s) at which to evaluate the CDF (m/s).

        Returns
        -------
        float or ndarray
            CDF value(s) at *x* in [0, 1].
        """
        return np.sum([self.probabilities[ii]*d.cdf(x) for ii,d in enumerate(self.dists)],axis=0)
    
    def mean_speed(self):
        """Compute the marginal mean wind speed.

        E[V] = Σᵢ pᵢ · E[Vᵢ]

        Returns
        -------
        float
            Mean wind speed in m/s.
        """
        return np.sum([self.probabilities[ii]*d.mean() for ii,d in enumerate(self.dists)])
    
    def var(self):
        """Compute the marginal variance of wind speed.

        Uses the law of total variance for mixture distributions:
        Var[V] = Σᵢ pᵢ · (Var[Vᵢ] + E[Vᵢ]²) − E[V]²

        See https://en.wikipedia.org/wiki/Mixture_distribution#Moments

        Returns
        -------
        float
            Variance of wind speed in (m/s)².
        """
        return np.sum([self.probabilities[ii]*(d.var()+d.mean()**2) for ii,d in enumerate(self.dists)]) - self.mean_speed()**2  # see https://en.wikipedia.org/wiki/Mixture_distribution#Moments

    def ppf(self, p: float):
        """Compute the percent point function (inverse CDF) of the wind speed.

        Finds v such that CDF(v) = p using numerical root-finding
        (``scipy.optimize.fsolve``) with ``mean_speed()`` as the initial guess.

        Parameters
        ----------
        p : float
            Probability level in [0, 1].

        Returns
        -------
        ndarray
            Wind speed (m/s) at which the CDF equals *p*.
        """
        return fsolve(lambda x: self.cdf(x)-p,self.mean_speed())

    def fit(self, direction: np.ndarray, speed: np.ndarray, az_edges: np.ndarray | None = None, 
            plot: bool = False, hist_kwargs: dict = {}, fit_kwargs: dict = {}):
        """Fit per-sector wind speed distributions to observational data.

        Bins wind observations by direction sector, computes sector
        probabilities from observation counts, and fits a Weibull distribution
        to the wind speeds within each sector using ``speed_fit``.

        Parameters
        ----------
        direction : array-like
            Observed wind directions in degrees.
        speed : array-like
            Observed wind speeds (m/s), aligned with *direction*.
        az_edges : array-like or None, optional
            Azimuth bin edges to use instead of those set at construction.
            Only applied if ``self.azimuth_bin_edges`` is empty.
        plot : bool, optional
            If True, produce one figure per sector showing a density-normalised
            histogram of the sector's wind speeds with the fitted PDF overlaid.
            Empty sectors (no observations) are skipped. Default is False.
        hist_kwargs : dict, optional
            Keyword arguments forwarded to ``matplotlib.axes.Axes.hist`` for
            each sector plot. Ignored when *plot* is False.
        fit_kwargs : dict, optional
            Keyword arguments forwarded to the PDF line plot for each sector.
            Ignored when *plot* is False.
        handles : tuple or None, optional
            Reserved for future plotting use.

        Returns
        -------
        None
            Updates ``self.probabilities`` and ``self.dists`` in place.
        """
        # data is [direction,speed]
        self.dists = []
        assert (self.azimuth_bin_edges is not []) or (az_edges is not None), 'Must provide azimuth bin edges either in object or as a keyword argument to the fitting'

        if self.azimuth_bin_edges is []:
            self.azimuth_bin_edges = az_edges
            self.azimuth_bin_centers = az_edges[:-1] + np.diff(az_edges)/2

        # fits a distribution to the wind data for each direction
        dir,sp = direction,speed
        dir_bins = np.digitize(dir,self.azimuth_bin_edges,right=True)
        dir_bins[dir_bins == len(self.azimuth_bin_edges)] = 0 # wrap around to zero
        prob_hat = np.bincount(dir_bins,minlength=len(self.azimuth_bin_edges))
        self.probabilities = prob_hat/np.sum(prob_hat)

        n_sectors = len(self.azimuth_bin_edges)
        for ii in range(n_sectors):
            mask = dir_bins == ii
            if mask.any():
                if plot:
                    d, fig, ax = speed_fit(sp[mask], plot=True, hist_kwargs=hist_kwargs, fit_kwargs=fit_kwargs)
                    ax.set_title(f'Sector {ii}: {self.azimuth_bin_centers[ii]:.1f}°  (n={mask.sum()},p={self.probabilities[ii]*100:.1f}%)')
                else:
                    d = speed_fit(sp[mask], plot=False)
            else:
                d = stats.weibull_min(2, scale=1)  # placeholder; prob=0 so it never contributes
            self.dists.append(d)

    def print_params(self):
        """Print the fitted parameters of each directional sector distribution."""
        for ii,d in enumerate(self.dists):
            print(f'Sector {ii} ({self.azimuth_bin_centers[ii]:.1f}°): p={self.probabilities[ii]*100:.1f}%, c={d.kwds["scale"]:.2f}, k={d.args[0]:.2f}')

    def wind_speed_curve(self, x: np.ndarray, hours_per_year: float = 8760, plot: bool = True):
        """Compute (and optionally plot) the wind speed exceedance curve.

        Returns the probability that the wind speed exceeds each value in *x*,
        and optionally converts this to hours per year.

        R(v) = 1 − CDF(v)

        Parameters
        ----------
        x : array-like
            Wind speeds at which to evaluate the exceedance probability (m/s).
        hours_per_year : float, optional
            Number of hours in a year used to scale the x-axis when plotting.
            Default is 8760 (standard year).
        plot : bool, optional
            If True, plot hours-per-year (x-axis) vs. wind speed (y-axis).
            Default is True.

        Returns
        -------
        R : ndarray
            Exceedance probabilities at each value of *x*.
        fig : matplotlib.figure.Figure or None
            Figure object; None if *plot* is False.
        ax : matplotlib.axes.Axes or None
            Axes object; None if *plot* is False.
        """
        R = 1-self.cdf(x)

        if plot:
            fig,ax = plt.subplots()
            ax.plot(R*hours_per_year,x)
            ax.set_xlabel('Hours per year')
            ax.set_ylabel('Wind speed')
        else:
            fig = None
            ax = None

        return R,fig,ax

    def mean_direction(self):
        """Compute the circular mean wind direction.

        Uses the complex-exponential method to correctly handle the 0°/360°
        wraparound: mean = angle(Σᵢ pᵢ · exp(j · θᵢ · π/180)) · 180/π.

        Returns
        -------
        float
            Mean wind direction in degrees, in the range [0°, 360°).
        """
        # Circular mean of the wind direction
        m = np.angle( np.sum(np.exp(np.pi/180 * self.azimuth_bin_centers * 1j) * self.probabilities))
        if m > 0:  # the angle returns the principal value between -np.pi and np.pi, but I want 0 to 360
            return m * 180/np.pi
        else: 
            return 360 + m*180/np.pi
class turbine:
    """Wind turbine model storing power and thrust performance curves.

    All internally stored quantities use SI units (W, N, m, m/s). Conversion
    to other units is handled at the point of output (``get_power``, ``plot``,
    ``get_aep``, ``export_to_sam_format``).

    Performance data can be loaded from the NREL turbine-models library via
    ``import_nrel_power_curve``, or set manually via ``set_performance``.

    Attributes
    ----------
    wind_speeds : list or ndarray
        Wind speeds for the performance curve (m/s).
    power : list or ndarray
        Electrical power output at each wind speed (W).
    Cp : list or ndarray
        Power coefficient at each wind speed (dimensionless).
    Ct : list or ndarray
        Thrust coefficient at each wind speed (dimensionless).
    thrust : list or ndarray
        Rotor thrust force at each wind speed (N).
    cut_in_speed : float or None
        Minimum wind speed for power generation (m/s).
    cut_out_speed : float or None
        Maximum operating wind speed (m/s).
    rotor_diameter : float or None
        Rotor diameter (m).
    hub_height : float or None
        Hub height above ground (m).
    rated_power : float or None
        Nameplate (rated) power output (W).
    """

    def __init__(self):
        self.wind_speeds = []       # m/s
        self.power = []             # W
        self.Cp = []                # [-]
        self.Ct = []                # [-]
        self.thrust = []            # N
        self.cut_in_speed = None    # m/s
        self.cut_out_speed = None   # m/s
        self.rotor_diameter = None  # m
        self.hub_height = None      # m
        self.rated_power = None     # W

    def set_performance(self, wind_speeds: np.ndarray, power: np.ndarray | None = None, Cp: np.ndarray | None = None,
                        Ct: np.ndarray | None = None, thrust: np.ndarray | None = None):
        """Manually set turbine performance curves.

        Use this as an alternative to ``import_nrel_power_curve`` when loading
        data from a custom source. All arrays should be the same length and
        ordered by ascending wind speed.

        Parameters
        ----------
        wind_speeds : array-like
            Wind speeds (m/s).
        power : array-like or None, optional
            Electrical power at each wind speed (W).
        Cp : array-like or None, optional
            Power coefficient at each wind speed (dimensionless).
        Ct : array-like or None, optional
            Thrust coefficient at each wind speed (dimensionless).
        thrust : array-like or None, optional
            Rotor thrust force at each wind speed (N).
        """
        self.wind_speeds = wind_speeds
        self.power = power
        self.Cp = Cp
        self.Ct = Ct
        self.thrust = thrust
    
    def import_nrel_power_curve(self, turbine_name: str):
        """Import turbine specifications from the NREL turbine-models library.

        Populates all turbine attributes from the NREL dataset. Power is
        converted from kW (NREL convention) to W internally. If thrust or Ct
        data are absent in the dataset, a warning is printed and the
        corresponding attributes remain empty.

        Parameters
        ----------
        turbine_name : str
            Turbine model name as it appears in the NREL turbine-models
            database (e.g. ``'NREL_2p3_116'``). Case-sensitive.

        Returns
        -------
        None
            Populates ``self.wind_speeds``, ``self.power``, ``self.Cp``,
            ``self.Ct``, ``self.thrust``, ``self.cut_in_speed``,
            ``self.cut_out_speed``, ``self.rotor_diameter``,
            ``self.hub_height``, and ``self.rated_power`` in place.
        """
        # From https://github.com/NREL/turbine-models
        try:
            from turbine_models.parser import Turbines
        except ImportError as e:
            raise ImportError(
                "turbine_models is required for this method but could not be imported. "
                "Install it manually if available."
            ) from e

        turb = Turbines()
        specs = turb.specs(turbine_name)
        cols = [c.lower() for c in specs['power_curve'].columns]

        def find_col(substr):
            mask = [substr in c for c in cols]
            if True in mask:
                return cols[mask.index(True)]
            else:
                return None

        self.wind_speeds = specs['power_curve']['wind_speed_ms'].values
        self.power = specs['power_curve']['power_kw'].values*1000 # NREL data in kW
        self.Cp = specs['power_curve']['cp'].values
        self.cut_in_speed = specs['cut_in_wind_speed']
        self.cut_out_speed = specs['cut_out_wind_speed']
        self.rotor_diameter = specs['rotor_diameter']
        self.hub_height = specs['hub_height']
        self.rated_power = specs['rated_power']*1000 # NREL data in kW

        # import thrust if available
        thrust_column = find_col('thrust')
        ct_column = find_col('ct')
        if thrust_column is not None:
            self.thrust = specs['power_curve'][thrust_column].values
        else:
            print('No thrust data. Skipping import.')

        if ct_column is not None:
            self.Ct = specs['power_curve'][ct_column].values
        else:
            print('No thrust data. Skipping import.')
                                               
    def plot(self, ax: Any | None = None, nonzero_only: bool = False, power_units: str = 'MW', plt_kwargs: dict = {}):
        """Plot the turbine power curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If None, a new figure and axes are created.
        nonzero_only : bool, optional
            If True, plot only the tabulated data points (no interpolation).
            If False (default), interpolate a smooth curve from 0 m/s to
            slightly beyond the maximum tabulated wind speed.
        power_units : {'W', 'kW', 'MW'}, optional
            Units for the power axis. Default is ``'MW'``.
        plt_kwargs : dict, optional
            Additional keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        if ax is None:
            fig,ax = plt.subplots()
        
        if nonzero_only:
            x = self.wind_speeds
            fx = self.power
        else:
            ws = self.wind_speeds
            x = np.linspace(0,ws[-1]+0.1*(max(ws)-min(ws)),1000)
            fx = self.get_power(x)

        if power_units.lower() == 'w':
            ax.plot(x,fx,**plt_kwargs)
            ax.set_ylabel('Power [W]')
        elif power_units.lower() == 'kw':
            ax.plot(x,fx/1e3,**plt_kwargs)
            ax.set_ylabel('Power [kW]')
        elif power_units.lower() == 'mw':
            ax.plot(x,fx/1e6,**plt_kwargs)
            ax.set_ylabel('Power [MW]')
        else:
            raise ValueError('Invalid units. Must be "W", "kW", or "MW"')
        
        ax.set_xlabel('Wind speed [m/s]')
        return ax

    def get_power(self, wind_speed: float | np.ndarray, power_units: str = 'W'):
        """Interpolate power output at given wind speed(s).

        Uses linear interpolation on the tabulated power curve. Power is set
        to zero for wind speeds at or below ``cut_in_speed`` or strictly
        above ``cut_out_speed``.

        Parameters
        ----------
        wind_speed : float or ndarray
            Wind speed(s) in m/s.
        power_units : {'W', 'kW', 'MW'}, optional
            Units for the returned power. Default is ``'W'``.

        Returns
        -------
        float or ndarray
            Power output in the requested units. Returns the same type
            (scalar or array) as the input.
        """
        y = np.interp(wind_speed,self.wind_speeds,self.power)

        if power_units.lower() == 'w':
            scale = 1.0
        elif power_units.lower() == 'kw':
            scale = 1e-3
        elif power_units.lower() == 'mw':
            scale = 1e-6
        else:
            raise ValueError('Invalid units. Must be "W", "kW", or "MW"')

        if isinstance(wind_speed, np.ndarray):
            mask = (wind_speed <= self.cut_in_speed) | (wind_speed > self.cut_out_speed)
            y[mask]=0
        else:
            if (wind_speed <= self.cut_in_speed) | (wind_speed > self.cut_out_speed):
                y = 0.0

        return scale*y
        
    def get_thrust(self, wind_speed: float):
        """Interpolate rotor thrust at a given wind speed.

        Parameters
        ----------
        wind_speed : float
            Wind speed in m/s.

        Returns
        -------
        float or None
            Thrust force in N if *wind_speed* is within the operating range
            [``cut_in_speed``, ``cut_out_speed``); otherwise returns None.
        """
        if (wind_speed >= self.cut_in_speed) and (wind_speed < self.cut_out_speed):
            return np.interp(wind_speed,self.wind_speeds,self.thrust)

    def get_aep(self, dist: Any, ref_height: float | None = None, alpha: float | None = None, 
                units: str = 'MW', quad_kwargs: dict = {}, dt: float = 1.0, dist_type: str = 'weibull', 
                wind_profile: str = 'power_law'):
        
        """Compute the Annual Energy Production (AEP) of the turbine.

        Integrates the product of the power curve and the wind speed PDF over
        [0, ``cut_out_speed``], then scales by the number of time steps per year:

        AEP = ∫₀^v_cut_out P(v) · f(v) dv · dt · (8760 / dt)

        If *ref_height* is provided and differs from the hub height, the
        distribution is first scaled to hub height using the power law.

        Parameters
        ----------
        dist : scipy.stats frozen distribution
            Wind speed probability distribution.
        ref_height : float or None, optional
            Height (m) at which *dist* was measured. If None, or if equal to
            the turbine hub height, the distribution is used directly without
            height correction. Default is None.
        alpha : float or None, optional
            Power law exponent required when *ref_height* is not None.
        units : {'W', 'kW', 'MW'}, optional
            Output energy units. For example, ``'MW'`` returns MWh.
            Default is ``'MW'``.
        quad_kwargs : dict, optional
            Extra keyword arguments forwarded to ``scipy.integrate.quad``.
        dt : float, optional
            Time-step size in hours (used only in the year-scaling factor).
            Default is 1.0 (hourly).
        dist_type : {'weibull', 'rayleigh'}, optional
            Distribution family, used by ``change_height``. Default is
            ``'weibull'``.
        wind_profile : {'power_law'}, optional
            Height extrapolation model. Currently only ``'power_law'`` is
            supported. Default is ``'power_law'``.

        Returns
        -------
        float
            Annual energy production in the units implied by *units*
            (e.g. MWh when ``units='MW'``).
        """
        # dt is in hours. Default is 1.0 for hourly data.
    
        num_dt_per_year = int(8760/dt) # number of time steps per year

        if ref_height is None or ref_height == self.hub_height:
            if ref_height is None:
                print('No reference height provided. Assuming wind speed distribution is at the hub height.')
            else:
                print(f'Reference height ({ref_height} m) matches hub height. Using distribution directly.')
            integrand = lambda x: self.get_power(x, power_units=units) * dist.pdf(x)
        else:
            if wind_profile.lower() == 'power_law':
                assert alpha is not None, 'Must provide alpha for power law model'
                print(f"Changing height from {ref_height} to {self.hub_height}m using power law with alpha = {alpha}")
                dist_at_hub_height = change_height(dist, ref_height, self.hub_height, alpha, dist_type)
                integrand = lambda x: self.get_power(np.array([x]), power_units=units) * dist_at_hub_height.pdf(x)
            else:
                raise ValueError('Invalid height model. Only power_law implemented in this method so far.')

        res = quad(integrand,0,self.cut_out_speed,**quad_kwargs) # integrate from 0 to cut out speed
        return res[0]*dt*num_dt_per_year # AEP in MWh

    def export_to_sam_format(self, name: str):
        """Format turbine data as a SAM wind turbine CSV line.

        Produces a single comma-separated string compatible with the SAM wind
        turbine library format. Wind speeds and power values within each field
        are separated by pipe characters (``|``). Power is converted from W to
        kW for the SAM format.

        Parameters
        ----------
        name : str
            Turbine model name written into the first field.

        Returns
        -------
        str
            CSV-formatted string:
            ``name, rated_power_kW, rotor_diameter_m, unknown,
            ws1|ws2|..., power_kW1|power_kW2|...``
        """
        out = ",".join([name,str(self.rated_power/1000),str(self.rotor_diameter),'unknown'])
        
        wind_speeds = [f'{v}' for v in self.wind_speeds]
        wind_speeds = "|".join(wind_speeds)
        out = f"{out},{wind_speeds}"
        
        power = [f'{v/1000}' for v in self.power]
        power = "|".join(power)
        out = f"{out},{power}"

        return out

def initial_layout_rectangle(x_bnds:list,
                             y_bnds:list,
                             n_turbines:int,
                             n_grid_points:int = None,
                             rotation:float = 0.0,
                             placed_turbine_positions:list[np.ndarray]=None):
    """Generate an initial wind farm layout within a rectangular bounding box.

    Uses a greedy "farthest-first" algorithm: at each step the next turbine is
    placed at the candidate grid point that is farthest (in Euclidean distance)
    from all already-placed turbines. This heuristic spreads turbines across
    the domain but does not enforce a minimum separation distance.

    If *rotation* is non-zero the layout is rotated about the centre of the
    bounding box and then scaled to fit back inside the box.

    Parameters
    ----------
    x_bnds : list of [float, float]
        ``[x_min, x_max]`` bounds of the bounding box in metres.
    y_bnds : list of [float, float]
        ``[y_min, y_max]`` bounds of the bounding box in metres.
    n_turbines : int
        Number of turbines to place.
    n_grid_points : int or None, optional
        Total number of candidate grid points. Defaults to
        ``100 * n_turbines`` if None. The grid is approximately square
        (``floor(√n)`` × ``ceil(√n)`` points).
    rotation : float, optional
        Rotation angle of the layout grid in degrees. Default is 0.0 (no
        rotation).
    placed_turbine_positions : list of ndarray or None, optional
        Pre-positioned turbines (each an array of shape (2,)) to treat as
        already placed before the greedy selection begins. If None, the first
        turbine is chosen randomly from the grid.

    Returns
    -------
    ndarray, shape (n_turbines, 2)
        Array of turbine (x, y) positions in metres.
    """
    
    rotation = np.deg2rad(rotation)
    
    # Farthest first over a bounding box
    if n_grid_points is None:
        n_grid_points = 100*n_turbines

    xg = np.linspace(x_bnds[0], x_bnds[1], int( np.floor(np.sqrt(n_grid_points))) )
    yg = np.linspace(y_bnds[0], y_bnds[1], int(np.ceil(np.sqrt(n_grid_points))))
    xg,yg = np.meshgrid(xg,yg)
    xg = xg.flatten()
    yg = yg.flatten()
    pg = np.c_[xg,yg]

    if placed_turbine_positions is None:
        idx = np.random.random_integers(0,pg.shape[0]-1,1)
        points = [np.r_[xg[idx],yg[idx]]]
    else:
        points = placed_turbine_positions

    for n in range(len(points),n_turbines):
        points,pg = _add_min_squared_distance(points,pg)
        
    # scale the rectangle to the bounding box
    if np.abs(rotation) > 0.0:
        corners = np.array([[x_bnds[0],y_bnds[0]],
                            [x_bnds[1],y_bnds[0]],
                            [x_bnds[0],y_bnds[1]],
                            [x_bnds[1],y_bnds[1]]])
        W = x_bnds[1] - x_bnds[0]
        H = y_bnds[1] - y_bnds[0]

        R = rotation_matrix(rotation)
        center = np.mean(corners,axis=0)
        rot_corners = np.dot(corners-center,R.T) + center
        M = np.max(np.abs(rot_corners-center),axis=0)
        scale = np.min([W/2/M[0],H/2/M[1]])
        
        points = scale*np.dot(points-center,R.T) + center

    return np.array(points)

def _add_min_squared_distance(points: list, grid: np.ndarray):
    """Select the next turbine position using the farthest-first criterion.

    For each remaining candidate grid point, computes the minimum squared
    Euclidean distance to any already-placed turbine. The candidate with the
    largest such minimum distance is selected as the next turbine location and
    removed from the grid.

    Parameters
    ----------
    points : list of ndarray
        Already-placed turbine positions, each of shape (2,).
    grid : ndarray, shape (n_candidates, 2)
        Remaining candidate grid points.

    Returns
    -------
    points : list of ndarray
        Updated list with the newly selected turbine appended.
    grid : ndarray, shape (n_candidates - 1, 2)
        Candidate grid with the selected point removed.
    """
    M = len(points)
    d2 = np.inf*np.ones((grid.shape[0],M))
    for ii,p in enumerate(points):
        d2[:,ii] = np.sum((grid - p)**2,axis=1)
    
    d2 = np.min(d2,axis=1)
    mask = np.argsort(d2)
    points.append(grid[mask[-1],:])
    grid = np.delete(grid,mask[-1],axis=0)
    
    return points,grid

def rotation_matrix(ϕ: float, homogeneous: bool = False):
    """Construct a 2-D rotation matrix.

    R = [[cos ϕ, −sin ϕ],
         [sin ϕ,  cos ϕ]]

    Rotates a column vector counter-clockwise by *ϕ* radians in the x–y plane.

    Parameters
    ----------
    ϕ : float
        Rotation angle in radians.
    homogeneous : bool, optional
        If True, return a 3×3 homogeneous rotation matrix (the rotation is
        embedded in the upper-left 2×2 block with a 1 in the bottom-right
        corner). Default is False (returns a 2×2 matrix).

    Returns
    -------
    ndarray, shape (2, 2) or (3, 3)
        Rotation matrix.
    """

    if homogeneous:
        R = np.array([[np.cos(ϕ),-np.sin(ϕ),0],[np.sin(ϕ),np.cos(ϕ),0],
                      [0,0,1]])
    else:
        R = np.array([[np.cos(ϕ),-np.sin(ϕ)],[np.sin(ϕ),np.cos(ϕ)]])
    
    return R

def cp_max(λ):

    a1 = 0.25
    fun = lambda a: λ**2 - (1-a)*(1-4*a)**2 / (1-3*a)
    res = root_scalar(fun,bracket=(0.25,1.0/3.0-1e-6))
    a2 = res.root
    
    integrand = lambda a: ( (1-a)*(1-2*a)*(1-4*a) / (1-3*a) ) **2
    val,err =  quad(integrand,a=a1,b=a2)
    result = {'Cp_max': 24.0/λ**2 * val, 'a2': a2, 'error':err}
    return result

def betz_blade(λ,α,C_lift,number_of_blades=3,r=np.linspace(1e-6,1,100)):
    ϕ = np.atan(2/(3*r*λ))
    c_norm = (8*np.pi)/(3*number_of_blades*C_lift*λ)*np.sin(ϕ)
    θp =  ϕ - np.deg2rad(α)
    result = {  'r/R':r,
                'c/R':c_norm,
                'angle_of_relative_wind':np.rad2deg(ϕ),
                'section_pitch':np.rad2deg(θp)}
    return result
