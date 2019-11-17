'''
This module defines a number of functions to make it easier to plot data with contours,
and optional quivers, subsetting by a variable given by a pandas time series
'''


def FormatAxes(axes, coast=True, grid=True, border=True, river = False, extent=None, ticks=None):
    import cartopy.feature
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    '''
    A function to format all the axes to look the same.
    Allow boolean arguments for how the formatting should look.
    ticks: [x,y]
    '''
    for ax in axes.flat:
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        if coast: ax.coastlines()
        if grid: ax.gridlines()
        if border: ax.add_feature(cartopy.feature.BORDERS)
        if river: ax.add_feature(cartopy.feature.RIVERS)
        if ticks is not None:
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
            ax.set_xticks(ticks[0], crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            # repeat for y
            ax.set_yticks(ticks[1], crs=ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
        if extent is not None:
            ax.set_extent(extent, crs = ccrs.PlateCarree())
    plt.tight_layout()



def SetupAxes(ncol, nax, proj, figsize = [12, 7]):
    import matplotlib.pyplot as plt
    import numpy as np
    '''
    This function creates a plot object and axes object.
    ncol: number of columns desired
    nax: the number of sub-plots which will be generated
    The primary advantage of this function is that you don't need to calculate
    the number of rows and columns; it is essentially  just a wrapper for plt.subplots()
    '''
    nrow = int(np.ceil(nax / ncol))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                             subplot_kw={'projection': proj},
                             figsize=figsize)
    return(fig, axes)



def GetRowCol(i, axes):
    '''
    This function gets the row and column of the ith subplot in axes.
    This may be too simplistic for some plots.
    '''
    if len(axes.shape) > 1:
        nrow = axes.shape[0]
        ncol = axes.shape[1]
        row_i = i // ncol
        col_i = i - (row_i * ncol)
        return(row_i, col_i)
    else:
        return(i)


def SubSetMean(ds, i, cat_ts):
    import pandas as pd
    '''
    Takes ds, subsets using cat_ts, and returns the time-mean of all points with the ith unique value
    ds: a DataSet or DataArray (xarray object)
    cat_ts: a time series with a column determining the category of each time point
    groupbyvar: the name of that column
    '''
    times = cat_ts[cat_ts == pd.unique(cat_ts)[i]].index
    times = pd.to_datetime(times)
    sub = ds.sel(time = times)
    sub = sub.mean(dim='time')
    return(sub)


def AxisContourQuiver(ds, ax, contour_name = None, quiver_name = None,
    lonname = 'lon', latname = 'lat',
    maxval = 10, n_level = 26, levels = None,
    cmap = 'seismic'):
    import cartopy.crs as ccrs
    import numpy as np
    '''
    contour_name: name of variable in ds to be plotted as contour
    quiver_name: [name of u, name of v]
    maxval: range of colorbar
    '''
    assert contour_name is not None
    X, Y = np.meshgrid(ds[lonname].values, ds[latname].values)
    Z = ds[contour_name]
    if levels is None:
        levels = np.linspace(-maxval, maxval, n_level)
    Z.plot.contourf(
        transform = ccrs.PlateCarree(),
        ax = ax,
        levels = levels,
        cmap = cmap,
        add_colorbar=True)
    if quiver_name is not None:
        U, V = ds[quiver_name[0]].values, ds[quiver_name[1]].values
        ax.quiver(X,Y,U,V, transform=ccrs.PlateCarree())
