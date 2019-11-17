"""
Make Weather Types
------------------

This script takes in the field on which to cluster and
creates a specified number of weather types in EOF-space.

"""

import xarray as xr
import numpy as np
import sklearn.cluster as cl
from eofs.xarray import Eof
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latlim', type=int, nargs=2, help = 'LATMIN, LATMAX')
parser.add_argument('--lonlim', type=int, nargs=2, help = 'LONMIN, LATMAX')
parser.add_argument('--n_clusters', type=int, nargs=1, help = 'How many clusters to create?')
parser.add_argument('--prop', nargs=1, type=float, help = 'Proportion of variance to keep')
parser.add_argument('--variable', nargs=1, help = 'The variable [ie hgt]')
parser.add_argument('--infile', nargs=1, help='the folder containing the input file')
parser.add_argument('--outfile', nargs=1, help='path of combined file')
args = parser.parse_args()

def ReSort(x):
        # a helper function to re-sort the cluster labels
        x = np.int_(x)
        x_from = np.unique(x).argsort()
        counts = np.array([np.sum(x == xi) for xi in x_from])
        orders = (-counts).argsort().argsort()
        return(orders[x_from[x]])

def CalcACC(P,Q):
    acc = np.dot(P,Q) / np.sqrt(np.dot(P,P) * np.dot(Q,Q))
    return(acc)

def XrEofCluster(ds, n_clusters=5, prop=0.90, nsim=100, variable='hgt'):
    # translate data to EOF space
    solver = Eof(ds[variable])
    var_frac = solver.varianceFraction()
    cumvar = np.cumsum(var_frac.values)
    eofs_keep = np.where(cumvar >= prop)[0].min()
    pc = solver.pcs(npcs=eofs_keep) # time series of PC
    # initialize
    centroids = np.ones([nsim, n_clusters, pc.shape[1]])
    clusters = np.ones([nsim, pc.shape[0]])
    Aij = np.ones([nsim, n_clusters, n_clusters])
    Aprime = np.ones([nsim, n_clusters])
    Abest = np.ones(nsim)
    # run simulations
    for i in range(0, nsim):
        centroids[i,:,:], clusters[i, :], _ = cl.k_means(pc, n_clusters=n_clusters, n_init=1)
        clusters[i, :] = ReSort(clusters[i, :])
        clusters[i, :] += 1
    # compute classifiability index
    for n in range(0, nsim):
        for i in range(0, n_clusters):
            for j in range(0, n_clusters):
                if i == j:
                    Aij[n, i, j] = np.nan
                else:
                    P = centroids[n, i, :]
                    Q = centroids[n, j, :]
                    Aij[n, i, j] = CalcACC(P,Q)
            Aprime[n, :] = np.nanmax(Aij[n,:,:], axis=0)
        Abest = Aprime.min(axis=1)
    # extract useful
    classifiability = Abest.mean()
    best_part = np.where(Abest == Abest.max())[0][0]
    best_centroid = centroids[best_part,:,:]
    best_ts = clusters[best_part,:]
    # done
    return(best_centroid, best_ts, classifiability)


def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


def main():
    # convert the args to useful info
    latmin, latmax = np.min(args.latlim), np.max(args.latlim)
    lonmin, lonmax = np.min(args.lonlim), np.max(args.lonlim)
    n_clusters = args.n_clusters[0]
    variable = args.variable[0]
    prop = args.prop[0]
    infile = args.infile[0]
    outfile = args.outfile[0]

    # Subset the data
    selector = lambda ds: ds.sel(
        lon = slice(lonmin, lonmax),
        lat = slice(latmax, latmin))

    # Read data, calculate anomalies
    print('Reading in raw data...')
    years = np.int_(np.linspace(1980, 2016, 2016+1-1980))
    ds = [selector(xr.open_dataset(infile)).sel(time = slice('{}-12-01'.format(y - 1), '{}-02-28'.format(y))) for y in years]
    ds = xr.concat(ds, dim='time')
    ds = ds - ds.mean(dim='time')

    # run clustering
    nsim = 100
    _, weather_types, classifiability = XrEofCluster(ds, n_clusters=n_clusters, prop=prop, nsim=nsim, variable=variable)
    print('classifiability is {}'.format(classifiability))

    # Save to file
    print('saving to file...')
    weather_types = xr.DataArray(np.int_(weather_types), coords=[ds.time], dims=['time'])
    weather_types = xr.Dataset({'WT': weather_types})
    weather_types = weather_types.to_dataframe()
    weather_types.to_csv(outfile)

if __name__ == '__main__':
    main()
