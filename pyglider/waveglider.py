import os
import pyglider.utils as utils
import logging
import polars as pl
import xarray as xr
import numpy as np

_log = logging.getLogger(__name__)

possible_time_names = ["gliderTimeStamp", "timeStamp"]

def get_id(deployment: dict) -> str:
    return deployment['metadata']['glider_name']

def raw_to_rawnc(
    indir,
    outdir,
    deploymentyaml,
    incremental=True,
    dropna_subset=None,
    dropna_thresh=1
):
    """
    Convert seaexplorer text files to raw parquet pandas files.

    Parameters
    ----------
    indir : str
        Directory with the raw files are kept.  Recommend naming this
        directory "raw"

    outdir : str
        Directory to write the matching ``*.nc`` files. Recommend ``rawnc``.

    deploymentyaml : str
        YAML text file with deployment information for this glider.

    incremental : bool, optional
        If *True* (default), only netcdf files that are older than the
        binary files are re-parsed.

    min_samples_in_file : int
        Minimum number of samples in a raw file to trigger writing a netcdf
        file. Defaults to 5

    dropna_subset : list of strings, default None
        If more values than *dropna_thresh* of the variables listed here are
        empty (NaN), then drop this line of data.  Useful for raw payload files
        that are heavily oversampled.  Get the variable names from the raw text
        file.  See `pandas.DataFrame.dropna`.

    dropna_thresh : integer, default 1
        Number of variables listed in dropna_subset that can be empty before
        the line is dropped.


    Returns
    -------
    status : bool
        *True* success.

    Notes
    -----

    This process can be slow for many files.

    For the *dropna* functionality, list one variable for each of the sensors
    that is *not* over-sampled.  For instance, we had an AROD, GPCTD, and
    FLBBCD and the AROD was grossly oversampled, whereas the other two were not,
    but were not sampled synchronously.  In that case we chose:
    `dropna_subset=['GPCTD_TEMPERATURE', 'FLBBCD_CHL_COUNT']` to keep all
    rows where either of these were good, and dropped all other rows.
    """
    
    os.makedirs(outdir, exist_ok=True)


    # currently just works with csv file exports
    for file in os.listdir(indir):
        full_path = os.path.join(indir, file)
        root, ext = os.path.splitext(file)
        if ext != ".csv":
            _log.info(f"{file} is not a .csv, skipping")
            continue
        _log.info(f"Continuing to read file {file}")
        print(f"- reading file: {file}")
        
        # for some reason AIS report generates a .0 in an int column, here's a generic soltuion to apply to other files in case it happens.
        try:
            out = pl.read_csv(full_path)
        except pl.exceptions.ComputeError as e:
            # _log.warning(f"{file} failed to load, trying again while ignoring errors, this will result in some null rows, \n{e}")
            out = pl.read_csv(full_path, ignore_errors=True)

        # probably should add more time columns here
        # 2026-05-19T17:41:41.29
        time_formats = ['%m/%d/%Y %I:%M:%S %p', '%Y-%m-%dT%H:%M:%S%.f']
        for time_var in possible_time_names:
            if time_var in out.columns:
                time_cast_succsess = False
                for time_format in time_formats:
                    try:
                        out = out.with_columns(
                            pl.col(time_var).str.strptime(
                                pl.Datetime, format=time_format
                            )
                        )
                        time_cast_succsess = True
                        print(f"\t * Time format index {time_formats.index(time_format)} worked!")
                        break
                    except pl.exceptions.InvalidOperationError as e:
                        continue
                if not time_cast_succsess:
                    print(f"\t * BAD FORMAT")

        fnout = os.path.join(outdir, f'{root}.parquet')
        out.write_parquet(fnout)





def merge_parquet(indir, outdir, deploymentyaml, only_in_deployment=True):
    """
    Merge all the raw netcdf files in indir.  These are meant to be
    the raw flight and science files from the slocum.

    Parameters
    ----------
    indir : str
        Directory where the raw ``*.ebd.nc`` and ``*.dbd.nc`` files are.
        Recommend: ``./rawnc``

    outdir : str
        Directory where merged raw netcdf files will be put. Recommend:
        ``./rawnc/``.  Note that the netcdf files will be named following
        the data in *deploymentyaml*:
        ``glider_nameglider_serial-YYYYmmddTHHMM-rawebd.nc`` and
        ``...rawdbd.nc``.

    deploymentyaml : str
        YAML text file with deployment information for this glider.

    only_in_deployment : bool
        Only merge parquet files that are referenced in the deployment yaml
    """
    os.makedirs(outdir, exist_ok=True)
    deployment = utils._get_deployment(deploymentyaml)

    metadata = deployment['metadata']
    ncvar = deployment['netcdf_variables']
    out_path = os.path.join(outdir, f'{get_id(deployment)}-merged.parquet')

    if only_in_deployment:
        files_to_read = set([_get_filename_from_source(attrs) for _var, attrs in ncvar.items()])
    else:
        files_to_read = [os.path.splitext(file)[0] for file in os.listdir(indir) if os.path.splitext(file)[1] == ".parquet"]

    print(files_to_read)

    merged_df = None
    for file_to_read in files_to_read:
        print(f"reading: {file_to_read}")
        file_path = os.path.join(indir, f'{file_to_read}.parquet')
        df = pl.read_parquet(file_path)
        time_var = None
        for possible_time_name in possible_time_names:
            if possible_time_name in df.columns:
                time_var = possible_time_name
        if time_var is None:
            continue
            # raise ValueError(f"No possible time variables found in: {file_path}")
        df = df.rename({time_var: 'time'})
        df = df.rename({col:f'{file_to_read}:{col}' for col in df.columns if col != 'time'})
        df = df.unique(subset=['time'], keep='first') #for some reason the CTD is sampling twice at once on the hour
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, on='time', how='full', coalesce=True)

        merged_df = merged_df.sort('time')
        # merged_df.write_csv("TEST2.csv")
        merged_df.write_parquet(out_path)




def _get_glider_depth_nan(ds: xr.Dataset):
    attr = {
        'long_name': 'glider depth',
        'standard_name': 'depth',
        'units': 'm',
        'comment': 'No pressure/depth sensor on platform',
    }
    ds['depth'] = np.nan
    ds['depth'].attrs = attr
    return ds

        
# def _get_col_from_source(attrs: dict):
#     return attrs["source"].split(":")[1]

def _get_filename_from_source(attrs: dict):
    return attrs["source"].split(":")[0]

# def _get_df_from_ncvar(attrs, parquet_data) -> pl.DataFrame:
#     return parquet_data[_get_filename_from_source(attrs)]

def raw_to_timeseries(indir, outdir, deploymentyaml, interpolate=False, fnamesuffix=''):
    """
    Parameters
    ----------
    indir : string
        Directory with raw netcdf files.
    outdir : string
        Directory to put the merged timeseries files.
    Returns
    -------
    outname : string
        name of the new merged netcdf file.
    """

    deployment = utils._get_deployment(deploymentyaml)
    metadata: dict = deployment['metadata']
    ncvar: dict = deployment['netcdf_variables']
    device_data: dict = deployment['glider_devices']
    thenames = list(ncvar.keys())
    time_name = 'time'
    thenames.remove(time_name)
    merged_parquet_path = os.path.join(indir, f'{get_id(deployment)}-merged.parquet')
    if not os.path.exists(merged_parquet_path):
        raise ValueError("Merged parquet doesn't exist")

    print("opening merged parquet file")
    parquet_data = pl.read_parquet(merged_parquet_path)



    ds = xr.Dataset()
    attr = {}
    for atts in ncvar[time_name].keys():
        if atts != 'coordinates':
            attr[atts] = ncvar[time_name][atts]

    vals = parquet_data.select(['time']).to_numpy()[:, 0]
    indctd = np.where(~np.isnan(vals))[0]
    ds[time_name] = (
        (time_name),
        parquet_data.select(time_name).to_numpy()[:, 0].astype('datetime64[ns]'),
        attr)

    for ncvar_name, ncvar_attrs in ncvar.items():
        if ncvar_name == time_name:
            continue
        if 'method' not in ncvar_attrs:
            if 'conversion' in ncvar_attrs:
                convert = getattr(utils, ncvar_attrs['conversion'])
            else:
                convert = utils._passthrough
            sensorname = ncvar_attrs['source']
            if sensorname in parquet_data.columns:
                print(f"sensorname: {sensorname}")
                val = convert(parquet_data.select(sensorname).to_numpy()[:, 0])
                # We don't want to interpolate strings

                print(parquet_data[sensorname].dtype)
                interpolate_var = ncvar_attrs.pop("interpolate", interpolate)
                if (parquet_data[sensorname].dtype != pl.String and parquet_data[sensorname].dtype != pl.Boolean) and (interpolate and interpolate_var and not np.isnan(val).all()):
                    print("interpolating "+ncvar_name)
                    time_original = parquet_data.select('time').to_numpy()[:, 0]
                    time_var = time_original[np.where(~np.isnan(val))[0]]
                    var_non_nan = val[np.where(~np.isnan(val))[0]]
                    time_timebase = parquet_data.select('time').to_numpy()[indctd, 0]
                    if val.dtype == '<M8[us]':
                        # for datetime, must convert to numerical, interpolate, then convert back
                        us_since_1970 = (
                            var_non_nan - np.datetime64('1970-01-01')
                        ).astype(int)
                        val_int = np.interp(
                            time_timebase.astype(float),
                            time_var.astype(float),
                            us_since_1970,
                        )
                        val_us = val_int.astype('timedelta64[us]')
                        val = np.datetime64('1970-01-01') + val_us
                    else:
                        val = np.interp(
                            time_timebase.astype(float),
                            time_var.astype(float),
                            var_non_nan,
                        )
                else:
                    val = val[indctd]
                # ncvar['method'] = 'linear fill'
            else:
                raise ValueError(f"Name not in source data: {sensorname}")

            # make the attributes:
            attrs = ncvar_attrs
            attrs.pop('coordinates', None)
            attrs = utils.fill_required_attrs(attrs)
            # get rid of any units hiding in the column name
            ds[ncvar_name] = (('time'), val, attrs)

    if 'pressure' in ncvar:
        ds = utils.get_glider_depth(ds)
    else:
        ds = _get_glider_depth_nan(ds)
    try:
        ds = utils.get_derived_eos_raw(ds)
    except AttributeError as e:
        logging.warning(f"Error trying to get EOS\n{e}")

    ds = ds.sortby(ds.time)
    ds = utils.fill_metadata(ds, metadata, device_data)
    start = ds['time'].values[0]
    end = ds['time'].values[-1]

    ds.attrs['deployment_start'] = str(start)
    ds.attrs['deployment_end'] = str(end)

    os.makedirs(outdir, exist_ok=True)
    id0 = ds.attrs['deployment_name']
    outname = os.path.join(outdir, id0 + fnamesuffix + '.nc')
    _log.info('writing %s', outname)
    if 'units' in ds.time.attrs.keys():
        ds.time.attrs.pop('units')
    if 'calendar' in ds.time.attrs.keys():
        ds.time.attrs.pop('calendar')
    if 'ad2cp_time' in list(ds):
        if 'units' in ds.ad2cp_time.attrs.keys():
            ds.ad2cp_time.attrs.pop('units')
    ds.to_netcdf(
        outname,
        'w',
        encoding={
            'time': {'units': 'seconds since 1970-01-01T00:00:00Z', 'dtype': 'float64'}
        },
    )
    ds.to_pandas().to_csv(outname.replace(".nc", ".csv"))
    return outname




    


