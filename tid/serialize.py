import h5py
from pathlib import Path


def save_stations(station_data: dict, fname: Path, *, mode="w-"):
    """Write the station data for a scenario to an hdf5 file."""
    with h5py.File(fname, mode) as fout:
        for station, sats in station_data.items():
            for prn, data in sats.items():
                fout[f"{station}/{prn}"] = data


def save_conn_mapp(conn_map: dict, fname: Path, *, mode="w-"):
    """Write the connection map for a scenario to an hdf5 file."""
    with h5py.File(fname, mode) as fout:
        for station, sats in conn_map.items():
            for prn, conns in sats.items():
                for j, conn in enumerate(conns.connections):
                    fout[f"{station}/{prn}/{j}"] = conn.observations
