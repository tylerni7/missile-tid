import awkward as ak
import numpy as np


def conn_to_awkward(conn_map):
    r = {}
    for station, sats in conn_map.items():
        local = {}
        for prn, connections in sats.items():
            conn_stack = []
            for conn in connections.connections:
                conn_stack.append(ak.from_numpy(conn.observations)[np.newaxis])

            if len(conn_stack) == 0:
                continue
            tmp = ak.concatenate(conn_stack)
            local[prn] = ak.from_regular(
                ak.zip({k: tmp[k] for k in ak.fields(tmp)}, depth_limit=1)[np.newaxis],
                axis=1,
            )

        r[station] = ak.zip(local, depth_limit=1)

    R = ak.zip(r, depth_limit=1)[0]
    return R
