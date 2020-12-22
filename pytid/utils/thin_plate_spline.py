import datetime, os, math, re
import numpy as np
import scipy as sp
import laika, math
from laika.lib.coordinates import ecef2geodetic
# from pytid.utils.io import DATETIME_FMT_PRINTABLE, save_to_pickle_file
import subprocess, time

C=laika.constants.SPEED_OF_LIGHT
f1=laika.constants.GPS_L1
f2=laika.constants.GPS_L2
delay_factor = (f1**2 * f2**2) / (f1**2 - f2**2); K = 40.308e16;
DK=delay_factor/K
GPS_PRNS = [f"G{i:02d}" for i in range(1,33)]
VERBOSE=False
DEBUG=True

# Kernel_TPS = lambda x,y: np.nan_to_num(np.linalg.norm(y-x)**2*np.log(np.linalg.norm(y-x)),nan=0.)
# Kernel_TPS = lambda x,y: 0. if np.linalg.norm(y-x)==0. else np.nan_to_num(np.linalg.norm(y-x)**2*np.log(np.linalg.norm(y-x)),nan=0.)
DATETIME_FMT_PRINTABLE = '%Y-%m-%d - %H:%M:%S'

checkpoint_list=[(datetime.datetime.now(), 'Module Loaded')]
def io_save_to_pickle(dat, file_path):
    '''Simple wrapper function to save time when saving data as a pickled file. Used to save/load TPS object.'''
    import pickle
    tgt_path = os.path.abspath(file_path)
    with open(tgt_path, 'wb') as dat_f:
        pickle.dump(dat, dat_f)

def io_load_from_pickle(file_path):
    '''Simple wrapper function to save time when loading data as a pickled file. Used to save/load TPS object.'''
    import pickle
    tgt_path = os.path.abspath(file_path)
    with open(tgt_path, 'rb') as dat_f:
        dat = pickle.load(dat_f)
    return dat

def get_mem_usage(PID):
    '''DEBUGGING function. Runs a system call to get the memory usage (virtual, resident set) for a particular process ID.'''
    time.sleep(1)
    p1=('ps -q %s v' % str(PID)).split(' ')
    p2 = 'tail -n 1'.split(' ')
    p3 = ['sed', 's/ \\+/,/g']
    p4 = 'cut -f 7,8 -d ,'.split(' ')
    p1p = subprocess.Popen(p1, stdout=subprocess.PIPE)
    p2p = subprocess.Popen(p2, stdin=p1p.stdout, stdout=subprocess.PIPE)
    p3p = subprocess.Popen(p3, stdin=p2p.stdout, stdout=subprocess.PIPE)
    p4p = subprocess.Popen(p4, stdin=p3p.stdout, stdout=subprocess.PIPE)
    o,e=p4p.communicate()
    vals=list(map(int,o.decode('ascii').strip().split(',')))
    return int(int(vals[0])/100000)/10, int(int(vals[1])/100000)/10 #round to 0.1 gb

def print_nparray_sizes(nparrs):
    '''DEBUGGING function. Prints size info for a dictionary of numpy arrays. I used this to help
    dampen the memory footprint a little.'''
    print('%10s %10s %12s %8s %12s %12s %9s' % ('array', 'shape', 'size', 'dtype', 'totbytes', 'expbytes', 'mem (GB)'))
    tot_mem=0
    for k in nparrs.keys():
        totbyt, shp, sz, dt = get_nparray_size_data(nparrs[k])
        expbyt = int(sz * np_dtype_size(dt))
        mem_gb = float(totbyt)/1e9
        tot_mem += mem_gb
        print(f"{k:>10} {str(shp):>10s} {sz:>12d} {str(dt):>8s} {totbyt:>12d} {expbyt:>12d} {mem_gb:>5.1f}")
    print('%21s %12s %8s %12s %12s %9s' % ('Total Memory Usage','','','','',('%.1f'% tot_mem)))

def np_dtype_size(dt):
    '''DEBUGGING function. Takes a numpy.dtype and converts that into the number of bytes per data item. E.g. np.float64
    will return 8'''
    dts=str(dt)
    m = re.search('[0-9]+$',dts)
    return int(int(m.group(0))/8)

def get_nparray_size_data(myarr, return_str=False):
    '''DEBUGGING function. Useful for debugging. Returns a tuple (or optionally a string representation thereof) of
    some key bits of data about how big a matrix is: (# bytes, shape, size, data-type)'''
    if sp.sparse.issparse(myarr):
        if sp.sparse.isspmatrix_csr(myarr):
            tot_bytes=myarr.data.nbytes + myarr.indptr.nbytes + myarr.indices.nbytes
        else:
            tot_bytes=myarr.data.nbytes
    else:
        tot_bytes=myarr.nbytes
    shp=myarr.shape
    sz=myarr.size
    dt=myarr.dtype
    if return_str:
        return '(%d, %s, %d, %s)' % (tot_bytes, shp, sz, dt)
    return tot_bytes, shp, sz, dt

def checkpoint(msg, verbose=VERBOSE):
    '''DEBUGGING function. helper function for debugging. optionally prints a message but mainly stores the time
    points at every check point.'''
    t = datetime.datetime.now()
    last_t = checkpoint_list[-1][0]
    first_t = checkpoint_list[0][0]
    td1=t-last_t
    td2=t-first_t
    td1p=datetime.timedelta(td1.days, td1.seconds, 0)
    td2p = datetime.timedelta(td2.days, td2.seconds, 0)
    mem=get_mem_usage()
    memstr='(%.1f , %.1f)' % mem
    if verbose:
        print('%s - %s (%s cume) - %s (GB) -- %s' % (t.strftime(DATETIME_FMT_PRINTABLE), td1p, td2p, memstr, msg))
    checkpoint_list.append((t, msg))

def tps_kernel_matrix(xr, xc):
    '''
    This is an important function for this module. This takes two sets of coordinates (n x 2 column vectors)
    and returns the kernel matrix evaluated over them, pairwise. Speicifcally, this function implements the thin-plate
    spline kernel: k(r)=r^2 * log(r), although it does it exclusively with numpy functions which speeds this
    function WAY up.

    Both inputs are matrices of width 2 (coordinate matrices). 'xr' is the coordinates that will
    vary down the rows of the kernel matrix. 'xc' is the ones that vary across columns.
    '''
    div_curr = np.geterr()['divide']; inv_curr=np.geterr()['invalid'];
    np.seterr(divide='ignore', invalid='ignore')
    nr = xr.shape[0]
    nc = xc.shape[0]
    R = np.linalg.norm(np.dstack((np.repeat(xr[:, 0], nc).reshape((nr, nc)) - np.tile(xc[:, 0], nr).reshape((nr, nc)),
                        np.repeat(xr[:, 1], nc).reshape((nr, nc)) - np.tile(xc[:, 1], nr).reshape((nr, nc)))),axis=2)
    Rz = np.where(R==0.)
    K = np.multiply(R**2, np.log(R))
    K[Rz]=0.0
    np.seterr(divide=div_curr, invalid=inv_curr)
    return K


def get_lat_lon_lists(lon_res, lat_res, NSWE_bounds):
    '''Returns two lists of points, one for lat and one for lon, that are spaced evenly between the bounds of the map.
    Specifically, the desired resolution is adjusted slightly downward to fit an integer number of steps exactly
    on the map. The resulting actual resolutions are returned along with the lists.
    NOTE: assumes that the east longitude is greater than the west longitude, which may not necessarily be true.
    '''
    #TODO: make it so this can handle when East < West due to stradling the international date line.
    N = max(NSWE_bounds[0], NSWE_bounds[1])
    S = min(NSWE_bounds[0], NSWE_bounds[1])
    E = NSWE_bounds[3]
    W = NSWE_bounds[2]
    NSdist = (N-S)
    EWdist = (E-W) if E>W else 360.-abs(E-W) #if bounds cross dateline then take complement
    longitude_mod_func = lambda x: (x + 180.) % 360 - 180.
    n_lat_levels = int(NSdist/lat_res) + 1
    if NSdist/lat_res==int(NSdist/lat_res):
        n_lat_levels -= 1
    n_lon_levels = int(EWdist / lon_res) + 1
    if EWdist/lon_res==int(EWdist/lon_res):
        n_lon_levels -= 1
    lat_res_adj = NSdist / n_lat_levels
    lon_res_adj = EWdist / n_lon_levels
    lat_list = list(map(lambda x: S + x*lat_res_adj, range(n_lat_levels+1)))
    lon_list = list(map(lambda x: longitude_mod_func(W + x * lon_res_adj), range(n_lon_levels + 1)))
    return lon_list, lat_list, lon_res_adj, lat_res_adj

def get_lons_in_Xrange_for_horizontal_band(X, center_lat, lat_halfwidth, lon_list, lon_res):
    '''For a particular horizontal band on the map, return a grid of evenly spaced longitudes
    contained within the range of the X-values in that band.'''
    # TODO: figure out what to do if East-West map boundaires span the international date line.
    minlat = center_lat - lat_halfwidth;
    maxlat = center_lat + lat_halfwidth;
    hband_x_positions = np.where((X[:, 2] >= minlat) & (X[:, 2] <= maxlat))[0]
    minXlon = np.min(X[hband_x_positions, 1])
    maxXlon = np.max(X[hband_x_positions, 1])
    out_lons = list(filter(lambda x: (x > minXlon - lon_res) and (x < maxXlon + lon_res), lon_list))
    out_lons.sort()
    return out_lons

class ThinPlateSpline():
    '''
    This class implements a generic Thin-Plate spline model. It has routines for fitting the model to a set of
    training data as well as for evaluating the model on arbitrary points. It does not have any of the bells and
    whistles, for example a semi-parametric version or the multi-layer version used in the bias calc functions
    later on.
    '''
    X=None
    Y=None
    n=None
    K = None
    lambdasmooth = 0.1
    w_hat = None; a_hat = None;
    Y_hat = None; e_hat = None;
    std_err_resid = None;
    Xnew = None; Ynew_hat = None; Knew = None;
    prediction_grid_points = None;
    prediction_grid_values = None;
    prediction_grid_image = None;
    map_bounds_NSWE = (50., 20., -130., -60.);
    lats_list = None; lons_list = None;
    self_pickle_file_path = None;

    def __init__(self, Y=None, X=None, verbose=True, lambda_smoothing=None, from_file_path=None):
        '''
        Initialize by giving Y and X as numpy arrays. Y must be (n x 1) and X must be (n x 3) with the first coordinate
        set to 1.0 (i.e. homogenous coordinates).
        :param Y: numpy array (N x 1), dependent variable
        :param X: numpy array (N x 3), locations in homogenous coordinates (i.e. [1.0, x, y])
        '''
        # if a file path to load is specified, default to that...
        if from_file_path is not None and os.path.isfile(from_file_path):
            self.load_from_pickle_file(from_file_path)
            return
        # ...otherwise use the inputs (if any are given)
        if X is not None and Y is not None:
            Yn = Y.shape[0]
            Xn = X.shape[0]
            assert Yn==Xn, 'Y and X must be the same height. (Y is shape %s, X is shape %s)' % (str(Y.shape), str(X.shape))
            assert Y.shape[1] == 1, 'Y must have width 1. (Y is shape %s)' % str(Y.shape)
            assert X.shape[1] == 3, 'X must have width 3. (X is shape %s)' % str(X.shape)
            assert np.all(X[:,0]==1.0), 'X must have 1.0 in the first column of every row.'
            self.n = Xn
            self.Y = Y
            self.X = X
        if verbose:
            self.verbose = True
        else:
            self.verbose = False
        if lambda_smoothing is not None:
            self.lambdasmooth = lambda_smoothing

    def save_as_pickle_file(self, file_path=None):
        '''Saves this object as a pickle file rather than a giant numpy array. (Or tries to anyway, I'm not sure
        if this function works or has been tested). Basically just saves a python dictionary of the most important
        attributes.
        '''
        if file_path is not None:
            self.self_pickle_file_path = file_path
        if self.self_pickle_file_path is None:
            raise FileNotFoundError
        tps_data = {
            'X': self.X,
            'Y': self.Y,
            'n': self.n,
            'K': self.K,
            'lambdasmooth': self.lambdasmooth,
            'w_hat': self.w_hat,
            'a_hat': self.a_hat,
            'Y_hat': self.Y_hat,
            'e_hat': self.e_hat,
            'std_err_resid': self.std_err_resid,
            'Xnew': self.Xnew,
            'Ynew_hat': self.Ynew_hat,
            'Knew': self.Knew,
            'self_pickle_file_path': self.self_pickle_file_path
        }
        io_save_to_pickle(tps_data, self.self_pickle_file_path)

    def load_from_pickle_file(self, file_path=None):
        '''Loads the data for a previously created TPS from a pickle file. Must be in the same form as saved
        by self.save_to_pickle_file.
        '''
        if file_path is not None:
            self.self_pickle_file_path = file_path
        if self.self_pickle_file_path is None:
            raise FileNotFoundError
        tps_data = io_load_from_pickle(self.self_pickle_file_path)
        self.X = tps_data['X']
        self.Y = tps_data['Y']
        self.n = tps_data['n']
        self.K = tps_data['K']
        self.lambdasmooth = tps_data['lambdasmooth']
        self.w_hat = tps_data['w_hat']
        self.a_hat = tps_data['a_hat']
        self.Y_hat = tps_data['Y_hat']
        self.e_hat = tps_data['e_hat']
        self.std_err_resid = tps_data['std_err_resid']
        self.Xnew = tps_data['Xnew']
        self.Ynew_hat = tps_data['Ynew_hat']
        self.Knew = tps_data['Knew']

    def copy_from_existing_TPS(self, existing_TPS):
        '''
        Takes an existing ThinPlateSpline object and turns self into a copy.
        '''
        self.n = existing_TPS.n
        self.Y = existing_TPS.Y
        self.X = existing_TPS.X
        self.verbose = existing_TPS.verbose
        self.Kernel_Xnew = existing_TPS.Kernel_Xnew
        self.K = existing_TPS.K
        self.w_hat = existing_TPS.w_hat
        self.a_hat = existing_TPS.a_hat
        self.Y_hat = existing_TPS.Y_hat
        self.e_hat = existing_TPS.e_hat
        self.std_err_resid = existing_TPS.std_err_resid

    def compute_kernel_matrix(self):
        '''
        Computes the (N x N) kernel matrix for the training data points X_1, .... , X_N
        '''
        st = datetime.datetime.now()
        if self.verbose:
            print('Computing K matrix beginning at %s' % st.strftime(DATETIME_FMT_PRINTABLE), end = '', flush=True)

        self.K = tps_kernel_matrix(self.X[:,1:], self.X[:,1:])
        if self.verbose:
            et = datetime.datetime.now()
            print(', done... (took %s)' % (et - st))

    def solve_parameters(self):
        '''
        Solves the TP-Spline for the input values. Gets the W and A parameter vectors and additionally computes
        fitted Y values and residuals.
        '''
        if self.K is None:
            if self.verbose:
                print('Kernel matrix has not yet been computed. Doing that first.')
            self.compute_kernel_matrix()
        st = datetime.datetime.now()
        if self.verbose:
            print('Solving thin-plate spline params, beginning at %s' % st, end='', flush=True)

        L = np.vstack((np.hstack((self.K + self.n * self.lambdasmooth * np.identity(self.n), self.X)),
                       np.hstack((self.X.T, np.zeros((3, 3), dtype=np.float64)))))
        wa_hat = np.linalg.inv(L).dot(np.vstack((self.Y, np.zeros((3, 1), dtype=np.float64))))
        self.w_hat = wa_hat[:self.n]
        self.a_hat = wa_hat[self.n:]
        self.fit_Y_hat()

        if self.verbose:
            et = datetime.datetime.now()
            print(', done... (took %s)' % (et - st))

    def fit_Y_hat(self):
        '''
        For all the values of the training data, computes the 'fitted' predicted value 'Y-hat'. Also computes the
        residual 'e_hat' and the model's standard error.
        :return:
        '''
        self.Y_hat = self.X.dot(self.a_hat) + self.K.dot(self.w_hat)
        self.e_hat = self.Y - self.Y_hat
        self.std_err_resid = np.std(self.e_hat).item()

    def report_diagnostics(self):
        '''
        This is a work in progress. Essentially should be sort of like a 'summary' function in R. At the very least
        we want sample size, std err and that sort of stuff.
        '''
        print('Sample Size:    %s' % self.n)
        print('Mean Residual:  %s' % np.mean(self.e_hat).item())
        print('Residual S.D.:  %s' % np.std(self.e_hat).item())

    def predict(self, Xnew, store_XYKnew = False):
        '''
        Computes predicted value of the spline at a new point(s) Xnew. Xnew must be in the same format at self.X (i.e.
        for K different points it should be (K x 3) with each row in homogenous coordinates: [1.0, x, y]). Since a set
        of predictions can range in its importance to the model itself, there is the option to store this information with
        the model so it may not need to be calculated again. There is also a function clear_predictions() to remove any
        old predictions.

        :param Xnew: (K x 3) numpy array of points in X-domain for which to compute fitted value.
        :return: (K x 1) numpy array of predicted values
        '''
        k = Xnew.shape[0]
        assert Xnew.shape[1] == 3, 'variable Xnew must have width 3'
        Kmat_Xnew = tps_kernel_matrix(Xnew[:,1:], self.X[:,1:])

        Ynew_hat = Xnew.dot(self.a_hat) + Kmat_Xnew.dot(self.w_hat)
        if store_XYKnew:
            self.clear_predictions()
            self.Xnew = Xnew
            self.Ynew_hat = Ynew_hat
            self.Knew = Kmat_Xnew
        else:
            return Ynew_hat

    def clear_predictions(self):
        '''Removes old values of self.Xnew, self.Ynew_hat, self.Knew'''
        if self.Xnew is not None:
            del self.Xnew; self.Xnew = None;
        if self.Ynew_hat is not None:
            del self.Ynew_hat; self.Ynew_hat = None;
        if self.Knew is not None:
            del self.Knew; self.Knew = None;

    def grid_predict(self, lon_resolution = 1.0, lat_resolution = 1.0, map_bounds_NSWE=None):
        '''
        Identifies a proper grid of points to serve as prediction points for mapping the spline's predicted
        vTEC values. Grid should ideally not go far outside the range of the training data. Points are spaced
        every 'lon_resolution' in the horizontal direction and 'lat_resolution' vertically.
        '''
        if map_bounds_NSWE is not None:
            self.map_bounds_NSWE = map_bounds_NSWE
        prediction_grid_point_list = []
        self.lons_list, self.lats_list, lon_res, lat_res= get_lat_lon_lists(lon_resolution, lat_resolution, self.map_bounds_NSWE)
        n_lons = len(self.lons_list); n_lats = len(self.lats_list);

        for mylat in self.lats_list:
            mylons = get_lons_in_Xrange_for_horizontal_band(self.X, mylat, lat_resolution, self.lons_list, lon_resolution)
            prediction_grid_point_list += list(map(lambda x: (1., x, mylat), mylons))

        self.prediction_grid_points = np.array(prediction_grid_point_list, dtype=np.float64)
        self.prediction_grid_values = self.predict(self.prediction_grid_points)
        self.prediction_grid_image = np.empty((n_lats, n_lons), dtype=np.float64)
        self.prediction_grid_image[:,:]=np.nan

        # fill in the prediction image:
        for i in range(self.prediction_grid_points.shape[0]):
            img_r = self.lats_list.index(self.prediction_grid_points[i,2].item())
            img_c = self.lons_list.index(self.prediction_grid_points[i,1].item())
            self.prediction_grid_image[img_r, img_c] = self.prediction_grid_values[i]


def bias_multi_tps_get_data_for_time(scenario, base_tick, stns, knot_delays, unknot_delays, prns=GPS_PRNS):
    '''Function to get the data for a thin-plate spline at a single time tick, plus
    the tick following it by "knotdelay". See multiple TPS notes for details.

    For a single tick 'tval', this searches through station_vtecs for valid station-satellite pairs
    and grabs the relevant data, including geographic position of the thin-shell pierce point, etc...
    It also does this for each of the values in `knot_delays` and `unknot_delays` respectively and
    adds that to the result. The values for knotted are separate from un-knotted in the return object.

    Returns a dictionary (see bottom of function for specs).'''

    # **** Helper function depends on data structure ****
    if scenario.station_data_structure=='dense':
        def get_data_point(scenario, tickpair, tick, z_index):
            '''helper function to get the information of interest from the station_vtecs file (in new dense numpy structure).'''
            s = tickpair[0]; p = tickpair[1];
            vtec_meas = scenario.get_measure(s, p, tick, vtec=True)
            ecef = (vtec_meas['ion_loc_x'], vtec_meas['ion_loc_y'], vtec_meas['ion_loc_z'] )
            vtec = vtec_meas['raw_vtec']; slant = vtec_meas['s_to_v'];
            lat, lon, _ = ecef2geodetic(ecef)
            prn_index = z_index.index(p)
            stn_index = z_index.index(s)
            return lat, lon, vtec, slant * DK, prn_index, stn_index
    else:
        def get_data_point(scenario, tickpair, tick, z_index):
            '''helper function to get the information of interest from the station_vtecs file (in old dict structure).'''
            s=tickpair[0]; p=tickpair[1]
            ecef = scenario.station_vtecs[s][p][0][tick]
            lat, lon, _ = ecef2geodetic(ecef)
            vtec=scenario.station_vtecs[s][p][1][tick]
            slant=scenario.station_vtecs[s][p][2][tick]
            prn_index = z_index.index(p)
            stn_index = z_index.index(s)
            return lat, lon, vtec, slant*DK, prn_index, stn_index

    tickpairs = []
    tickpairs_un = []
    z_size = len(prns)+len(stns)
    z_index = prns + stns
    if scenario.station_data_structure=='dense':
        for s in stns:
            for p in prns:
                for d in knot_delays:
                    if scenario.check_sta_prn_tick_exist(s, p, base_tick + d) and \
                            not np.isnan(scenario.get_measure(s,p,base_tick + d, vtec=True)['raw_vtec']):
                        tickpairs.append((s, p, base_tick + d))
                for d in unknot_delays:
                    if scenario.check_sta_prn_tick_exist(s, p, base_tick + d) and \
                            not np.isnan(scenario.get_measure(s,p,base_tick + d, vtec=True)['raw_vtec']):
                        tickpairs_un.append((s, p, base_tick + d))
    else:
        for s in stns:
            for p in prns:
                for d in knot_delays:
                    if len(scenario.station_vtecs[s][p][1])>base_tick+d and not math.isnan(scenario.station_vtecs[s][p][1][base_tick + d]):
                        tickpairs.append((s, p, base_tick + d))
                for d in unknot_delays:
                    if len(scenario.station_vtecs[s][p][1])>base_tick+d and not math.isnan(scenario.station_vtecs[s][p][1][base_tick + d]):
                        tickpairs_un.append((s, p, base_tick + d))

    n_knot = len(tickpairs)
    n_unkn = len(tickpairs_un)
    TPkn = np.array(tickpairs, dtype=np.string_)
    TPun = np.array(tickpairs_un, dtype=np.string_)
    Xkn = np.zeros((n_knot, 3), dtype=np.float64)
    Xun = np.zeros((n_unkn, 3), dtype=np.float64)
    Ykn = np.zeros((n_knot, 1), dtype=np.float64)
    Yun = np.zeros((n_unkn, 1), dtype=np.float64)
    Zkn = np.zeros((n_knot, z_size), dtype=np.float64)
    Zun = np.zeros((n_unkn, z_size), dtype=np.float64)
    Xkn[:,0]=1.; Xun[:,0]=1.0;

    for i in range(n_knot):
        dt = get_data_point(scenario, tickpairs[i], tickpairs[i][2], z_index)
        Xkn[i,1] = dt[1]; Xkn[i, 2] = dt[0];
        Ykn[i,0] = dt[2];
        Zkn[i, dt[4]] = -1.0 * dt[3]
        Zkn[i, dt[5]] = 1.0 * dt[3]
    for i in range(n_unkn):
        dt = get_data_point(scenario, tickpairs_un[i], tickpairs_un[i][2], z_index)
        Xun[i,1] = dt[1]; Xun[i, 2] = dt[0];
        Yun[i,0] = dt[2];
        Zun[i, dt[4]] = -1.0 * dt[3]
        Zun[i, dt[5]] = 1.0 * dt[3]

    Kkn = tps_kernel_matrix(Xkn,Xkn)
    Kun=tps_kernel_matrix(Xun, Xkn)

    return_args = {
        'z_index':z_index, 'n_knot':n_knot, 'n_unkn':n_unkn,
        'Xkn':Xkn, 'Xun':Xun, 'Ykn':Ykn, 'Yun':Yun, 'Zkn':Zkn, 'Zun':Zun,
        'Kkn':Kkn, 'Kun':Kun, 'TPkn': TPkn, 'TPun': TPun
    }
    return return_args

def bias_multi_tps_parameters_count_prns(z_ind, prns=GPS_PRNS):
    '''Simple helper function to count how many Satellites are left in the parameter index. This can vary because
    some satellites may have all-zero data or something like that, so this is really used to tell us where in the
    parameter index vector the station-bias parameters start.'''
    return len(list(set(prns).intersection(set(z_ind))))

def bias_multi_tps_fix_Z_matrix(Z, z_index, prns=GPS_PRNS, verbose=VERBOSE):
    '''Does some fixes to the Z matrix. First removes any all-zero columns. Then Sets the baseline PRN and Rec.
    values, including removing the baseline Rec column (i.e. the last one) and setting the baseline PRN column (i.e.
    the first one) to be an intercept value. Adjusts z_index accordingly.'''

    # Make sure no faulty columns are included in Z, remove names from z_index
    if np.where(np.sum(Z,0)==0.)[0].shape[0]  > 0:
        Z_nullcols = np.where(np.sum(Z,0)==0.)[0]
        Z=np.delete(Z,Z_nullcols,1)
        Z_nullcols_names = list(map(lambda x: z_index[x], tuple(Z_nullcols)))
        if verbose:
            print('Removing the following all-zero columns from Z: %s' % str(list(zip(Z_nullcols,Z_nullcols_names))))
        for znm in Z_nullcols_names:
            z_index.remove(znm)
    #
    # Set stations to make Z matrix full rank (b_sats will sum to 0). Basically we remove the last column of Z and
    #   subtract it from every other `station` column. This way the betas for the stations sum to 0.
    stn_cols_init = np.array([i for i in range(len(z_index)) if z_index[i] not in prns]) #indices of station columns
    base_stn = stn_cols_init[-1]; base_stn_ind = z_index[-1];    #Baseline station column (last one)
    stn_cols_final = stn_cols_init[:-1]
    stn_count = stn_cols_final.shape[0]
    base_stn_rows = np.where(Z[:, -1] != 0.)[0]; base_stn_rows_ct=base_stn_rows.shape[0]; #Baseline station rows
    base_stn_values = Z[base_stn_rows, -1]*-1.0
    # For each row that is in the baseline category, set every other column in the row equal to -1.0 x (last column)
    #   because the betas sum to 0.
    Z[np.repeat(base_stn_rows, stn_count), np.tile(stn_cols_final, base_stn_rows_ct)]=np.repeat(base_stn_values, stn_count)
    Z = np.delete(Z,-1,1)
    z_prn_count = bias_multi_tps_parameters_count_prns(z_index)
    z_stn_count = len(z_index)-z_prn_count-1
    z_size_total = len(z_index)-1
    if verbose:
        print('Z rank %s, Z shape %s' % (np.linalg.matrix_rank(Z), str(Z.shape)))
    return Z, z_prn_count, z_stn_count, z_size_total, base_stn_ind

def bias_multi_tps_params_to_csv(params_b, z_index, prns=GPS_PRNS, filepath=None, pretty=False):
    zprns = [i for i in z_index if i in prns]
    zprn_inds = [z_index.index(i) for i in zprns]
    nlines = len(zprns)

    if pretty:
        lns = list(map(lambda x: '%6s%9s  |  ' % (zprns[x], ('%.3f' % params_b[x].item())), zprn_inds))
    else:
        lns = list(map(lambda x: '%s,%.18e,,' % (zprns[x], params_b[x]), zprn_inds))
    for i in range(nlines, len(z_index)):
        if pretty:
            lns[(i - nlines) % nlines] += ('%6s%9s  ' % (z_index[i], ('%.3f' % params_b[i].item())))
        else:
            lns[(i - nlines) % nlines] += ('%s,%.18e,' % (z_index[i], params_b[i]))
    if filepath is None:
        print('\n'.join(lns))
    else:
        with open(filepath, 'w') as myf:
            cct = myf.write('\n'.join(lns))
    return

def bias_multi_tps_solve(scenario, first_tick=0, knot_tick_gaps=[0, ], no_knot_tick_gaps=[10, ],
                         btw_groups_delay=120*2, lambdasmooth=0.1, num_groups=12, prns=GPS_PRNS,
                         stns=None, return_full_model=False, verbose=VERBOSE, use_sparse=True):
    '''
    Gets an estimate of the Sat & Recv biases using a multiple thin-plate spline model. The idea is to take several
    time points from within a few days; close enough that the biases should be the same but far enough apart that the
    true vTEC values would have shifted. The model uses a separated thin-plate spline setup with a set of shared
    parameters representing the bias. I have the solution worked out in my notes. The idea here is that in order to
    solve for the extra parameters, we have to use a smaller number of spline knots than are available. To do this,
    For each "time-point" we actually use two, a short (e.g. 5-minute) time apart. The first one is used as the knot
    points, the second is not. Spreading them out a little bit gives some geographic separation of the data points but
    keeping them that close allows the assumption that the vTEC values don't change in between.
    Notes:
        - Generally here, the suffix "kn" refers to something on the "knotted" data points and "un" is on the
            "un-knotted" points. Without either suffix typically means it's both combined, but who knows if that's
            been done consistently.
        - Notation used in my notes and carried through here:
            n = n_kn + n_un --> # of data points, respectively in total, in all the knotted sections and the un-knotted
                                sections. Separately, the average number of points-per-group is important so I'll just
                                call that n_group. Running time is approximately Quadratic or Cubic in n_group (Cubic
                                really, but those routines are in numpy so it's populating the Kernel matrix that takes
                                forever in this implementation, and that's technically Quadratic.)
            m        -->    # of groups. Herein usually just the variable num_groups. So 'n' is approximately m times
                            n_group.
            p = p_sat + p_rec   --> # of shared parameters to solve for. Specifically, # satellites + # receivers.
            [w, a, b]'  --> The parameter vector to be solved. In the single TPS model we just had [w, a], but here
                            The matrix Z is multiplied by b. Also a has size (3 * m), w is
            TP  = "tick-pairs", namely the set of (station, PRN) pairs that are active for a particular tick. In
                            practice though this refers to the entire list of them. But it is a helpful reference
                            because it's the only record of which data-point came from where.

    Algorithm steps as noted below:
        1) Go into the station_vtecs object and pull out all the data specified by the inputs to this function for
            each layer separately. Store this data in mini versions of the X/K/Z/Y matrices to be used later.
        2) Combine the per-layer data into the basic inputs: X, K, Z, Y and X_knotted.
        3) Run some corrections on Z. Z may have all-zero columns in it which is a problem. Also we have to remove one
            of the station columns to make it full rank. By convention, the sum of the station biases is 0.
        4) Compute the solutions for every parameter of the model. Mostly we are interested in the bais coefficients
            though of course. There are two ways to do this, the sparse and the dense computation. There is a small
            difference in running time but the memory footprint is about half with the sparse, so that is the default.
        5) Finish up, separate the parameter vector into its components. Populate some model metadata for the output.

    :param station_vtecs:
    :param first_tick:      (int) Tick for the first time-point in the series.
    :param knot_tick_gaps:  (list of int) A list of gaps from the first tick point for each layer for which to add
                                knotted points. Really this should just be [0,], but if for some reason you want to ahve
                                two different sets of locations under the assumption of the same vTEC map, adding extra
                                points to a single map may make sense.
    :param no_knot_tick_gaps:  (list of int) A list of gaps from the first tick point for each layer for which to add
                                points without knots. Should probably correspond to the number of knot_tick_gaps.
                                EXAMPLE: if knot_tick_gaps=[0,20] and no_knot_tick_gaps=[10,30], then each layer is
                                going to have knotted points drawn from <tick_start> and <tick_start+20>, and will have
                                un-knotted points from <tick_start+10> and <tick_start+30>.
    :param btw_groups_delay: (int) separation between time points for the course of the day. These must be spread out
                            enough that every PRN is captured and (ideally) every station-PRN combo shows up.
    :param lambdasmooth:    (float) Smoothing parameter for the thin-plate spline (default is 0.1)
    :param num_groups:      (int) Number of time points to use. Specific ticks to mark each time point will be chosen
                            based on the math of (first_tick + (i)*btw_groups_delay , i = 1, ..., num_groups).
    :param use_sparse:      (boolean) If True, computations are done using a parsimonious approach to memory,
                                specifically using scipy sparse arrays in one particular case to avoid an explosion of
                                memory usage at the expense of longer computing time. For 24 layers, 483 stations, 1
                                tick each (knotted/un-knotted) per layer, one a 20-core machine with 256gb RAM, the sparse
                                version ran in 8:34 and topped out at 95.7gb memory (143 virtual) (that includes
                                station_vtecs objects and other process overhead). The dense version ran in 4:05 and used
                                160.9GB at its peak (256gb virtual).

    :return bias_parameters: (np array). Array of model parameters representing the satellite and receiver bias
                            calculations. (I.e. the thing we're after).
    :return parameter_lookup:   (list of strings). Same lengths as 'bias_parameters'. Used as a reference to map the
                            values back to the Sat/Recv each one belongs to.
    :return model_info:     (dict) Depending on the value of 'return_full_model' this can contain a lot or a little
                            bit of information about the model and intermediate values as it was run. At a bare min
                            it contains a dictionary objet with the arguments provided and the list of stations
                            found in 'station_vtecs'.

    '''
    v_chkpt=DEBUG
    run_start = datetime.datetime.now()
    if stns is None:
        stns = list(scenario.station_vtecs.keys())
    else:
        stns = list(set(stns).intersection(set(scenario.station_vtecs.keys())))
    z_size = len(prns) + len(stns)
    z_index = prns + stns
    info = {'first_tick': first_tick, 'knot_tick_gaps': knot_tick_gaps,  'no_knot_tick_gaps': no_knot_tick_gaps,
            'btw_groups_delay': btw_groups_delay, 'lambdasmooth': lambdasmooth, 'num_groups': num_groups,
            'stations': stns, 'num_stations_init': len(stns), 'run_start_time': run_start.strftime(DATETIME_FMT_PRINTABLE)}

    # This variable will become a dictionary of dictionaries (hence 'dd') with an outer key for each of the groups. See
    # the function bias_multi_tps_get_data_for_time(...) for what it gathers.
    dd = dict.fromkeys(range(num_groups))
    array_list={}


    #****** 1) Get the data for the spline model, one layer at a time
    for i in range(0, num_groups):
        dd[i]= bias_multi_tps_get_data_for_time(scenario, i * btw_groups_delay, stns, knot_tick_gaps,
                                                no_knot_tick_gaps)
        n_knot = dd[i]['n_knot']
        Ki = np.vstack((dd[i]['Kkn'] + n_knot * lambdasmooth * np.identity(n_knot), dd[i]['Kun']))
        Xi = np.vstack((dd[i]['Xkn'], dd[i]['Xun']))
        dd[i]['Ablock'] = Ki.T.dot(Ki) + dd[i]['Xkn'].dot(dd[i]['Xkn'].T)
        dd[i]['Ainv'] = np.linalg.inv(dd[i]['Ablock'])
        dd[i]['X'] = Xi
        dd[i]['K'] = Ki
        dd[i]['Z'] = np.vstack((dd[i]['Zkn'], dd[i]['Zun']))

    n_kn_all = sum(map(lambda x: dd[x]['n_knot'], dd.keys()))
    n_un_all = sum(map(lambda x: dd[x]['n_unkn'], dd.keys()))
    n_all = n_kn_all + n_un_all
    if verbose:
        print('(Nkn, Nun, Nall)=(%s,%s,%s), m=%s (3m=%s), p=%s' % (n_kn_all, n_un_all, n_all, num_groups
                                                               , num_groups * 3, z_size))
    # The separated nature of the splines means that a lot of the matrices we'll use are block diagonal, in particular
    #   the K matrix, which is important because that makes inverting it much faster.
    # Here, Y is the output, X refers to the lat-lon in homogenous-coordinate format. Z is the matrix of D/K*(b_rec -
    #   b_sat) from the bias calculations. The first 32 columns (less of one of the PRNs has no data) are the Sats, the
    #   rest are the CORS. The position of each is catalogued by the variable 'z_index' created earlier.

    #****** 2) Combine the data per layer into the basic inputs
    Y = np.vstack(tuple(map(lambda x: np.vstack((dd[x]['Ykn'], dd[x]['Yun'])), range(num_groups))))
    X = sp.linalg.block_diag(*tuple(map(lambda x: np.vstack((dd[x]['Xkn'], dd[x]['Xun'])), range(num_groups))))
    Xkn = sp.linalg.block_diag(*tuple(map(lambda x: dd[x]['Xkn'], range(num_groups))))
    Z = np.vstack(tuple(map(lambda x: np.vstack((dd[x]['Zkn'], dd[x]['Zun'])), range(num_groups))));
    z_index = prns + stns;
    TP = np.vstack(tuple(map(lambda x: np.vstack((dd[x]['TPkn'], dd[x]['TPun'])), range(num_groups))));

    # ****** 3) Fix Z matrix: remove all-zero columns and set it so the sat-biases sum to 0 (Springer handboox, Sec 31)
    Z, psat, prec, ptot, blCORS = bias_multi_tps_fix_Z_matrix(Z, z_index, verbose=False)
    array_list.update({'X': X, 'Xkn': Xkn, 'Y': Y, 'Z': Z})

    # Lots of sanity checking:
    if verbose:
        print('Z structure: # Satellite Columns = %s, # Receiver Columns = %s. (Total P = %s)' % (psat, prec, ptot))
        print('Shapes of... [Y: %s, X: %s, Z: %s]' % (str(Y.shape), str(X.shape), str(Z.shape)))
        print('Rank Checks: X --> rank=%s, width=%s (***%s***),  Z --> rank=%s, width=%s (***%s***)' % (
            np.linalg.matrix_rank(X), X.shape[1], str(np.linalg.matrix_rank(X) == X.shape[1]).upper(),
            np.linalg.matrix_rank(Z), Z.shape[1], str(np.linalg.matrix_rank(Z) == Z.shape[1]).upper()))

    # Here the L-matrix is [[ K, X, Z], [Xkn', 0, 0]] and we have to solve (L'L)^-1 (L'Y). To compute the inverse of
    #   L'L, we will turn it into a block-diagonal matrix with K'K+XknXkn' in the top left and other blocks combined
    #   accordingly. That top-left matrix is by far the largest and we can invert it by inverting the blocks one at
    #   a time. The variables below refer to a generic 2x2 block diagnoal matrix [[A, B],[C,D]] (see wikipedia). Here
    #   though, C = B' so we don't have a C.

    # ****** 4) Do the computations to solve for the bias parameters. We have a sparse and dense version:
    if not use_sparse:
        # ***** Dense Version *****
        Ktmp = sp.linalg.block_diag(*tuple(map(lambda x: dd[x]['K'], range(num_groups))))
        A_inv = sp.linalg.block_diag(*tuple(map(lambda x: dd[x]['Ainv'], range(num_groups))))
        B = Ktmp.T.dot(np.hstack((X, Z)))
        Droot = np.hstack((X, Z))
        D = Droot.T.dot(Droot)

        shape1 = (X.T.shape[0], Xkn.shape[1]); shape2 = (Z.T.shape[0], Xkn.shape[1]);
        L_T = np.vstack((np.hstack((Ktmp.T, Xkn)),
                         np.hstack((X.T, np.zeros(shape1, dtype=np.float64))),
                         np.hstack((Z.T, np.zeros(shape2, dtype=np.float64)))))

        # delete Ktmp
        Ktmp_size_str = get_nparray_size_data(Ktmp, True); del Ktmp;

        BtAinv = B.T.dot(A_inv)
        BtAinvB = BtAinv.dot(B)
        AinvB = A_inv.dot(B)

        # force symmetric
        Dnew_inv = D - (BtAinvB + BtAinvB.T) / 2.
        Dnew = np.linalg.inv(Dnew_inv)
        AinvBDnew = AinvB.dot(Dnew)
        Cnew = -1.0 * Dnew.dot(BtAinv)
        Bnew = -1.0 * AinvBDnew

        # LtL_inv = np.block([[Anew, Bnew], [Cnew, Dnew]]);
        LtL_inv = np.block([[A_inv + (AinvBDnew).dot(BtAinv), Bnew], [Cnew, Dnew]]);
        del A_inv # A_inv no longer needed, so clear out some memory

        Y_full = np.vstack((Y, np.zeros((3 * num_groups, 1), dtype=np.float64)))
        Lt_Y = L_T.dot(Y_full)
        params = LtL_inv.dot(Lt_Y)
    else:
        # ***** Sparse Version *****
        Ktmp = sp.linalg.block_diag(*tuple(map(lambda x: dd[x]['K'], range(num_groups)))); #checkpoint('Made ktmp (721) %s' % get_nparray_size_data(Ktmp,True),v_chkpt)
        A_inv = sp.linalg.block_diag(*tuple(map(lambda x: dd[x]['Ainv'], range(num_groups)))); #checkpoint('Made A_inv (722) %s' % get_nparray_size_data(A_inv,True),v_chkpt)
        B = Ktmp.T.dot(np.hstack((X, Z))); #checkpoint('Made B (723)',v_chkpt)
        Droot = np.hstack((X, Z)); #checkpoint('Made Droot (724)',v_chkpt)
        D = Droot.T.dot(Droot); #checkpoint('Made D (725)',v_chkpt)
        L_T_csr = sp.sparse.bmat([[sp.sparse.csr_matrix(Ktmp.T), sp.sparse.csr_matrix(Xkn)],
                                  [sp.sparse.csr_matrix(X.T), None],[sp.sparse.csr_matrix(Z.T), None]], format='csr')
        # checkpoint('Made L_T_csr (728) %s' % get_nparray_size_data(L_T_csr,True), v_chkpt, 10)
        # array_list.update({'Ktmp': Ktmp,'A_inv': A_inv,'B': B,'Droot': Droot,'D': D,'L_T_csr': L_T_csr})
        # print_nparray_sizes(array_list); checkpoint('Printed array sizes (730)', v_chkpt, 10)

        # delete Ktmp
        Ktmp_size_str = get_nparray_size_data(Ktmp, True); del Ktmp;
        # checkpoint('deleted Ktmp %s' % Ktmp_size_str, v_chkpt, 5)

        BtAinv = B.T.dot(A_inv); #checkpoint('done with BtAinv [%s X %s]' % (str(B.T.shape), str(A_inv.shape)),v_chkpt)
        BtAinvB = BtAinv.dot(B); #checkpoint('done with BtAinvB [%s X %s]' % (str(BtAinv.shape), str(B.shape)),v_chkpt)
        AinvB = A_inv.dot(B); #checkpoint('done with AinvB [%s X %s]' % (str(A_inv.shape), str(B.shape)),v_chkpt)

        Dnew_inv = D - (BtAinvB + BtAinvB.T) / 2.  # force symmetric
        # checkpoint('done with Dnew_inv (shape = %s)' % str(Dnew_inv.shape),v_chkpt)
        Dnew = np.linalg.inv(Dnew_inv); #checkpoint('done with Dnew',v_chkpt)
        AinvBDnew = AinvB.dot(Dnew); #checkpoint('done with AinvBDnew',v_chkpt)
        Cnew = -1.0 * Dnew.dot(BtAinv); #checkpoint('done with Cnew',v_chkpt)
        Bnew = -1.0 * AinvBDnew; #checkpoint('done with Bnew',v_chkpt)

        # LtL_inv = np.block([[Anew, Bnew], [Cnew, Dnew]]);
        LtL_inv = np.block([[A_inv + (AinvBDnew).dot(BtAinv), Bnew], [Cnew, Dnew]]);

        #delete A_inv
        A_inv_size_str = get_nparray_size_data(A_inv, True); del A_inv;

        Y_full = sp.sparse.csr_matrix(np.vstack((Y, np.zeros((3 * num_groups, 1), dtype=np.float64))))
        Lt_Y_csr = L_T_csr.dot(Y_full)
        params = LtL_inv.dot(Lt_Y_csr.toarray())

    # ****** 5) Wrap everything up and output results.
    # Seaprate out the results. The beta parameters are the ones were really interested in here:
    params_w = params[:n_kn_all, :] # size = N_all
    params_a = params[n_kn_all:n_kn_all + 3 * num_groups, :] # size = (3 * m)
    params_b = params[n_kn_all + 3 * num_groups:, :]    # size = p
    params_b = np.vstack(
        (params_b, -np.sum(params_b[psat:]).reshape((1, 1))))  # Add back the contrast for the last CORS bias

    total_time = datetime.datetime.now()-run_start
    info['run_time_sec']=total_time

    # Decide what to return and then exit:
    if return_full_model:
        info.update({
            'dd': dd, 'Y': Y, 'X': X, 'Xkn': Xkn, 'Z': Z, 'z_index': z_index, 'TP': TP,
            'LtL_inv': LtL_inv, 'Nall': n_all, 'Nkn': n_kn_all, 'Nun': n_un_all,
            'params_w': params_w, 'params_a': params_a, 'params_b': params_b,
            'num_stations_final': prec, 'N_group_avg': float(n_all/num_groups),
            'Nkn_group_avg': float(n_kn_all/num_groups), 'Nun_group_avg': float(n_un_all/num_groups)
        })

    return params_b, z_index, info, dict(zip(z_index, params_b[:,0]))
