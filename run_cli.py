"""
Provides a command-line interface to the tid library with basic commands to create a scenario, plot a scenario,
etc...
"""


import argparse, sys, os, datetime, logging, math

from pathlib import Path
from typing import Optional

from laika import AstroDog

from tid import plot, util, scenario
from tid.config import Configuration

logger = logging.getLogger(__name__)

# load configuration data
conf = Configuration()


def get_station_network_list(network=None):
    """Helper function to get the list of networks included in the 'station_networks.json' file. If the input argument
    `network` is given, returns a list of station names for that network. If it is omitted, returns a list of the
    available networks in the .json file."""
    mydir = os.path.split(os.path.abspath(__file__))[0]
    snfile = os.path.join(mydir, "tid", "lookup_tables", "station_networks.json")
    import json

    with open(snfile, "r") as jsf:
        station_networks = json.load(jsf)
    if network is None:
        return sorted(list(set(station_networks.values())))
    else:
        return sorted(
            [
                i
                for i in station_networks
                if station_networks[i].lower() == network.lower()
            ]
        )

def read_station_list_file(f):
    """Reads a comma-and/or-newline delimited list of station names from a read-only opened file object."""
    lns = f.readlines()
    stn_list = []
    for ln in lns:
        csep = ln.strip().split(",")
        for i in csep:
            if i != "":
                stn_list.append(i)
    f.close()
    return stn_list

def convert_scenario_definition_args(args):
    '''Helper function to run the logic of parsing the various command-line options that could define a scenario,
    i.e. could be given by start-date/duration/station-list or by station-network. Does converting to proper types
    and whatnot and returns cache_key and path.'''
    if not (args.start_date is not None and args.duration_days is not None and (args.station_network is not None or args.station_list is not None)):
        logger.error(f'Command line arguments must include start_date, duration and either station_network or station_list.')
        sys.exit(0)

    start_date = util.datetime_fromstr(args.start_date)
    duration = args.duration_days * util.DAYS
    if args.station_network != None:
        station_list = get_station_network_list(args.station_network)
    else:
        station_list = read_station_list_file(args.station_list)

    cache_key = scenario.Scenario.compute_cache_key(start_date, duration, station_list)
    cache_filepath = Path(conf.cache_dir) / "scenarios" / f"{cache_key}.hdf5"

    return start_date, duration, station_list, cache_key, cache_filepath

def main():
    """Main function to parse arguments and use function callback to direct sub-commands to the correct
    subroutine."""
    p = argparse.ArgumentParser()

    # Generic arguments here include the four scenario-definition parameters (start/duration/station-(net/list)), but
    #   which are all optional so algorithms that don't require them at all can just ignore them.
    p.add_argument("-st", "--start_date", dest="start_date", type=str, default=None,
                                 help="The first date of the scenario, formatted as 'YYYY-MM-DD'.")
    p.add_argument("-dur", "--duration_days", dest="duration_days", type=int, default=None,
                                 help="Duration of the scenario, in days.")
    p.add_argument("-net", "--station_network", dest="station_network", type=str, default=None,
                                 choices=get_station_network_list(),
                                 help='If given, the network of stations to use for the scenario (i.e. one of "Korea", '
                                      '"Japan", "Mongolia", or "US"). If this argument is given, \'--station_list\' is '
                                      "ignored.")
    p.add_argument("-stns", "--station_list", dest="station_list", type=argparse.FileType("r"), default=None,
                                 help="Path to a text file containing a list of stations to include. Text file should be "
                                      "formatted where either commas or newlines (or both) separate the stations.")

    subparsers = p.add_subparsers(
        help="The algorithm the command should run.", dest="algorithm"
    )

    # Algorithm 1: Construct a Scenario from a set of dates and a set of stations...
    p_create_scenario_file = subparsers.add_parser("create_scenario_file",
        help="Create a scenario file from a start-date, a duration, and a list of stations. "
        "The list of stations can be given either as a string representing a network "
        "or a file containing a comma/newline separated list of station names.")
    p_create_scenario_file.add_argument("--cache_file_to_stdout", dest="cache_file_to_stdout", action="store_true",
        help="If given, logging level is raised to critical errors only and the full "
        "path to the scenario cache file is printed to stdout. Useful for shell "
        "scripting.")
    p_create_scenario_file.set_defaults(func=create_scenario_from_dates_and_stations)

    # Algorithm 2: Plot a Scenario from a set of dates and a set of stations. If it has not been previously created, then
    #   create it first.
    p_plot_scenario = subparsers.add_parser("plot_scenario",
        help="Construct a plot from a scenario (given by either a start/duration/station-list or by a cache-key or "
             "cache file-path. Scenario start/duration/station-list argument-set is specified identically as in the "
             "\'create_scenario_file\' command. ")
    p_plot_scenario.add_argument("-c", "--cache_key", dest="cache_key", type=str, default=None,
        help="Cache key defining scenario (or equivalently, full absolute-path to a cached scenario file). If given, all "
             "other scenario-definition arguments are ignored (e.g. start_date, duration, network/station-list). ")
    p_plot_scenario.add_argument("-o", "--output_folder", dest="output_folder", type=Path, required=True,
        help="The directory in which to save plots")
    p_plot_scenario.add_argument("-f", "--output_file", dest="output_file", type=str, default='scenario',
        help="The file prefix to save the animations within the output folder. Files will be appended with \'_#\' "
             "as additional files are needed render the full length of the scenario.")
    p_plot_scenario.add_argument("-n", "--frames_per_plot", dest="frames_per_plot", type=int, default=240,
        help="Number of frames to save per plot.")
    p_plot_scenario.set_defaults(func=plot_scenario)

    # Algorithm 3: Compute cache key for scenario, print scenario summary data and file-exists info, exit:
    p_print_scenario_details = subparsers.add_parser("print_scenario_details",
        help="Takes the basic scenario parameters (start-date, duration, station-network/listfile) and prints some "
             "basic summary info about the scenario, namely the cache key and whether the cache file exists.")
    p_print_scenario_details.set_defaults(func=print_scenario_details)

    myargs = p.parse_args()
    myargs.func(myargs)

def print_scenario_details(args):
    '''
    Algorithm #3: Run the parser and print some details to the logger about the scenario, namely whether the cache
    file exists or not...
    '''
    start_date, duration, station_list, sc_hashkey, sc_cache_path = convert_scenario_definition_args(args)
    logger.info(f'Scenario: start_date={start_date:%Y-%m-%d}, duration={args.duration_days}, # stations: {len(station_list)}, ')
    logger.info(f'          cache_key={sc_hashkey}, cache_file_exits={os.path.isfile(sc_cache_path)}, ')
    logger.info(f'          cache_file_path: {sc_cache_path}')

def plot_scenario(args):
    '''
    Algorithm #2: 'plot_scenario'

    Take a previously created scenario object and process it (i.e. identify connections and solve biases),
    then create plot files in specified output directory.
    '''
    dog = AstroDog(cache_dir=conf.cache_dir)

    # First: retrieve the scenario file...
    if args.cache_key is not None:
        if os.path.isfile(args.cache_key):
            cache_file = args.cache_key
            sc = scenario.Scenario.from_hdf5(cache_file, dog=dog)
        elif os.path.isfile(Path(conf.cache_dir) / "scenarios" / f"{args.cache_key}.hdf5"):
            cache_file = Path(conf.cache_dir) / "scenarios" / f"{args.cache_key}.hdf5"
            sc = scenario.Scenario.from_hdf5(cache_file, dog=dog)
        else:
            logger.error(f'Could not fine cached file with cache-key or full path: {args.cache_key}')
            sys.exit(0)
    else:
        start_date, duration, station_list, sc_hashkey, sc_cache_path = convert_scenario_definition_args(args)
        logger.info(f'Computed cache_key: {sc_hashkey}')
        if os.path.isfile(sc_cache_path):
            sc = scenario.Scenario.from_hdf5(sc_cache_path, dog=dog)
        else:
            logger.info(f'Cached file for scenario does not exist, creating from command-line arguments...')
            sc = scenario.Scenario.from_daterange(start_date, duration, station_list, dog)

    # Second: Run plotting script as-in vandenberg demo:
    #   a) make sure output folder exists:
    if not args.output_folder.exists():
        args.output_folder.mkdir()

    #   b) Run two main scenario-solving steps:
    logger.info("Scenario initialized, creating connections")
    sc.make_connections()

    logger.info("Connections created, resolving biases")
    sc.solve_biases()

    logger.info("Preparing animation")
    # TODO: make the extents and frame-ranges calculate automatically or make them a CL input...
    extent = sc.get_extent()
    n_ticks = sc.tick_count
    for i in range(0, math.ceil(n_ticks/args.frames_per_plot)):
        anim = plot.plot_map(sc, extent=extent, frames=range(i*args.frames_per_plot, min((i+1)*args.frames_per_plot, n_ticks)))

        if args.output_folder:
            logger.info(f"Saving animation to {args.output_folder}: {args.output_file}_{i}.gif")
            plot.save_plot(anim, f'{args.output_file}_{i}', args.output_folder)

def create_scenario_from_dates_and_stations(args):
    """
    Algorithm #1: 'create_scenario_file'

    Run the download and scenario-creation for the start-date, duration and station-list given in
    the command-line inputs.
    """
    MAX_STATION_NAMES_TO_PRINT = 250
    # Determine whether to log at normal level or only critical errors:
    if args.cache_file_to_stdout:
        logger.setLevel(logging.CRITICAL)

    # Establish dog and main command-line arguments:
    dog = AstroDog(cache_dir=conf.cache_dir)
    start_date, duration, station_list, sc_hashkey, sc_cache_path = convert_scenario_definition_args(args)
    station_list_str = ",".join(station_list[:MAX_STATION_NAMES_TO_PRINT])

    logger.info(
        f"Creating the following scenario: Start={start_date:%Y-%m-%d}, Duration={args.duration_days}, # Stations={len(station_list)}, "
    )
    logger.info(
        f"Station List: {station_list_str}, ...({max(len(station_list)-MAX_STATION_NAMES_TO_PRINT,0)} more)..."
    )

    sc = scenario.Scenario.from_daterange(start_date, duration, station_list, dog)
    logger.info(f"Scenario Cache Key: {sc_hashkey} (file: {sc_cache_path})")
    logger.info("Done creating scenario.")

    if args.cache_file_to_stdout:
        cct = sys.stdout.write(f"{sc_cache_path}\n")

if __name__ == "__main__":
    main()
