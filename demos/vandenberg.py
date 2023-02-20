"""
Plot data from a Falcon 9 launch into SSO from Vandenberg AFB
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

from laika import AstroDog

from tid import plot, util, scenario
from tid.config import Configuration

logger = logging.getLogger(__name__)

# load configuration data
conf = Configuration()


def main(output_path: Optional[Path] = None):
    # create our helpful astro dog
    dog = AstroDog(cache_dir=conf.cache_dir)

    # time of interest for our thing
    date = util.datetime_fromstr("2019-06-12")

    # near vandenberg
    # fmt: off
    ca_stations = [
        "srs1", "cru1", "vndp", "p513", "ozst", "copr",
        "ana1", "cat3", "sni1", "scip", "p172", "islk",
        "p563", "p591", "azu1", "vdcy", "p231", "p300",
        "p467", "sio5", "trak", "p588", "casn", "tono",
        "csst", "bar1", "p475", "tost", "ores", "p472",
        "p473", "p066", "p480", "p483", "p066", "p500",
        "p003", "gol2", "p600", "p463", "p572", "p091",
        "nvag", "p651", "p523", "jplm", "sbcc", "bill",
        "guax", "farb", "p277", "slac", "mhcb", "p217",
    ]
    # fmt: on

    logger.info("Starting scenario (downloading files, etc)")
    sc = scenario.Scenario.from_daterange(date, 1 * util.DAYS, ca_stations, dog)

    logger.info("Downloading complete, creating connections")
    sc.make_connections()

    logger.info("Connections created, resolving biases")
    sc.solve_biases()

    logger.info("Preparing animation")
    extent = (-128, -115, 29.6, 38.1)
    anim = plot.plot_map(sc, extent=extent, frames=range(1600, 1800))

    if output_path:
        logger.info(f"Saving animation to {output_path}")
        plot.save_plot(anim, "vandenburg", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output-path", help="The directory in which to save plots", type=Path
    )

    parser.add_argument("-v", "--verbose", action="count")
    args = parser.parse_args()

    log_map = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        level=log_map.get(args.verbose, conf.log_level),
        format="%(asctime)s [%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
        datefmt=conf.logging.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Default to configuration values if none specified at command line
    if (output_path := args.output_path) is None:
        output_path = Path("demos/outputs")

    # Check if output path exists first
    if not output_path.exists():
        output_path.mkdir()

    main(output_path)
