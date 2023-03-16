"""
Fetch data hourly and save it as an animation
"""
import datetime
import time
import sys

from laika import AstroDog
from laika.gps_time import GPSTime

from tid import plot, util, scenario
from tid.config import Configuration


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} output_folder [hours to run]")
    sys.exit(0)
output_folder = sys.argv[1]
if len(sys.argv) == 2:
    count = float("inf")
else:
    count = int(sys.argv[2])


# load configuration data
conf = Configuration()

# create our helpful astro dog
dog = AstroDog(cache_dir=conf.cache_dir)


# spread throughout Japan
# fmt: off
jp_stations = [
    "0002", "0010", "0016", "0020", "0021", "0022",
    "0023", "0024", "0026", "0027", "0033", "0035",
    "0036", "0037", "0043", "0047", "0050", "0055",
    "0059", "0061", "0066", "0067", "0068", "0075",
    "0089", "0093", "0094", "0099", "0102", "0103",
    "0106", "0114", "0125", "0135", "0135", "0142",
    "0143", "0144", "0163", "0164", "0175", "0176",
    "0180", "0181", "0193", "0197", "0199", "0199",
    "0201", "0206", "0212", "0218", "0225", "0230",
    "0235", "0242", "0244", "0246", "0257", "0258",
    "0263", "0265", "0268", "0291", "0295", "0303",
    "0305", "0310", "0318", "0318", "0322", "0323",
    "0324", "0331", "0341", "0348", "0351", "0356",
    "0359", "0360", "0364", "0373", "0380", "0399",
    "0428", "0439", "0443", "0448", "0448", "0450",
    "0456", "0457", "0459", "0460", "0474", "0476",
    "0480", "0483", "0486", "0492", "0493", "0496",
    "0514", "0523", "0525", "0553", "0561", "0564",
    "0575", "0575", "0578", "0578", "0579", "0581",
    "0582", "0585", "0586", "0595", "0597", "0612",
    "0618", "0621", "0623", "0625", "0628", "0640",
    "0641", "0642", "0652", "0656", "0666", "0668",
    "0676", "0679", "0680", "0689", "0697", "0698",
    "0699", "0702", "0703", "0710", "0731", "0741",
    "0745", "0749", "0756", "0761", "0765", "0766",
    "0774", "0785", "0790", "0792", "0794", "0795",
    "0800", "0807", "0807", "0810", "0812", "0819",
    "0823", "0825", "0829", "0835", "0838", "0852",
    "0858", "0867", "0869", "0876", "0877", "0888",
    "0888", "0902", "0904", "0909", "0915", "0918",
    "0931", "0945", "0947", "0959", "0965", "0970",
    "0972", "0989", "0990", "1000", "1002", "1002",
    "1011", "1014", "1016", "1030", "1035", "1036",
    "1051", "1059", "1061", "1066", "1068", "1069",
    "1073", "1075", "1076", "1077", "1085", "1087",
    "1091", "1095", "1096", "1112", "1126", "1130",
    "1145", "1149", "1150", "1155", "1162", "1167",
    "1167", "1170", "1171", "1172", "1173", "1185",
    "1195", "1203", "1207", "1208", "1214", "1216",
    "1218", "2005", "3004", "3005", "3018", "3019",
    "3020", "3039", "3045", "3051", "3053", "3060",
    "3062", "3064", "3065", "3066", "3077", "3079",
    "3084", "3091", "3093", "3097", "3104", "5113",
]
# fmt: on


def next_update(time: datetime.datetime):
    if time.minute < 6:
        return datetime.datetime(time.year, time.month, time.day, time.hour, 6)
    else:
        return (
            datetime.datetime(time.year, time.month, time.day, time.hour, 6)
            + util.HOURS
        )


next_time = next_update(datetime.datetime.utcnow())

tick = 0
while tick < count:
    # just run this to get sat info earlier, because it's slow -_-
    dog.get_all_sat_info(GPSTime.from_datetime(next_time))

    conf.logger.info(
        f"Waiting until next window ({next_time - datetime.datetime.utcnow()})"
    )
    while datetime.datetime.utcnow() < next_time:
        time.sleep(10)

    date = datetime.datetime(
        next_time.year, next_time.month, next_time.day, next_time.hour
    )
    conf.logger.info("Starting scenario (downloading files, etc)")
    sc = scenario.Scenario.from_daterange(
        date - util.HOURS, util.HOURS, jp_stations, dog, use_cache=False
    )

    conf.logger.info("Downloading complete, creating connections")
    sc.make_connections()

    conf.logger.info("Connections created, resolving biases")
    sc.solve_biases()

    conf.logger.info("Preparing animation")
    extent = (123, 149, 33, 48)

    ani = plot.plot_map(
        sc, extent=extent, frames=range(1, 119), raw=False, display=False
    )
    ani.save(
        f"{output_folder}/{date.strftime('%Y-%m-%d_%H')}_wide_short_borders_350km.mp4",
        dpi=350,
    )

    next_time = next_update(datetime.datetime.utcnow())
    tick += 1
