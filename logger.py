import logging
import sys
from copy import copy

MAPPING = {
    'DEBUG': 37,     # white
    'INFO': 36,      # cyan
    'WARNING': 33,   # yellow
    'ERROR': 31,     # red
    'CRITICAL': 41,  # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'


class ColoredFormatter(logging.Formatter):

    def __init__(self, patern):
        logging.Formatter.__init__(self, patern)

    def format(self, record):
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37)      # default white
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(PREFIX, seq, levelname, SUFFIX)
        colored_record.levelname = colored_levelname
        return logging.Formatter.format(self, colored_record)


rootlog = logging.getLogger()
rootlog.setLevel(logging.DEBUG)
logging.root.manager.loggerDict["urllib3.connectionpool"] \
       .setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
handler.setFormatter(formatter)
cf = ColoredFormatter(
    "[%(name)s][%(levelname)s]  %(message)s (%(filename)s:%(lineno)d)"
    )
handler.setFormatter(cf)
rootlog.addHandler(handler)

# .setFormatter(formatter))
# logging.root.manager.loggerDict["urllib3.connectionpool"].addHandler(handler.setFormatter(cf))
