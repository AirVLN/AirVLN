import logging
import sys
from pathlib import Path
import os

from src.common.param import args


class AirsimLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)

        self._formatter = logging.Formatter(format_str, dateformat, style)

        if filename is not None:
            handler = logging.FileHandler(filename, filemode, encoding='utf-8')
            handler.setFormatter(self._formatter)
            super().addHandler(handler)

        handler = logging.StreamHandler(stream)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename, encoding='utf-8')
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


if not os.path.exists(Path(args.logger_file_name).parent):
    os.makedirs(Path(args.logger_file_name).parent, exist_ok=True)


if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    format_str = "{} - %(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s".format(
        int(os.environ.get("RANK", -1)),
    )

    args.logger_file_name = Path(args.logger_file_name).parent / '{}_{}.log'.format(str(Path(args.logger_file_name).stem), int(os.environ.get("RANK", -1)))
else:
    format_str = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"


logger = AirsimLogger(
    name="airsim",
    level=logging.INFO if int(os.environ.get("RANK", -1)) in [-1, 0] else logging.WARNING,
    format_str=format_str,
    stream=sys.stdout,
    filename=str(args.logger_file_name),
)
