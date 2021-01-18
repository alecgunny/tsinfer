import os
import re
import time
import typing
from itertools import cycle

import numpy as np

from tsinfer.pipeline.common import StoppableIteratingBuffer

if typing.TYPE_CHECKING:
    import multiprocessing as mp


__all__ = [
    "DataGeneratorBuffer",
    "GwpyTimeSeriesDataGenerator",
    "DummyDataGenerator",
    "LowLatencyFrameDataGenerator",
]


class DataGeneratorBuffer(StoppableIteratingBuffer):
    def __init__(self, data_generator, q_out):
        self.data_generator = iter(data_generator)
        super().__init__(q_out=q_out)

    def get_data(self):
        return next(self.data_generator)

    def run(self, packages):
        self.put(packages)


class DummyDataGenerator(DataGeneratorBuffer):
    def __init__(
        self, chanslist: typing.Union[typing.List[str], int], q_out: mp.Queue
    ):
        num_channels = _get_num_channels(chanslist)

        def data_generator():
            while True:
                yield np.random.randn(num_channels).astype(np.float32)

        super().__init__(data_generator(), q_out)


class GwpyTimeSeriesDataGenerator(DataGeneratorBuffer):
    def __init__(
        self,
        chanslist: typing.List[str],
        t0: float,
        duration: float,
        fs: float,
        q_out: mp.Queue,
    ):
        TimeSeriesDict = _import_gwpy()
        data = TimeSeriesDict.get(
            chanslist,
            t0,
            t0 + duration,
            nproc=4,
            allow_tape=True,
            verbose="DOWNLOAD",
        )
        data.resample(fs)
        data = np.stack([data[channel].value for channel in chanslist])

        def data_generator():
            for idx in cycle(range(int(fs * duration))):
                yield data[:, idx]

        super().__init__(data_generator(), q_out=q_out)


class LowLatencyFrameDataGenerator(DataGeneratorBuffer):
    def __init__(
        self,
        data_dir: str,
        chanslist: typing.List[str],
        fs: float,
        q_out: mp.Queue,
        t0: typing.Optional[int] = None,
        file_pattern: typing.Optional[str] = None,
    ):
        TimeSeriesDict = _import_gwpy()

        if file_pattern is None and t0 is None:
            raise ValueError(
                "Must specify either a file pattern or initial timestamp"
            )
        elif file_pattern is None:
            file_pattern = re.compile(fr".*-{t0}-.*\.gwf")
            files = list(filter(file_pattern.full_match, os.listdir(data_dir)))
            if len(files) == 0:
                raise ValueError(
                    "Couldn't find any files matching timestamp {} "
                    "in directory {}".format(t0, data_dir)
                )
            elif len(files) > 1:
                raise ValueError(
                    "Found more than 1 file matching timestamp {} "
                    "in directory {}: {}".format(t0, data_dir, ", ".join(files))
                )
            file_pattern = files[0].replace(str(t0), "{}")
        elif t0 is None:
            prefix, postfix = file_pattern.split("{}")
            regex = re.compile(
                "(?<={})[0-9]{}(?={})".format(prefix, "{10}", postfix)
            )
            timestamps = map(regex.search, os.listdir(data_dir))
            if not any(timestamps):
                raise ValueError(
                    "Couldn't find any timestamps matching the "
                    "pattern {}".format(file_pattern)
                )
            timestamps = [int(t.group(0)) for t in timestamps if t is not None]
            t0 = max(timestamps)

        def data_generator(t0):
            while True:
                start_time = time.time()
                while time.time() - start_time < 1:
                    try:
                        path = os.path.join(data_dir, file_pattern.format(t0))
                        data = TimeSeriesDict.read(path, chanslist)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    raise FileNotFoundError(
                        "Couldn't find next timestep file {}".format(path)
                    )
                data.resample(fs)
                data = np.stack([data[channel].value for channel in chanslist])

                # TODO: how does rounding work for non-integer fs?
                # TODO: pass chunks, accommodate on preproc end
                for idx in range(int(fs)):
                    yield data[:, idx]
                t0 += 1

        super().__init__(data_generator(t0 + 0), q_out=q_out)


def _get_num_channels(chanslist):
    try:
        return len(chanslist)
    except TypeError:
        if not isinstance(chanslist, int):
            raise TypeError("chanslist was type {}".format(type(chanslist)))
        return chanslist


def _import_gwpy():
    try:
        from gwpy.timeseries import TimeSeriesDict
    except ModuleNotFoundError as e:
        msg = "Must install gwpy, try conda install -c conda-forge gwpy"
        raise ModuleNotFoundError(msg) from e
    return TimeSeriesDict
