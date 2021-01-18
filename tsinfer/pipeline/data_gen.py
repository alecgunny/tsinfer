import multiprocessing as mp
import os
import re
import time
import typing
from itertools import cycle

import attr
import numpy as np

from tsinfer.pipeline.common import StoppableIteratingBuffer

# if typing.TYPE_CHECKING:
#     import multiprocessing as mp


__all__ = [
    "DataGeneratorBuffer",
    "GwpyTimeSeriesDataGenerator",
    "DummyDataGenerator",
    "LowLatencyFrameDataGenerator",
]


class DataGeneratorBuffer(StoppableIteratingBuffer):
    def __init__(self, gen_fn, idx_range, q_out):
        self.idx = 0
        self.idx_range = idx_range
        self.gen_fn = gen_fn
        super().__init__(q_out=q_out)

    def get_data(self):
        try:
            return self.gen_fn(self.idx)
        except IndexError:
            self.idx = 0
            return self.gen_fn(self.idx)

    def run(self, packages):
        self.put(packages)
        self.idx += 1
        if self.idx == self.idx_range:
            self.idx = 0


@attr.s(auto_attribs=True)
class DummyDataGeneratorFn:
    num_channels: int

    def __call__(self, idx):
        return np.random.randn(self.num_channels).astype(np.float32)


class DummyDataGenerator(DataGeneratorBuffer):
    def __init__(
        self, chanslist: typing.Union[typing.List[str], int], q_out: mp.Queue
    ):
        num_channels = _get_num_channels(chanslist)
        super().__init__(DummyDataGeneratorFn(num_channels), 1, q_out)


@attr.s(auto_attribs=True)
class GwpyTimeSeriesDataGeneratorFn:
    data: np.ndarray

    def __call__(self, idx):
        return self.data[:, idx]


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
        gen_fn = GwpyTimeSeriesDataGeneratorFn(data)
        super().__init__(gen_fn, int(fs * duration), q_out=q_out)


@attr.s(auto_attribs=True)
class LowLatencyFrameDataGeneratorFn:
    path_pattern: str
    t0: int
    fs: float
    chanslist: typing.List[str]
    read_fn: typing.Callable

    def __attrs_post_init__(self):
        self.data = None

    def __call__(self, idx):
        if idx == int(self.fs) or self.data is None:
            start_time = time.time()
            while time.time() - start_time < 1:
                try:
                    path = self.path_pattern.format(self.t0)
                    data = self.read_fn(path, self.chanslist)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise ValueError(f"Couldn't find next timestep file {path}")

            data.resample(self.fs)
            self.data = np.stack(
                [data[channel].value for channel in self.chanslist]
            )
            self.t0 += 1
            raise IndexError
        return self.data[:, idx]


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

        gen_fn = LowLatencyFrameDataGeneratorFn(
            os.path.join(data_dir, file_pattern),
            t0 + 0,
            fs,
            chanslist,
            TimeSeriesDict.read,
        )
        super().__init__(gen_fn, int(fs) + 1, q_out=q_out)


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
