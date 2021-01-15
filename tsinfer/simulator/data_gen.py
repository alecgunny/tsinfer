import os
import re
import time
from itertools import cycle

import numpy as np

from tsinfer.pipeline.common import StoppableIteratingBuffer

__all__ = [
    "DataGeneratorBuffer",
    "GwpyTimeSeriesDataGenerator",
    "DummyDataGenerator",
    "LowLatencyFrameDataGenerator",
]


class DataGeneratorBuffer(StoppableIteratingBuffer):
    def __init__(self, data_generator, **kwargs):
        self.data_generator = iter(data_generator)
        super().__init__(**kwargs)

    def get_data(self):
        samples, target = next(self.data_generator)
        return samples, target, None

    def run(self, x, y, batch_start_time=None):
        self.put((x, y))


class DummyDataGenerator(DataGeneratorBuffer):
    def __init__(self, chanslist, **kwargs):
        def data_generator():
            while True:
                data = np.random.randn(len(chanslist)).astype(np.float32)
                samples = {channel: x for channel, x in zip(chanslist, data)}
                target = np.float32(np.random.randn())
                yield samples, target

        super().__init__(data_generator(), **kwargs)


class GwpyTimeSeriesDataGenerator(DataGeneratorBuffer):
    def __init__(self, chanslist, t0, duration, fs, target_channel=0, **kwargs):
        target_channel, chanslist = _validate_target_channel(
            target_channel, chanslist
        )
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

        target = data.pop(target_channel).value
        channels = list(data.keys())
        data = np.stack([data[channel].value for channel in channels])

        def data_generator():
            for idx in cycle(range(int(fs * duration))):
                samples = {
                    channel: x for channel, x in zip(channels, data[:, idx])
                }
                yield samples, target[idx]

        super().__init__(data_generator(), **kwargs)


class LowLatencyFrameDataGenerator(DataGeneratorBuffer):
    def __init__(
        self,
        data_dir,
        chanslist,
        fs,
        t0=None,
        file_pattern=None,
        target_channel=0,
        **kwargs,
    ):
        target_channel, chanslist = _validate_target_channel(
            target_channel, chanslist
        )
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
        self.t0 = t0

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

                target = data.pop(target_channel).value
                channels = list(data.keys())
                data = np.stack([data[channel].value for channel in channels])

                # TODO: how does rounding work for non-integer fs
                # TODO: pass chunks, accommodate on preproc end
                for idx in range(int(fs)):
                    samples = {
                        channel: x for channel, x in zip(channels, data[:, idx])
                    }
                    yield samples, target[idx]
                t0 += 1

        super().__init__(data_generator(t0 + 0), **kwargs)


def _validate_target_channel(target_channel, chanslist):
    if isinstance(target_channel, str):
        if target_channel not in chanslist:
            chanslist = chanslist + [target_channel]
    elif isinstance(target_channel, int):
        target_channel = chanslist[target_channel]
    else:
        raise ValueError
    return target_channel, chanslist


def _import_gwpy():
    try:
        from gwpy.timeseries import TimeSeriesDict
    except ModuleNotFoundError as e:
        msg = "Must install gwpy, try conda install -c conda-forge gwpy"
        raise ModuleNotFoundError(msg) from e
    return TimeSeriesDict
