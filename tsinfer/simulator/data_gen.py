from itertools import cycle

from gwpy.timeseries import TimeSeriesDict
import numpy as np

from tsinfer.pipeline.common import StoppableIteratingBuffer


__all__ = ["GwpyTimeSeriesDataGenerator"]


class DataGeneratorBuffer(StoppableIteratingBuffer):
    def __init__(self, data_generator, **kwargs):
        self.data_generator = iter(data_generator)
        super().__init__(**kwargs)

    def loop(self):
        samples, target = next(self.data_generator)
        self.put((samples, target))


class GwpyTimeSeriesDataGenerator(DataGeneratorBuffer):
    def __init__(
            self,
            chanslist,
            t0,
            duration,
            fs,
            target_channel=0,
            **kwargs
    ):
        if isinstance(target_channel, str):
            if target_channel not in chanslist:
                chanslist = chanslist + [target_channel]
        elif isinstance(target_channel, int):
            target_channel = chanslist[target_channel]
        else:
            raise ValueError

        data = TimeSeriesDict.get(
            chanslist,
            t0,
            t0+duration,
            nproc=4,
            allow_tape=True,
            verbose="DOWNLOAD"
        )
        data.resample(fs)

        target = data.pop(target_channel).value
        channels = list(data.keys())
        data = np.stack([data[channel].value for channel in channels])

        def data_generator():
            for idx in cycle(range(int(fs*duration))):
                samples = {channel: x for channel,x in zip(channels, data[:, idx])}
                yield samples, target[idx]
        super().__init__(data_generator(), **kwargs)
