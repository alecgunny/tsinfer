import queue
import time
import typing
from functools import partial

import numpy as np

from tsinfer.pipeline.common import Package, Path, StoppableIteratingBuffer


class Preprocessor(StoppableIteratingBuffer):
    __name__ = "Preprocessor"

    def __init__(
        self,
        paths: typing.Union[Path, typing.Dict[str, Path]],
        batch_size: int,
        kernel_size: float,
        kernel_stride: float,
        fs: float,
        q_out: queue.Queue,
    ):
        self.paths = paths
        if isinstance(paths, dict):
            q_in = {name: path.q for name, path in paths.items()}
            init_kwargs = {
                name: path.processing_fn_kwargs for name, path in paths.items()
            }
        elif isinstance(paths, Path):
            q_in = paths.q
            init_kwargs = paths.processing_fn_kwargs or {}
        else:
            raise TypeError

        self.initialize(
            batch_size=batch_size,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            fs=fs,
            **init_kwargs
        )
        super().__init__(q_in=q_in, q_out=q_out)

    def initialize(self, batch_size, kernel_size, kernel_stride, fs, **kwargs):
        # define sizes for everything
        num_samples_frame = int(kernel_size * fs)
        num_samples_stride = int(kernel_stride * fs)
        num_samples_total = (
            batch_size - 1
        ) * num_samples_stride + num_samples_frame
        num_samples_update = batch_size * num_samples_stride
        batch_overlap = num_samples_total - num_samples_update

        # initialize arrays up front
        dtype = np.float32  # TODO: make path attr?

        if isinstance(self.paths, Path):
            paths = {None: self.paths}
        else:
            paths = self.paths

        self._data = {}
        self._batch = {}
        self._extensions = {}
        self._preprocessing_fns = {}
        for name, path in paths.items():
            try:
                num_channels = len(path.channels)
            except TypeError:
                num_channels = path.channels

            # _data holds the single time series that gets
            # windowed into our batch
            self._data[name] = np.empty(
                (num_channels, num_samples_total), dtype=dtype
            )
            # _batch holds the windowed version of _data
            self._batch[name] = np.empty(
                (batch_size, num_channels, num_samples_frame), dtype=dtype
            )

            # _extension holds the empty array used to extend _data at
            # each iteration
            self._extensions[name] = np.empty(
                (num_channels, num_samples_update), dtype=dtype
            )

            if path.processing_fn is not None:
                if name is None:
                    fn_kwargs = kwargs
                else:
                    fn_kwargs = kwargs[name] or {}
                self._preprocessing_fns[name] = partial(
                    path.processing_fn, **fn_kwargs
                )

        # save this since we can get everything we need
        # from this and the first dimension of _data
        self._num_samples_total = num_samples_total
        self._num_samples_frame = num_samples_frame
        self._batch_overlap = batch_overlap

        # tells us how to window a 2D stream of data into a 3D batch
        slices = []
        for i in range(batch_size):
            start = i * num_samples_stride
            stop = start + num_samples_frame
            slices.append(slice(start, stop))
        self.slices = slices

        self.secs_per_sample = 1.0 / fs
        self._last_sample_time = None

        self.params = {
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "kernel_stride": kernel_stride,
            "fs": fs,
        }
        self.params.update(kwargs)

    def read_data_sources(self):
        while True:
            try:
                return super().get_data()
                break
            except queue.Empty as e:
                if self.paused or self.stopped:
                    raise e

    def initialize_loop(self):
        for i in range(self._batch_overlap):
            packages = self.read_data_sources()
            for name, x in packages.items():
                self._data[name][:, i] = x

    def maybe_wait(self):
        """
        function for making sure we're not peeking ahead
        at samples that can't exist yet. Shouldn't
        be necessary for real deployment where that
        obviously isn't an issue
        """
        if self._last_sample_time is not None:
            curr_time = time.time()
            while (curr_time - self._last_sample_time) < self.secs_per_sample:
                curr_time = time.time()
        else:
            curr_time = time.time()
        self._last_sample_time = curr_time

    def get_data(self):
        if self._last_sample_time is None:
            self.initialize_loop()

        # start by reading the next batch of samples
        # TODO: add named channel support
        for i in range(self._batch_overlap, self._num_samples_total):
            packages = self.read_data_sources()
            # self.maybe_wait()

            for name, x in packages.items():
                self._data[name][:, i] = x

            # measure the time the batch was created for profiling
            # purposes. Again, not necessary for a production
            # deployment
            if i == self._num_samples_frame:
                batch_start_time = time.time()

        packages = {
            name: Package(x, batch_start_time) for name, x in self._data.items()
        }
        return packages

    @StoppableIteratingBuffer.profile
    def preprocess(self, x, name=None):
        """
        perform any preprocessing transformations on the data
        """
        # TODO: With small enough strides and batch sizes,
        # does there reach a point at which it makes sense
        # to do preproc on individual samples (assuming it
        # can be done that locally) to avoid doing thousands
        # of times on the same sample? Where is this limit?
        # How to let user decide if preproc can be done locally?
        preproc_fn = self._preprocessing_fns.get(name)
        if preproc_fn is not None:
            return self.preproc_fn(x)
        return x

    @StoppableIteratingBuffer.profile
    def make_batch(self, data, name=None):
        """
        take windows of data at strided intervals and stack them
        """
        for i, slc in enumerate(self.slices):
            self._batch[name][i] = data[:, slc]
        # doing a return here in case we decide
        # we need to do a copy, which I think we do
        return self._batch[name]

    @StoppableIteratingBuffer.profile
    def reset(self, name=None):
        """
        remove stale data elements and replace with empty
        ones to be filled out by data generator
        """
        self._data[name] = np.append(
            self._data[name][:, -self._batch_overlap :],
            self._extensions[name],
            axis=1,
        )

    def run(self, packages):
        # TODO: thread this?
        put_obj = {}
        for name, package in packages.items():
            x = self.preprocess(package.x, name=name)
            x = self.make_batch(package.x, name=name)
            put_obj[name] = (x, package.batch_start_time)

        if set(put_obj) == set([None]):
            put_obj = put_obj.pop(None)
        self.put(put_obj)

        for name in packages:
            self.reset(name)
