import queue
import time

import numpy as np

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer


class Preprocessor(StoppableIteratingBuffer):
    _LATENCY_WHITELIST = ["update"]
    __name__ = "Preprocessor"

    def __init__(
            self,
            batch_size,
            channels,
            kernel_size,
            kernel_stride,
            fs,
            preproc_fn=None,
            **kwargs
    ):
        self.channels = sorted(channels)
        self.initialize(
            batch_size=batch_size,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            fs=fs
        )
        self.preproc_fn = preproc_fn

        super().__init__(**kwargs)

    def initialize(
            self,
            batch_size,
            kernel_size,
            kernel_stride,
            fs
    ):

        # define sizes for everything
        num_samples_frame = int(kernel_size*fs)
        num_samples_stride = int(kernel_stride*fs)
        num_samples_total = (batch_size-1)*num_samples_stride + num_samples_frame
        num_samples_update = batch_size*num_samples_stride
        batch_overlap = num_samples_total - num_samples_update

        # initialize arrays up front
        dtype = np.float32
        self._data = np.empty(
            (len(self.channels), num_samples_total), dtype=dtype
        )
        self._extension = np.empty(
            (len(self.channels), num_samples_update), dtype=dtype
        )
        self._batch = np.empty(
            (batch_size, len(self.channels), num_samples_frame), dtype=dtype
        )
        self._target = np.empty((num_samples_total,), dtype=dtype)
    
        # save this since we can get everything we need
        # from this and the first dimension of _data
        self.batch_overlap = batch_overlap

        # tells us how to window a 2D stream of data into a 3D batch
        slices = []
        for i in range(batch_size):
            start = i*num_samples_stride
            stop = start + num_samples_frame
            slices.append(slice(start, stop))
        self.slices = slices

        self.secs_per_sample = 1. / fs
        self._last_sample_time = None
        self._batch_start_time = None

        self.params = {
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "kernel_stride": kernel_stride,
            "fs": fs
        }

    def initialize_loop(self):
        self._last_sample_time = time.time()
        for i in range(self.batch_overlap):
            x, y = self.read_sensor()
            self._data[:, i] = x
            self._target[i] = y

    def maybe_wait(self):
        '''
        function for making sure we're not peeking ahead
        at samples that can't exist yet. Shouldn't
        be necessary for real deployment where that
        obviously isn't an issue
        '''
        while (time.time() - self._last_sample_time) < self.secs_per_sample:
            continue
        self._last_sample_time = time.time()

    def read_sensor(self):
        '''
        read individual samples and return an array of size
        `(len(self.channels), 1)` for hstacking
        '''
        while True:
            try:
                samples, target = self.get(timeout=1e-7)
                break
            except queue.Empty as e:
                if self.paused:
                    raise e

        # make sure that we don't "peek" ahead at
        # data that isn't supposed to exist yet
        self.maybe_wait()

        samples = [samples[channel] for channel in self.channels]
        x = np.array(samples, dtype=np.float32)
        return x, target

    def get_data(self):
        if self._last_sample_time is None:
            self.initialize_loop()

        # start by reading the next batch of samples
        for i in range(self.batch_overlap, self._data.shape[1]):
            x, y = self.read_sensor()
            self._data[:, i] = x
            self._target[i] = y

            # measure the time the batch was created for profiling
            # purposes. Again, not necessary for a production
            # deployment
            if i == self._batch.shape[2]:
                batch_start_time = time.time()
        return self._data, self._target, batch_start_time

    @streaming_func_timer
    def preprocess(self, x):
        '''
        perform any preprocessing transformations on the data
        just does normalization for now
        '''
        # TODO: is there any extra preprocessing that should
        # be done? With small enough strides and batch sizes,
        # does there reach a point at which it makes sense
        # to do preproc on individual samples (assuming it
        # can be done that locally) to avoid doing thousands
        # of times on the same sample? Where is this limit?
        if self.preproc_fn is not None:
            return self.preproc_fn(x)
        return x

    @streaming_func_timer
    def make_batch(self, data):
        '''
        take windows of data at strided intervals and stack them
        '''
        for i, slc in enumerate(self.slices):
            self._batch[i] = data[:, slc]
        # doing a return here in case we decide
        # we need to do a copy, which I think we do
        return self._batch

    @streaming_func_timer
    def reset(self):
        '''
        shift over all the data elements so that we can populate
        the leftovers with the next batch. Also update the
        batch_start_time by a full batch worth of stride times
        '''
        # TODO: does it make sense to do the copy here, since we'll
        # need to be waiting for the next batch of samples to generate
        # anyway?
        # Also, would it be faster to do an append then a slice?
        # I think append allocates another full array so maybe not
        shift = -(self._data.shape[1] - self.batch_overlap)
        self._data = np.append(
            self._data[:, -self.batch_overlap:], self._extension, axis=1
        )
        self._target = np.append(
            self._target[-self.batch_overlap:], self._extension[0], axis=0
        )

    def run(self, x, y, batch_start_time):
        x = self.preprocess(x)
        x = self.make_batch(x)
        self.put((x, y, batch_start_time))
        self.reset()

