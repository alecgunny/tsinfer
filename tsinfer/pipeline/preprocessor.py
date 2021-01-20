import queue
import time
import typing
from functools import partial
from multiprocessing import Queue

import numpy as np
from tritonclient import grpc as triton

from tsinfer.pipeline.common import Package, StoppableIteratingBuffer


class Preprocessor(StoppableIteratingBuffer):
    __name__ = "Preprocessor"

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int,
        batch_size: int,
        kernel_size: float,
        kernel_stride: float,
        fs: float,
        postprocessor: typing.Callable,
        q_out: Queue,
        qsize: int = 100,
        profile: bool = False,
    ):
        # set up server connection and check that server is active
        client = triton.InferenceServerClient(url)
        if not client.is_server_live():
            raise RuntimeError("Server not live")
        self.client = client

        self.initialize(
            model_name,
            model_version,
            batch_size,  # TODO: we can get this from model
            kernel_size,
            kernel_stride,
            fs,  # TODO: we can get this from model
        )
        q_in = {name: Queue(qsize) for name in self.inputs}
        self.postprocessor = postprocessor
        super().__init__(q_in=q_in, q_out=q_out, profile=profile)

    def initialize(
        self,
        model_name,
        model_version,
        batch_size,
        kernel_size,
        kernel_stride,
        fs,
    ):
        # first unload existing model
        # if model_name != self.params["model_name"]:
        #     self.client.unload_model(model_name)

        # verify that model is ready
        if not self.client.is_model_ready(model_name):
            # if not, try to load use model control API
            try:
                self.client.load_model(model_name)

            # if we can't load the model, first check if the given
            # name is even valid. If it is, throw our hands up
            except triton.InferenceServerException:
                models = self.client.get_model_repository_index().models
                model_names = [model.name for model in models]
                if model_name not in model_names:
                    raise ValueError(
                        "Model name {} not one of available models: {}".format(
                            model_name, ", ".join(model_names)
                        )
                    )
                else:
                    raise RuntimeError(
                        "Couldn't load model {} for unknown reason".format(
                            model_name
                        )
                    )
            # double check that load worked
            assert self.client.is_model_ready(model_name)

        model_metadata = self.client.get_model_metadata(model_name)
        # TODO: find better way to check version, or even to
        # load specific version
        # assert model_metadata.versions[0] == model_version

        self.inputs = {}
        for input in model_metadata.inputs:
            self.inputs[input.name] = triton.InferInput(
                input.name, tuple(input.shape), input.datatype
            )
        self.outputs = [
            triton.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]

        self._initialize_tensors(batch_size, kernel_size, kernel_stride, fs)
        self.params = {
            "model_name": model_name,
            "model_version": model_version,
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "kernel_stride": kernel_stride,
            "fs": fs
        }

    def _initialize_tensors(
        self, batch_size, kernel_size, kernel_stride, fs, **kwargs
    ):
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

        self._data = {}
        self._batch = {}
        self._preprocessing_fns = {}
        for name, input in self.inputs.items():
            if len(input.shape()) == 2:
                num_channels = 1
            else:
                num_channels = input.shape()[1]

            # _data holds the single time series that gets
            # windowed into our batch
            self._data[name] = np.empty(
                (num_channels, num_samples_total), dtype=dtype
            )
            # _batch holds the windowed version of _data
            self._batch[name] = np.empty(
                (batch_size, num_channels, num_samples_frame), dtype=dtype
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
        self._start_time = time.time()
        self._inferences = 0

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

        return Package(self._data, batch_start_time)

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
        self.inputs[name].set_data_from_numpy(
            self._batch[name].astype("float32")
        )

    @StoppableIteratingBuffer.profile
    def reset(self):
        """
        remove stale data elements and replace with empty
        ones to be filled out by data generator
        """
        for name in self._data:
            self._data[name][:-self._batch_overlap] = self._data[name][
                self._batch_overlap:
            ]

    def run(self, package):
        for name, x in package.x.items():
            x = self.preprocess(x, name=name)
            self.make_batch(x, name=name)

        def callback(result, error):
            end_time = time.time()
            latency = end_time - package.batch_start_time
            throughput = (self._inferences+1) / (end_time - self._start_time)
            self._inferences += 1

            self.put((latency, throughput))

        self.client.async_infer(
            model_name=self.params["model_name"],
            model_version=str(self.params["model_version"]),
            inputs=list(self.inputs.values()),
            outputs=self.outputs,
            # request_id=request_id,
            callback=callback
        )
        self.reset()
