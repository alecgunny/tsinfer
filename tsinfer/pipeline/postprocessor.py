from functools import partial
import time

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer



class Postprocessor(StoppableIteratingBuffer):
    __name__ = "Postprocessor"

    def __init__(
            self,
            batch_size,
            kernel_size,
            kernel_stride,
            fs,
            postprocessing_fn=None,
            postprocessing_fn_kwargs=None,
            **kwargs
    ):
        # TODO: should windowing and aggregating really
        # be a part of post-processing or built in
        # functionality?
        self._postprocessing_fn = postprocessing_fn
        postprocessing_fn_kwargs = postprocessing_fn_kwargs or {}
        self.initialize(**postprocessing_fn_kwargs)
        super().__init__(**kwargs)

    def initialize(
            self,
            batch_size,
            kernel_size,
            kernel_stride,
            fs,
            **kwargs
        ):
        num_samples_frame = int(kernel_size*fs)
        num_samples_stride = int(kernel_stride*fs)
        num_samples_total = (batch_size-1)*num_samples_stride + num_samples_frame

        # initialize an empty tensor to store aggregated predictions
        self._prediction = np.empty(
            ((batch_size-1)*num_samples_stride+num_samples_frame,) dtype=np.float32
        )

        # initialize weights to multiply prediction by for
        # batch level alignment and averaging
        # e.g. if
        # num_samples_stride = 1
        # num_samples_frame = 9
        # batch_size = 4
        # then the inverse of the weight array should be
        # [ 1, 2, 3, 4, 4, 4, 4, 4]
        # [ 2, 3, 4, 4, 4, 4, 4, 2]
        # [ 3, 4, 4, 4, 4, 4, 3, 2]
        # [ 4, 4, 4, 4, 4, 3, 2, 1]
        self._aggregation_weights = np.ones(
            (batch_size, num_samples_frame), dtype=np.float32
        ) / batch_size
        for i in range(batch_size):
            idx = list(range(batch_size-1-i)) + list(range(-i, 0))
            weights = list(range(i+1, batch_size)) + list(range(batch_size-1, batch_size-i-1, -1))
            for j, w in zip(idx, weights):
                start = j*num_samples_stride
                end = (j+1)*num_samples_stride or None
                self._aggregation_weights[start:end] = 1/w

        if self._postprocessing_fn is not None:
            self.postprocessing_fn = partial(self._postprocessing_fn, **kwargs)

        self.params = {
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "kernel_stride": kernel_stride,
            "fs": fs
        }
        self.params.update(kwargs)

    @streaming_func_timer
    def postprocess(self, prediction):
        if self._postprocessing_fn is not None:
            return self.postprocessing_fn(prediction)
        return prediction

    @streaming_func_time
    def aggregate(self, x):
        prediction = self._prediction*0
        weighted = self._aggregation_weights*x
        for i in range(self.params["batch_size"]):
            start = i*int(self.params["fs"]*self.params["kernel_stride"])
            end = start + int(self.params["fs"] + self.params["kernel_size"])
            prediction[start:end] += weighted[i]
        return prediction

    def run(self, x, y, batch_start_time):
        prediction = self.aggregate(x)
        prediction = self.postprocess(prediction)

        # measure completion time for throughput measurement
        # here to be as accurate as possible
        batch_end_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, y, batch_start_time, batch_end_time))
