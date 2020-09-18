from functools import partial
import time

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer



class Postprocessor(StoppableIteratingBuffer):
    __name__ = "Postprocessor"

    def __init__(
            self,
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

    def initialize(self, **kwargs):
        if self._postprocessing_fn is not None:
            self.postprocessing_fn = partial(self._postprocessing_fn, **kwargs)
        self.params = kwargs

    @streaming_func_timer
    def postprocess(self, prediction):
        if self._postprocessing_fn is not None:
            return self.postprocessing_fn(prediction)
        return prediction

    def run(self, x, y, batch_start_time):
        prediction = self.postprocess(x)

        # measure completion time for throughput measurement
        # here to be as accurate as possible
        batch_end_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, y, batch_start_time, batch_end_time))
