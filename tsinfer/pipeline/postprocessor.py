import time

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer



class Postprocessor(StoppableIteratingBuffer):
    __name__ = "Postprocessor"

    def __init__(self, postprocess_fn, **kwargs):
        self.postprocess_fn = postprocess_fn
        super().__init__(**kwargs)

    @streaming_func_timer
    def postprocess(self, prediction):
        if self.postprocess_fn is not None:
            return self.postprocess_fn(prediction)
        return prediction

    def loop(self):
        prediction, target, batch_start_time = self.get()
        prediction = self.postprocess(prediction)

        # measure completion time for throughput measurement
        # here to be as accurate as possible
        batch_end_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, target, batch_start_time, batch_end_time))
