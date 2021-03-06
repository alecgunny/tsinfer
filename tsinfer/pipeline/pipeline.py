import multiprocessing as mp
import queue

from tsinfer.pipeline.preprocessor import Preprocessor
from tsinfer.pipeline.client import AsyncInferenceClient
from tsinfer.pipeline.postprocessor import Postprocessor


class Pipeline:
    def __init__(
            self,
            input_data_q,
            batch_size,
            channels,
            kernel_size,
            kernel_stride,
            fs,
            url,
            model_name,
            model_version,
            preprocessing_fn=None,
            preprocessing_fn_kwargs=None,
            postprocessing_fn=None,
            postprocessing_fn_kwargs=None,
            qsize=100,
            profile=False
    ):
        preprocess_q = mp.Queue(qsize)
        inference_q = mp.Queue(qsize)
        postprocess_q = mp.Queue(qsize)

        self.preprocessor = Preprocessor(
            batch_size,
            channels,
            kernel_size,
            kernel_stride,
            fs,
            preprocessing_fn,
            preprocessing_fn_kwargs=preprocessing_fn_kwargs,
            q_in=input_data_q,
            q_out=preprocess_q,
            profile=profile
        )

        self.client = AsyncInferenceClient(
            url,
            model_name,
            model_version,
            q_in=preprocess_q,
            q_out=inference_q,
            profile=profile
        )

        self.postprocessor = Postprocessor(
            batch_size,
            kernel_size,
            kernel_stride,
            fs,
            postprocessing_fn,
            postprocessing_fn_kwargs,
            q_in=inference_q,
            q_out=postprocess_q,
            profile=profile
        )

        self.processes = []
        for buff in self.buffers:
            self.processes.append(mp.Process(target=buff))

    @property
    def buffers(self):
        return [self.preprocessor, self.client, self.postprocessor]

    def get(self):
        return self.postprocessor.q_out.get()

    def put(self, x, timeout=None):
        self.preprocessor.q_in.put(x, timeout=timeout)

    def get_profiles(self, profile_dict, max_gets=20):
        for buff in self.buffers:
            for i in range(max_gets):
                try:
                    func, latency = buff.profile_q.get_nowait()
                except queue.Empty:
                    break
                profile_dict[buff][func].update(latency)
        return profile_dict

    def start(self):
        if self.is_alive:
            raise RuntimeError("Processes already started")
        for p in self.processes:
            p.start()

    @property
    def is_alive(self):
        return any([p.is_alive() for p in self.processes])

    def pause(self):
        for buff in self.buffers[::-1]:
            buff.pause()

    @property
    def paused(self):
        return all([buff.paused for buff in self.buffers])

    def resume(self):
        for buff in self.buffers:
            buff.resume()

    def stop(self):
        for buff in self.buffers:
            buff.stop()

        for p in self.processes:
            if p.is_alive():
                p.join()

        self.clear_qs()

    def update(self, params):
        assert self.paused
        for buff in self.buffers:
            if set(params.keys()) & set(buff.params.keys()):
                buff.param_q.put(params)
                buff.param_q.join()

    def _clear_a_q(self, q):
        # solution provided by https://stackoverflow.com/a/36018632
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def clear_profile_qs(self):
        for buff in self.buffers:
            self._clear_a_q(buff.profile_q)

    def clear_qs(self):
        for buff in self.buffers:
            self._clear_a_q(buff.q_out)
