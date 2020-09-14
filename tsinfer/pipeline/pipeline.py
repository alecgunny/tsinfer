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
            preproc_fn=None,
            postproc_fn=None,
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
            preproc_fn,
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
            kernel_size,
            kernel_stride,
            fs,
            postproc_fn,
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
                    func, latency = buff.latency_q.get_nowait()
                except queue.Empty:
                    break
                profile_dict[buff][func].update(latency)
        return profile_dict

    def clear_profile_qs(self):
        for buff in self.buffers:
            buff.latency_q.queue.clear()

    def start(self):
        if any([p.is_alive() for p in self.processes]):
            raise RuntimeError("Processes already started")
        for p in self.processes:
            p.start()

    def is_alive(self):
        return any([p.is_alive() for p in self.processes])

    def pause(self):
        for buff in self.buffers:
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

    def update_preprocessor(self, params=None):
        assert self.paused
        if params is not None:
            self.preprocessor._param_q.put(params)
        self.preprocessor._reset_flag.set()
        while self.preprocessor._reset_flag.is_set():
            pass

    def clear_qs(self):
        for buff in self.buffers:
            with buff.q_out.mutex:
                buff.q_out.queue.clear()
