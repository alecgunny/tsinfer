import multiprocessing as mp
import queue
import typing

from tsinfer.pipeline.client import AsyncInferenceClient
from tsinfer.pipeline.common import Path
from tsinfer.pipeline.postprocessor import Postprocessor
from tsinfer.pipeline.preprocessor import Preprocessor


def _dict_of(_type):
    return typing.Union[_type, typing.Dict[str, _type]]


class Pipeline:
    def __init__(
        self,
        batch_size: int,
        kernel_size: float,
        kernel_stride: float,
        fs: int,
        server_url: str,
        model_name: str,
        model_version: int,
        preprocessing_fn: typing.Optional[_dict_of(typing.Callable)] = None,
        preprocessing_fn_kwargs: typing.Optional[_dict_of(dict)] = None,
        postprocessing_fn: typing.Optional[_dict_of(typing.Callable)] = None,
        postprocessing_fn_kwargs: typing.Optional[_dict_of(dict)] = None,
        qsize: int = 100,
        profile: bool = False,
    ):
        preprocess_q = mp.Queue(qsize)
        inference_q = mp.Queue(qsize)
        postprocess_q = mp.Queue(qsize)

        self.client = AsyncInferenceClient(
            server_url,
            model_name,
            model_version,
            q_in=preprocess_q,
            q_out=inference_q,
            profile=profile,
        )

        model_metadata = self.client.get_model_metadata(model_name)
        paths = {}
        for input in model_metadata.input:
            channels = 1 if len(input.dims) < 3 else input.dims[1]
            q_in = mp.Queue(qsize)
            if isinstance(preprocessing_fn, dict):
                try:
                    preproc_fn = preprocessing_fn[input.name]
                except KeyError:
                    raise ValueError(
                        "No input for preprocessing fn {}".format(input.name)
                    )
                preproc_fn_kwargs = preprocessing_fn_kwargs.get(input.name, None)
            else:
                preproc_fn = preprocessing_fn
                preproc_fn_kwargs = preprocessing_fn_kwargs

            paths[input.name] = Path(
                channels=channels,
                q=q_in,
                processing_fn=preproc_fn,
                processing_fn_kwargs=preproc_fn_kwargs,
            )

        self.preprocessor = Preprocessor(
            paths=paths,
            batch_size=batch_size,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            fs=fs,
            q_out=preprocess_q,
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
            profile=profile,
        )

        self.processes = []
        for buff in self.buffers:
            self.processes.append(mp.Process(target=buff))

    @property
    def buffers(self):
        return [self.preprocessor, self.client, self.postprocessor]

    @property
    def input_data_qs(self):
        return [path.q for path in self.preprocessor.paths]

    def get(self):
        x = self.postprocessor.q_out.get()
        if isinstance(x, Exception):
            for buff in self.buffers:
                if not buff.stopped:
                    buff.stop()
            for p in self.processes:
                if p.is_alive():
                    p.join()
            raise x
        return x

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
