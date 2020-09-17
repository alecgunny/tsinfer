from functools import partial
import queue
import random
import string
import time

import tritongrpcclient as triton

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer


class AsyncInferenceClient(StoppableIteratingBuffer):
    __name__ = "InferenceClient"

    def __init__(self, url, model_name, model_version, **kwargs):
        # set up server connection and check that server is active
        client = triton.InferenceServerClient(url)
        if not client.is_server_live():
            raise RuntimeError("Server not live")

        # verify that model is ready
        if not client.is_model_ready(model_name):
            # if not, try to load use model control API
            try:
                client.load_model(model_name)

            # if we can't load the model, first check if the given
            # name is even valid. If it is, throw our hands up
            except triton.InferenceServerException:
                models = client.get_model_repository_index().models
                model_names = [model.name for model in models]
                if model_name not in model_names:
                    raise ValueError(
                        "Model name {} not one of available models: {}".format(
                            model_name, ", ".join(model_names))
                    )
                else:
                    raise RuntimeError(
                        "Couldn't load model {} for unknown reason".format(
                            model_name)
                    )
            # double check that load worked
            assert client.is_model_ready(model_name)

        model_metadata = client.get_model_metadata(model_name)
        # TODO: find better way to check version, or even to
        # load specific version
        # assert model_metadata.versions[0] == model_version

        model_input = model_metadata.inputs[0]
        data_type = model_input.datatype
        model_output = model_metadata.outputs[0]

        self.client_input = triton.InferInput(
            model_input.name, tuple(model_input.shape), data_type
        )
        self.client_output = triton.InferRequestedOutput(model_output.name)
        self.client = client

        self.model_name = model_name
        self.model_version = str(model_version)

        self._in_flight_requests = {}
        super().__init__(**kwargs)

    @streaming_func_timer
    def update_latencies(self):
        model_stats = self.client.get_inference_statistics().model_stats
        for model_stat in model_stats:
            if (
                    model_stat.name == self.model_name and
                    model_stat.version == self.model_version
            ):
                inference_stats = model_stat.inference_stats
                break
        else:
            raise ValueError
        count = inference_stats.success.count
        if count == 0:
            return

        steps = ["queue", "compute_input", "compute_infer", "compute_output"]
        for step in steps:
            avg_time = getattr(inference_stats, step).ns / (10**9 * count)
            self.latency_q.put((step, avg_time))

    def loop(self):
        while True:
            try:
                X, y, batch_start_time = self.get(timeout=1e-6)
                break
            except queue.Empty:
                if self.paused:
                    return

        callback=partial(
            self.process_result, target=y, batch_start_time=batch_start_time
        )
 
        # TODO: is there a way to uniquely identify inference
        # requests such that we can keep track of round trip latency?
        if self.profile:
            start_time = time.time()
            request_id = ''.join(random.choices(string.ascii_letters, k=16))
            self._in_flight_requests[request_id] = start_time
        else:
            request_id = None

        self.client_input.set_data_from_numpy(X.astype("float32"))
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[self.client_input],
            outputs=[self.client_output],
            request_id=request_id,
            callback=callback
        )

        if self.profile:
            self.update_latencies()
    
    def process_result(self, target, batch_start_time, result, error):
        # TODO: add error checking
        prediction = result.as_numpy(self.client_output.name())
        self.put((prediction, target, batch_start_time))
        if self.profile:
            end_time = time.time()
            start_time = self._in_flight_requests.pop(result.get_response().id)
            self.latency_q.put(("total", end_time - start_time))

