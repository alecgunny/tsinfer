import random
import string
import time
from functools import partial

from tritonclient import grpc as triton

from tsinfer.pipeline.common import Package, StoppableIteratingBuffer


class AsyncInferenceClient(StoppableIteratingBuffer):
    __name__ = "InferenceClient"

    def __init__(self, url, model_name, model_version, **kwargs):
        # set up server connection and check that server is active
        client = triton.InferenceServerClient(url)
        if not client.is_server_live():
            raise RuntimeError("Server not live")

        self.client = client
        self.params = {
            "model_name": model_name,
            "model_version": str(model_version),
        }
        self.initialize(model_name, model_version)

        self._in_flight_requests = {}
        super().__init__(**kwargs)

    def initialize(self, model_name, model_version):
        # first unload existing model
        if model_name != self.params["model_name"]:
            self.client.unload_model(model_name)

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

        self.params = {
            "model_name": model_name,
            "model_version": str(model_version),
        }

    @StoppableIteratingBuffer.profile
    def pull_stats(self):
        return self.client.get_inference_statistics().model_stats

    @StoppableIteratingBuffer.profile
    def update_profiles(self, model_stats):
        for model_stat in model_stats:
            if (
                model_stat.name == self.params["model_name"]
                and model_stat.version == self.params["model_version"]
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
            avg_time = getattr(inference_stats, step).ns / (10 ** 9 * count)
            self.profile_q.put((step, avg_time))

    def run(self, package):
        # TODO: this is a hack around a bug, figure this out
        package = package[None]
        callback = partial(
            self.process_result, batch_start_time=package.batch_start_time
        )

        request_id = "".join(random.choices(string.ascii_letters, k=16))
        if self.profile:
            start_time = time.time()
            self._in_flight_requests[request_id] = start_time

        if len(package.x) != len(self.inputs):
            raise ValueError(
                "Received {} inputs but expected {}".format(
                    len(package.x), len(self.inputs)
                )
            )
        if len(package.x) == 1 and None in package.x:
            package.x[list(self.inputs.keys())[0]] = package.x.pop(None)
        if set(package.x) != set(self.inputs):
            raise ValueError(
                "Expected inputs {}, received inputs {}".format(
                    ", ".join(set(package.x)), ", ".join(set(self.inputs))
                )
            )

        for name, x in package.x.items():
            # TODO: better dynamic casting
            self.inputs[name].set_data_from_numpy(x.astype("float32"))

        self.client.async_infer(
            model_name=self.params["model_name"],
            model_version=self.params["model_version"],
            inputs=list(self.inputs.values()),
            outputs=self.outputs,
            request_id=request_id,
            callback=callback,
        )

        if self.profile:
            stats = self.pull_stats()
            self.update_profiles(stats)

    def process_result(self, batch_start_time, result, error):
        # TODO: add error checking
        x = {}
        for output in self.outputs:
            name = output.name()
            x[name] = result.as_numpy(name)
        package = Package(x, batch_start_time)
        self.put(package)

        if self.profile:
            end_time = time.time()
            start_time = self._in_flight_requests.pop(result.get_response().id)
            self.profile_q.put(("total", end_time - start_time))
