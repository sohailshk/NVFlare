# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
from safetensors.torch import load as load_tensors

from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fuel.f3.streaming.download_service import ProduceRC
from nvflare.fuel.utils import fobs


@pytest.fixture(autouse=True)
def _tensor_decomposer():
    fobs.reset()
    fobs.register(TensorDecomposer)
    yield
    fobs.reset()


def _send_to_new_process(obj):
    data = fobs.dumps(obj)
    fobs.reset()  # the receiver only has the decomposers registered at startup
    fobs.register(TensorDecomposer)
    return fobs.loads(data)


class TestParameterSerialization:
    def test_parameter_values_survive_round_trip(self):
        weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))

        result = fobs.loads(fobs.dumps({"weight": weight}))["weight"]

        assert torch.equal(result.detach(), weight.detach())

    def test_named_parameters_received_by_another_process(self):
        model = torch.nn.Linear(3, 2)
        params = dict(model.named_parameters())

        result = _send_to_new_process(params)

        assert set(result) == {"weight", "bias"}
        for name, param in params.items():
            assert torch.equal(result[name].detach(), param.detach())

    def test_streamed_downloadable_carries_parameter_values(self):
        weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        downloadable = TensorDownloadable({"w": weight}, max_chunk_size=1024)

        rc, items, _ = downloadable.produce({}, "receiver")

        assert rc == ProduceRC.OK
        assert torch.equal(load_tensors(items[0])["w"], weight.detach())
        downloadable.release()
