"""Reject the observed V100 allocation and exercise CUDA-probe failures."""

from types import SimpleNamespace
import pytest
from tri_fair.cli.doctor import inspect_cuda_devices


class FakeTensor:
    def __add__(self, value):
        return self

    def sum(self):
        return self

    def item(self):
        return 16


@pytest.mark.parametrize(
    "name,capability,memory,broken,expected",
    [
        ("Tesla V100-PCIE-16GB", (7, 0), 16, False, False),
        ("NVIDIA A100", (8, 0), 40, False, False),
        ("NVIDIA A100", (8, 0), 80, False, True),
        ("NVIDIA H200", (9, 0), 141, False, True),
        ("NVIDIA H200", (9, 0), 141, True, False),
    ],
)
def test_gpu_preflight(name, capability, memory, broken, expected):
    def allocate(*args, **kwargs):
        if broken:
            raise RuntimeError("no kernel image is available for execution on the device")
        return FakeTensor()

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda i: SimpleNamespace(
                name=name, total_memory=memory * 1024 ** 3,
            ),
            get_device_capability=lambda i: capability,
            synchronize=lambda i: None,
        ),
        ones=allocate,
    )
    result = inspect_cuda_devices(torch)
    assert result["kernel_probe_ok"] is expected
    assert bool(result["errors"]) is (not expected)
    if broken:
        assert "CUDA kernel probe failed" in result["errors"][0]


def test_no_visible_cuda_devices():
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
    result = inspect_cuda_devices(torch)
    assert not result["kernel_probe_ok"]
    assert "No CUDA GPUs" in result["errors"][0]
