from rea_video.ort_providers import pick

ALL = ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_prefers_the_gpu_and_keeps_the_cpu_as_fallback():
    assert pick(ALL) == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_a_cpu_only_build_runs_on_the_cpu():
    assert pick(["AzureExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_the_env_value_forces_the_cpu():
    assert pick(ALL, "CPUExecutionProvider") == ["CPUExecutionProvider"]


def test_an_unknown_provider_in_the_env_value_is_skipped():
    assert pick(ALL, "CUDAExecutionProvider,DmlExecutionProvider") == ["DmlExecutionProvider", "CPUExecutionProvider"]
