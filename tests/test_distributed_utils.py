from utils import distributed


def test_install_mixed_pg_default_device_warning_filter_suppresses_exact_warnings():
    distributed.install_mixed_pg_default_device_warning_filter()

    assert any(
        item[2] == UserWarning
        and "`_get_pg_default_device` will be deprecated" in str(item[1])
        and getattr(item[3], "pattern", None) == "torch\\.distributed\\.distributed_c10d"
        for item in __import__("warnings").filters
    )
    assert any(
        item[2] == UserWarning
        and "Multiple backends are registered with this ProcessGroup" in str(item[1])
        and getattr(item[3], "pattern", None) == "torch\\.distributed\\.distributed_c10d"
        for item in __import__("warnings").filters
    )


def test_local_cuda_barrier_device_id_uses_local_rank_mod_device_count(monkeypatch):
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "5")

    assert distributed._local_cuda_barrier_device_id() == 1


def test_dist_barrier_passes_device_ids_when_cuda_is_available(monkeypatch):
    calls = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "_local_cuda_barrier_device_id", lambda: 2)
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda **kwargs: calls.append(kwargs),
    )

    distributed.dist_barrier()

    assert calls == [{"device_ids": [2]}]


def test_dist_barrier_is_noop_when_process_group_is_not_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda **kwargs: calls.append(kwargs),
    )

    distributed.dist_barrier()

    assert calls == []


def test_object_collective_device_is_cpu():
    assert distributed.object_collective_device().type == "cpu"
