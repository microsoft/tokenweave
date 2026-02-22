import os
import csv
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from vllm import _custom_ops as ops
import click


def benchmark_with_event(
    target_fn: Callable[[], None],
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    profile_ranks: Optional[List[int]] = None,
    flush_l2: bool = False,
    cuda_graph: bool = False,
) -> float:
    """Benchmark a function using CUDA events."""
    if cuda_graph:
        target_fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            target_fn()
        target_fn = lambda: g.replay()

    if "BENCHMARK_ITERS" in os.environ:
        benchmark_iters = int(os.environ["BENCHMARK_ITERS"])

    rank = dist.get_rank() if dist.is_initialized() else 0
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        target_fn()

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()

    begin_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    with nullcontext():
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            begin_events[i].record()
            target_fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]
    return torch.tensor(latencies).median().item() * 1000


@torch.inference_mode
def benchmark(
    bl: int,
    hidden_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    ctas: int,
    atol: float,
    rtol: float,
    csv_writer: Optional[csv.DictWriter] = None,
) -> None:
    """Run benchmarking for PyTorch and MultiMem collectives."""

    msg_sz_bytes = bl * hidden_size * torch.bfloat16.itemsize
    hidden_states = symm_mem.empty((bl, hidden_size), dtype=torch.bfloat16, device=device)
    symm_mem_hdl = symm_mem.rendezvous(hidden_states, dist.group.WORLD.group_name)

    ground_tensor = (torch.rand((bl, hidden_size), dtype=torch.float32, device=device) * 2 - 1).to(dtype=torch.bfloat16)
    ground_tensor *= (rank + 1) / world_size
    hidden_states.copy_(ground_tensor)

    x = torch.empty_like(hidden_states)
    x.copy_(ground_tensor)

    rs_out = torch.randn((bl // world_size, hidden_size), device=device, dtype=torch.bfloat16)

    blpr = bl // world_size
    start, end = rank * blpr, (rank + 1) * blpr
    offset = rank * blpr * hidden_size * hidden_states.element_size()
    device_ids = [device.index]

    assert x.data_ptr() != hidden_states.data_ptr(), "x and hidden_states unexpectedly share memory!"

    # Define baseline PyTorch collectives
    def pytorch_dist_ar(): dist.all_reduce(x, op=dist.ReduceOp.SUM)
    def pytorch_dist_rs(): dist.reduce_scatter_tensor(rs_out, x, op=dist.ReduceOp.SUM)
    def pytorch_dist_ag(): dist.all_gather_into_tensor(x, rs_out)

    # Define MultiMem collectives
    def multimem_ar(): ops.multimem_all_reduce(hidden_states[start:end], symm_mem_hdl.multicast_ptr + offset, symm_mem_hdl.signal_pad_ptrs_dev, rank, world_size, ctas)
    def multimem_rs(): ops.multimem_reduce_scatter(hidden_states[start:end], symm_mem_hdl.multicast_ptr + offset, symm_mem_hdl.signal_pad_ptrs_dev, rank, world_size, ctas)
    def multimem_ag(): ops.multimem_all_gather(hidden_states[start:end], symm_mem_hdl.multicast_ptr + offset, symm_mem_hdl.signal_pad_ptrs_dev, rank, world_size, ctas)

    # Run and verify correctness
    torch.cuda.synchronize()
    dist.barrier(device_ids=device_ids)
    pytorch_dist_rs()
    pytorch_dist_ag()
    torch.cuda.synchronize()
    dist.barrier(device_ids=device_ids)
    multimem_rs()
    multimem_ag()
    torch.cuda.synchronize()
    dist.barrier(device_ids=device_ids)
    results = {"correctness": torch.allclose(x, hidden_states, atol=atol, rtol=rtol)}
    # Benchmark all
    def bmark(fn):
        torch.cuda.synchronize() 
        dist.barrier(device_ids=device_ids)
        time.sleep(2)
        return benchmark_with_event(fn, flush_l2=True)

    results.update({
        "pytorch_dist_ar_us": bmark(pytorch_dist_ar),
        "pytorch_dist_rs_us": bmark(pytorch_dist_rs),
        "pytorch_dist_ag_us": bmark(pytorch_dist_ag),
        "multimem_ar_us": bmark(multimem_ar),
        "multimem_rs_us": bmark(multimem_rs),
        "multimem_ag_us": bmark(multimem_ag),
    })

    # Logging
    if rank == 0:
        print("\n=== Benchmark Results ===")
        print(f"{'BL':<25}: {bl}")
        print(f"{'Hidden Size':<25}: {hidden_size}")
        print(f"{'Data Type':<25}: {hidden_states.dtype}")
        print(f"{'Message Size (bytes)':<25}: {msg_sz_bytes}")
        print(f"{'CTAs':<25}: {ctas}")
        for k, v in results.items():
            print(f"  {k:<25}: {v if isinstance(v, bool) else f'{v:,.2f} µs'}")
        print("==========================", flush=True)
        if csv_writer:
            row = {"BL": bl, "hidden_size": hidden_size, "multimem_ctas": ctas, **results}
            csv_writer.writerow(row)


@click.command()
@click.option("--hidden-size", default=8192, type=int, help="Hidden dimension size")
@click.option("--output-dir", default="benchmark_results", type=str, help="Directory to save CSV results")
@click.option("--atol", default=2e-2, type=float, help="Absolute tolerance for correctness check")
@click.option("--rtol", default=2e-2, type=float, help="Relative tolerance for correctness check")
def main(hidden_size: int, output_dir: str, atol: float, rtol: float):
    """Main entrypoint to run benchmarks and save results."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)
    np.random.seed(42 + local_rank)
    world_size = dist.get_world_size()

    csv_file = None
    csv_writer = None

    try:
        if dist.get_rank() == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            csv_file = open(Path(output_dir) / f"figure_3_4_hs_{hidden_size}.csv", 'w', newline='')
            fieldnames = [
                "BL", "hidden_size", "multimem_ctas", "correctness",
                "pytorch_dist_ar_us", "multimem_ar_us",
                "pytorch_dist_rs_us", "multimem_rs_us",
                "pytorch_dist_ag_us", "multimem_ag_us"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        for ctas in [2, 4, 8, 16, 32]:
            for bl in [2 ** i for i in range(6, 16)]:
                benchmark(bl, hidden_size, local_rank, world_size, device, ctas, atol, rtol, csv_writer)
                time.sleep(10)
    finally:
        if csv_file:
            csv_file.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
