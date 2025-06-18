import os
import csv
from pathlib import Path
from contextlib import nullcontext
from typing import Callable, List, Optional
import time
import numpy as np
import click
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from vllm.model_executor.layers.activation import SiluAndMul
from vllm import _custom_ops as ops
import torch

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
    shard_intermediate_size: int,
    hidden_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    csv_writer: Optional[csv.DictWriter] = None,
) -> None:
    """Run benchmarking for Smart Split"""
    bs, cl = 1, bl
    num_tokens = bs * cl

    hidden_states = (torch.rand((bl, hidden_size), dtype=torch.float32, device=device) * 2 - 1).to(dtype=torch.bfloat16)
    hidden_states *= (rank + 1) / world_size

    weights_mlp1 = (torch.rand((2 * shard_intermediate_size, hidden_size), dtype=torch.float32, device=device) * 2 - 1).to(dtype=torch.bfloat16)
    weights_mlp1 *= (rank + 1) / world_size

    weights_mlp2 = (torch.rand((hidden_size, shard_intermediate_size), dtype=torch.float32, device=device) * 2 - 1).to(dtype=torch.bfloat16)
    weights_mlp2 *= (rank + 1) / world_size

    act_fn = SiluAndMul()
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device.index])

    def mlp_block_func(hidden_states):
        """MLP block function."""
        mlp1_out = torch.matmul(hidden_states, weights_mlp1.t())
        act_out = act_fn(mlp1_out)
        torch.matmul(act_out, weights_mlp2.t(), out=hidden_states)
        return hidden_states

    # Benchmark all
    def bmark(fn):
        torch.cuda.synchronize() 
        dist.barrier(device_ids=[device.index])
        time.sleep(2)
        return benchmark_with_event(fn, flush_l2=True)

    results = {}
    results["baseline_us"] = bmark(lambda: mlp_block_func(hidden_states))
    chunk_offsets = [0, 64, 128, 192, 256]
    elapsed_times = []
    for chunk_offset in chunk_offsets:
        chunk_size = bl // 2 + chunk_offset
        chunk_size = min(chunk_size, bl)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[device.index])
        time.sleep(2)
        split1_time = bmark(lambda: mlp_block_func(hidden_states[:chunk_size]))
        split2_time = bmark(lambda: mlp_block_func(hidden_states[chunk_size:]))
        torch.cuda.synchronize()
        dist.barrier(device_ids=[device.index])
        elapsed_times.append(split1_time + split2_time)
        results[f"offset_{chunk_offset}"] = split1_time + split2_time
    results["smart_us"] = min(elapsed_times)
    # Logging
    if rank == 0:
        print("\n=== Benchmark Results ===")
        print(f"{'BL':<25}: {bl}")
        print(f"{'Hidden Size':<25}: {hidden_size}")
        print(f"{'Data Type':<25}: {hidden_states.dtype}")
        for k, v in results.items():
            print(f"  {k:<25}: {v if isinstance(v, bool) else f'{v:,.2f} µs'}")
        print("==========================", flush=True)
        if csv_writer:
            row = {"BL": bl, "hidden_size": hidden_size, **results}
            csv_writer.writerow(row)


@click.command()
@click.option("--hidden-size", default=8192, type=int, help="Hidden dimension size")
@click.option("--output-dir", default="benchmark_results", type=str, help="Directory to save CSV results")
@click.option("--atol", default=4e-2, type=float, help="Absolute tolerance for correctness check")
@click.option("--rtol", default=4e-2, type=float, help="Relative tolerance for correctness check")
def main(hidden_size: int, output_dir: str, atol: float, rtol: float):
    """Main entrypoint to run benchmarks and save results."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)
    np.random.seed(42 + local_rank)
    world_size = dist.get_world_size()

    intermediate_size = 28672
    shard_intermediate_size = intermediate_size // world_size

    csv_file = None
    csv_writer = None

    try:
        if dist.get_rank() == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            csv_file = open(Path(output_dir) / f"figure_9_hs_{hidden_size}.csv", 'w', newline='')
            fieldnames = [
                "BL", "hidden_size",
                "baseline_us", "smart_us",
                "offset_0", "offset_64", "offset_128", "offset_192", "offset_256",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        for bl in [1024, 3072]:
            for bl_offset in [0, 256, 512, 768, 1024, 1280, 1536, 1792]:
                bl_updated = bl + bl_offset 
                if bl_updated > 3840:
                    continue
                benchmark(bl_updated, shard_intermediate_size, hidden_size, local_rank, world_size, device, csv_writer)
                time.sleep(2)
            time.sleep(10)
    finally:
        if csv_file:
            csv_file.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()