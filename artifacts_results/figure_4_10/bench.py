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
from vllm.model_executor.layers.layernorm import RMSNorm
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
    hidden_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    atol: float,
    rtol: float,
    ctas: int = 8,
    csv_writer: Optional[csv.DictWriter] = None,
) -> None:
    """Run benchmarking for Fused Kernel."""
    msg_sz_bytes = bl * hidden_size * torch.bfloat16.itemsize
    hidden_states = symm_mem.empty((bl, hidden_size), dtype=torch.bfloat16, device=device)
    symm_mem_hdl = symm_mem.rendezvous(hidden_states, dist.group.WORLD.group_name)

    ground_tensor = (torch.rand((bl, hidden_size), dtype=torch.float32, device=device) * 2 - 1).to(dtype=torch.bfloat16)
    ground_tensor *= (rank + 1) / world_size
    hidden_states.copy_(ground_tensor)

    residual = (2 * torch.rand((bl, hidden_size), dtype=torch.float32, device=device) - 1).to(dtype=torch.bfloat16)
    residual *= (rank + 1) / world_size
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device.index])
    torch.distributed.all_reduce(residual, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    torch.cuda.synchronize()

    # Create reference tensors
    ref_residual = residual.clone()
    ref_hidden_states = ground_tensor.clone()

    rms_norm = RMSNorm(hidden_size, eps=1e-5)
    weight_rms_norm = (2 * torch.rand(hidden_size, dtype=torch.float32, device=device) - 1).to(dtype=torch.bfloat16)
    weight_rms_norm *= (rank + 1) / world_size
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device.index])
    torch.distributed.all_reduce(weight_rms_norm, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    rms_norm.weight.data = weight_rms_norm

    blpr = bl // world_size
    start = rank * blpr
    end = (rank + 1) * blpr
    offset = rank * blpr * hidden_size * hidden_states.element_size()
    device_ids = [device.index]
    time.sleep(20)

    # RMSNorm functions
    def ref_rms_norm(): rms_norm(ref_hidden_states, ref_residual)
    def modified_rms_norm(MAX_CTAS=8): rms_norm(hidden_states, residual, MAX_CTAS=MAX_CTAS)
    def modified_rms_norm_partial(MAX_CTAS=8): rms_norm(hidden_states[start:end, :], residual[start:end, :], MAX_CTAS=MAX_CTAS)

    # Collective functions
    def torch_all_reduce(): torch.distributed.all_reduce(ref_hidden_states, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    def multimem_ar(MAX_CTAS=8): ops.multimem_all_reduce(hidden_states[start:end, :], symm_mem_hdl.multicast_ptr + offset, symm_mem_hdl.signal_pad_ptrs_dev, rank, world_size, MAX_CTAS)
    def multimem_rs(MAX_CTAS=8):
        ops.multimem_reduce_scatter(
            hidden_states[start:end, :],
            symm_mem_hdl.multicast_ptr + offset,
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank,
            world_size,
            MAX_CTAS
        )
    def multimem_ag(MAX_CTAS=8):
        ops.multimem_all_gather(
            hidden_states[start:end, :],
            symm_mem_hdl.multicast_ptr + offset,
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank,
            world_size,
            MAX_CTAS
        )
    def fused_ar_ln(MAX_CTAS=16):
        ops.fused_rs_ln_ag_cta(
            hidden_states[start:end, :],
            residual[start:end, :],
            rms_norm.weight.data,
            symm_mem_hdl.multicast_ptr + offset,
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank,
            world_size,
            MAX_CTAS,
            1e-5
        )
    def simple_fusion_ar_ln(MAX_CTAS=16):
        ops.simple_fusion_rs_ln_ag_cta(
            hidden_states[start:end, :],
            residual[start:end, :],
            rms_norm.weight.data,
            symm_mem_hdl.multicast_ptr + offset,
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank,
            world_size,
            MAX_CTAS,
            1e-5
        )

    # Set CTAs
    if ctas == 0:
        allreduce_ctas = 8
        rs_ctas = 8
        ag_ctas = 8
        partial_ctas = blpr
        full_ctas = bl
        fused_ctas = blpr
    else:
        allreduce_ctas = rs_ctas = ag_ctas = partial_ctas = full_ctas = fused_ctas = ctas

    torch.cuda.synchronize()
    dist.barrier(device_ids=device_ids)

    original_hidden, original_residual = hidden_states.clone(), residual.clone()
    ref_hidden_states.copy_(original_hidden)
    ref_residual.copy_(original_residual)

    torch_all_reduce()
    ref_rms_norm()
    expected_result = ref_hidden_states.clone()

    def check_correctness(run_fn):
        hidden_states.copy_(original_hidden)
        residual.copy_(original_residual)
        run_fn()
        return torch.allclose(expected_result, hidden_states, atol=atol, rtol=rtol)
    correct_multimem = check_correctness(lambda: (multimem_ar(full_ctas), modified_rms_norm(full_ctas)))
    correct_partial = check_correctness(lambda: (multimem_rs(rs_ctas), modified_rms_norm_partial(partial_ctas), multimem_ag(ag_ctas)))
    correct_fused = check_correctness(lambda: fused_ar_ln(fused_ctas))
    correct_simple_fusion = check_correctness(lambda: simple_fusion_ar_ln(fused_ctas))

    torch.cuda.synchronize()
    dist.barrier(device_ids=device_ids)
    time.sleep(20)
    results = {"correctness": all([correct_multimem, correct_partial, correct_fused, correct_simple_fusion])}

    # Benchmark all
    def bmark(fn):
        torch.cuda.synchronize() 
        dist.barrier(device_ids=device_ids)
        time.sleep(2)
        return benchmark_with_event(fn, flush_l2=True)

    results.update({
        "full_rms_norm_us": bmark(lambda: modified_rms_norm(full_ctas)),
        "partial_rms_norm_us": bmark(lambda: modified_rms_norm_partial(partial_ctas)),
        "torch_all_reduce_us": bmark(torch_all_reduce),
        "multimem_all_reduce_us": bmark(lambda: multimem_ar(allreduce_ctas)),
        "multimem_rs_us": bmark(lambda: multimem_rs(rs_ctas)),
        "multimem_ag_us": bmark(lambda: multimem_ag(ag_ctas)),
        "fused_arln_us": bmark(lambda: fused_ar_ln(fused_ctas)),
        "simple_fusion_arln_us": bmark(lambda: simple_fusion_ar_ln(fused_ctas)),
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

    csv_file = None
    csv_writer = None

    try:
        if dist.get_rank() == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            csv_file = open(Path(output_dir) / f"figure_4_10_hs_{hidden_size}.csv", 'w', newline='')
            fieldnames = [
                "BL", "hidden_size", "multimem_ctas", "correctness",
                "full_rms_norm_us", "partial_rms_norm_us", "torch_all_reduce_us",
                "multimem_all_reduce_us", "fused_arln_us", "simple_fusion_arln_us", 
                "multimem_rs_us", "multimem_ag_us"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        for bl in [2 ** i for i in range(6, 16)]:
            for ctas in [2, 4, 8, 16, 32, 64, 128, 0]: # zero means default
                benchmark(bl, hidden_size, local_rank, world_size, device, atol, rtol, ctas, csv_writer)    
                time.sleep(5) 
            time.sleep(10)
    finally:
        if csv_file:
            csv_file.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()