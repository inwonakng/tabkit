import torch


def gpu_usage() -> list[str]:
    stats = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        stats.append(
            f"Device {i}: {free/1024**3:.2f}GB free / {total/1024**3:.2f}GB total"
        )
    return stats
