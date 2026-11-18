def upstream_ready(state,queue_alive,children_alive):
    return state.get('status')=='complete' and not queue_alive and not children_alive
def gpu_idle(sample):
    return not sample['compute_pids'] and sample['memory_used_mib']<=1024 and sample['utilization_pct']<=5
