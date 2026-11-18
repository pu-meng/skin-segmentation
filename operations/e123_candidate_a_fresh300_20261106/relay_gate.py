def gate(alive, exitcode, budget, artifacts):
    if exitcode is not None and exitcode != 0:
        return 'failed'
    if alive:
        return 'wait'
    return 'ready' if exitcode == 0 and budget and artifacts else 'failed'
