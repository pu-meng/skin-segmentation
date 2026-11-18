def decide(states):
 if any(s in ('failed','blocked','failed_downstream_blocked') for s in states):
  raise RuntimeError('Upstream failure; downstream blocked')
 return all(s=='complete' for s in states)
