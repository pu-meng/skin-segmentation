import json
import entry as e
assert len(e.JOBS)==len(set(e.JOBS))==12
checks=[]
def check(name,expr):
 assert expr,name
 checks.append(name)
for count,expect in [(0,True),(1,True),(2,False),(3,False)]:
 ok,_=e.admission(1,0,True,{'compute_pids':list(range(count)),'free_mib':20000,'utilization':100},False)
 check('primary_count'+str(count),ok is expect)
check('secondary_reserved',not e.admission(1,1,False,{'compute_pids':[],'free_mib':20000},True)[0])
check('secondary_requires_resident_formal',not e.admission(1,1,True,{'compute_pids':[],'free_mib':20000},False)[0])
check('secondary_can_overlap_primary',e.admission(1,1,True,{'compute_pids':[1],'free_mib':20000},True)[0])
check('memory_reject_no_batch_reduction',not e.admission(1,0,True,{'compute_pids':[1],'free_mib':6000},True)[0])
real_read=e.read;real_upstream=e.upstream
state={'a234':'train','skip':'train','ready':False}
def fake_read(p):
 s=str(p)
 if 'a234_resume' in s:return {'status':state['a234']}
 if 'A1_SkipC' in s:return {'status':state['skip']}
 return {}
e.read=fake_read;e.upstream=lambda gpu:(state['ready'],'fixture same-card full reservation audit')
check('GPU0_protect_A234_phase_gap',not e.compatible(0,0)[0])
check('GPU1_share_SkipC_without_GPU0_dependency',e.compatible(1,0)[0])
check('GPU1_second_slot_stays_reserved',not e.compatible(1,1)[0])
state['skip']='complete'
check('yield_after_SkipC_for_missing3',not e.compatible(1,0)[0])
state['ready']=True;state['a234']='complete'
check('GPU0_release_after_own_reservations',e.compatible(0,0)[0])
check('GPU1_independent_release',e.compatible(1,1)[0])
e.read=real_read;e.upstream=real_upstream
print(json.dumps({'passed':True,'checks':checks,'count':len(checks),'arms':e.JOBS},indent=2))
