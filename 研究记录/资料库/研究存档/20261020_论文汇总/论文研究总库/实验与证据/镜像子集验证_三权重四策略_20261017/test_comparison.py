from reporting import compare
base=[{'case':'pos','tumor_positive':True,'tumor_dice_pct':60.,'liver_dice_pct':90.,'tumor_fp_voxels':10},{'case':'neg','tumor_positive':False,'tumor_dice_pct':None,'liver_dice_pct':80.,'tumor_fp_voxels':0}]
new=[dict(x) for x in base]; new[0]['tumor_dice_pct']=59.6; new[1]['tumor_fp_voxels']=4
v=compare(base,new)
assert v['observed_criteria_passed']
assert abs(v['tumor_mean_delta_pp']+0.4)<1e-10
assert v['negative_fp_voxels_current']==4
new[0]['tumor_dice_pct']=57.
assert not compare(base,new)['observed_criteria_passed']
try: compare(base,new[:1])
except ValueError: pass
else: raise AssertionError('incomplete cases accepted')
print('PASS comparison: paired percent/pp, positive-only tumor, negative FP, threshold failure, missing case rejection')
