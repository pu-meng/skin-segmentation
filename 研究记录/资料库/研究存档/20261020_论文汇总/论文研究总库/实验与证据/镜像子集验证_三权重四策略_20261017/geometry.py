"""Physical geometry checks: exact dimensions, bounded absolute errors, no resampling."""
import itertools
import numpy as np
import SimpleITK as sitk

POSITION_TOLERANCE_MM = 0.001
DIRECTION_TOLERANCE = 1e-6

def header(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return dict(size=list(reader.GetSize()), spacing=list(reader.GetSpacing()),
                origin=list(reader.GetOrigin()), direction=list(reader.GetDirection()))

def compare_geometry(a, b):
    if a['size'] != b['size'] or len(a['size']) != 3:
        raise ValueError('Geometry size mismatch: '+repr((a['size'], b['size'])))
    for g in (a, b):
        values = np.concatenate([g[k] for k in ('spacing','origin','direction')])
        if not np.isfinite(values).all() or min(g['spacing']) <= 0:
            raise ValueError('Nonfinite or invalid geometry')
    direction_delta = float(np.max(np.abs(np.array(a['direction'])-b['direction'])))
    def physical(g, index):
        return np.array(g['origin']) + np.array(g['direction']).reshape(3,3) @ (np.array(g['spacing'])*index)
    corners = itertools.product(*[(0,n-1) for n in a['size']])
    displacement = max(float(np.linalg.norm(physical(a,np.array(i))-physical(b,np.array(i)))) for i in corners)
    result = dict(max_corner_displacement_mm=displacement,
                  max_spacing_difference_mm=float(np.max(np.abs(np.array(a['spacing'])-b['spacing']))),
                  max_origin_difference_mm=float(np.max(np.abs(np.array(a['origin'])-b['origin']))),
                  max_direction_difference=direction_delta,
                  position_tolerance_mm=POSITION_TOLERANCE_MM, direction_tolerance=DIRECTION_TOLERANCE,
                  exact_equal=a==b)
    if displacement > POSITION_TOLERANCE_MM or direction_delta > DIRECTION_TOLERANCE:
        raise ValueError('Physical geometry exceeds tolerance: '+repr(result))
    return result

def audit_cases(cases, images, gt, pred=None):
    rows=[]
    for case in cases:
        try:
            image=header(images/f'{case}_0000.nii.gz'); reference=header(gt/f'{case}.nii.gz')
            row=dict(case=case,input_vs_reference=compare_geometry(image,reference))
            if pred is not None:
                prediction=header(pred/f'{case}.nii.gz')
                row.update(prediction_vs_input=compare_geometry(prediction,image),
                           prediction_vs_reference=compare_geometry(prediction,reference))
            rows.append(row)
        except Exception as e:
            raise ValueError(f'{case}: {e}') from e
    return dict(passed=True,cases=len(rows),rows=rows,position_tolerance_mm=POSITION_TOLERANCE_MM,
                direction_tolerance=DIRECTION_TOLERANCE,resampled=False)
