"""Small, synthetic, CPU-only recurrence demonstration; not a full Mamba model.

Demonstrates only the scalar version of V2 steps 4/5, not learned text/image
prediction, not CUDA performance, and not actual medical image results.
Input numbers stand for already-extracted image-token features.
"""
import json
import math
from pathlib import Path


def coefficients(token):
    # Fixed illustrative coefficients, not trained experimental weights.
    delta=math.log1p(math.exp(0.2*token))
    keep=math.exp(-delta)  # A=-1, so R=exp(delta*A).
    write=(0.3*token+0.1)*token  # J=B(T)*T, no delta injection (V2).
    read=0.2*token+0.4
    return delta,keep,write,read


def step(token,state):
    _,keep,write,read=coefficients(token)
    new_state=keep*state+write
    output=read*new_state+0.5*token  # State readout + D*T; D=0.5.
    return output,new_state


def run_sequence(tokens,state=0.0):
    outputs=[]
    for token in tokens:
        output,state=step(token,state)
        outputs.append(output)
    return outputs,state


if __name__=='__main__':
    tokens=[0.2,-0.4,0.8,0.1]
    full,full_state=run_sequence(tokens)
    state=0.0;incremental=[];trace=[]
    for index,token in enumerate(tokens,1):
        previous=state
        output,state=step(token,state)
        incremental.append(output)
        trace.append({'image_token':index,'feature':token,'old_state':previous,
                      'new_state':state,'output':output})
    first,saved_state=run_sequence(tokens[:2])
    second,chunk_state=run_sequence(tokens[2:],saved_state)
    assert full==incremental==first+second
    assert full_state==state==chunk_state
    result={'scope':'Synthetic scalar V2 steps 4/5 only; not a trained model or GPU benchmark',
            'all_image_features_known_at_start':tokens,'all_at_once_api_outputs':full,
            'one_token_at_a_time_outputs':incremental,'two_chunk_outputs':first+second,
            'exact_match':True,'trace':trace,
            'caveat':'All three use the same recurrence order. This proves streaming equivalence, not speedup or parallel scan implementation. T features are already available; encoder/conv streaming caches are outside this demo.'}
    Path(__file__).with_name('demo_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
