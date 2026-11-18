"""CPU verification of the illustrated axes and reset-after GRU formulas.

This validates the explanatory equations; it does not run TensorFlow or train.
"""
from pathlib import Path
import json
import torch


def check():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    B, N, K, D = 2, 13, 3, 8
    keys, values, queries = torch.randn(B,N,D), torch.randn(B,N,D), torch.randn(B,K,D)
    logits = keys @ queries.transpose(-1,-2) * D**-0.5
    attention = logits.softmax(dim=-1)
    torch.testing.assert_close(attention.sum(-1),torch.ones(B,N))
    weights = attention + 1e-8
    weights = weights / weights.sum(dim=-2,keepdim=True)
    torch.testing.assert_close(weights.sum(-2),torch.ones(B,K))
    updates = weights.transpose(-1,-2) @ values
    assert updates.shape == (B,K,D)
    perm = torch.randperm(N)
    logits_perm = keys[:,perm] @ queries.transpose(-1,-2) * D**-0.5
    wp = logits_perm.softmax(-1)+1e-8
    wp = wp/wp.sum(-2,keepdim=True)
    torch.testing.assert_close(wp.transpose(-1,-2)@values[:,perm],updates)
    # Keras kernels store gate chunks in z,r,n order; PyTorch uses r,z,n.
    Wx, Wh = torch.randn(D,3*D), torch.randn(D,3*D)
    bx, bh = torch.randn(3*D),torch.randn(3*D)
    old = torch.randn(B,K,D)
    az,ar,an = (updates@Wx+bx).chunk(3,-1)
    vz,vr,vn = (old@Wh+bh).chunk(3,-1)
    reset,update = torch.sigmoid(ar+vr),torch.sigmoid(az+vz)
    candidate = torch.tanh(an+reset*vn)
    manual = update*old+(1-update)*candidate
    cell = torch.nn.GRUCell(D,D)
    def reorder(x,dim):
        z,r,n=x.chunk(3,dim=dim)
        return torch.cat([r,z,n],dim=dim)
    with torch.no_grad():
        cell.weight_ih.copy_(reorder(Wx.T,0)); cell.weight_hh.copy_(reorder(Wh.T,0))
        cell.bias_ih.copy_(reorder(bx,0)); cell.bias_hh.copy_(reorder(bh,0))
        actual = cell(updates.reshape(B*K,D),old.reshape(B*K,D)).reshape(B,K,D)
    torch.testing.assert_close(actual,manual,rtol=1e-5,atol=1e-6)
    return {'passed':True,'device':'cpu','training_performed':False,
            'slot_softmax_axis_K':True,'weighted_mean_axis_N':True,
            'input_permutation_keeps_updates':True,'GRU_reset_after_equations_match_PyTorch_cell':True,
            'max_GRU_abs_error':float((actual-manual).abs().max()),
            'scope':'Equation check only; TensorFlow official model was read, not executed.'}


if __name__=='__main__':
    result=check()
    root=Path(__file__).resolve().parents[3]
    out=root/'log/研究记录_辅助产物/AD-GBC_SlotAttention_GRU'
    out.mkdir(parents=True,exist_ok=True)
    (out/'formula_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
