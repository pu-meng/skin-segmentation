"""Check official loading, inherited unused parameters, geometry, and updates."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import torch

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from Models.Encoder2026.candidates import classifier,build_model,SPECS,MATERIALS


def main(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report={'name':name,'status':'running','test_data_accessed':False,'formal_training_started':False}
    destination=MATERIALS/(name+'_integration.json');destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        reference,loading=classifier(name,pretrained=True)
        report.update(loading)
        reference=reference.cuda().train()
        x=torch.rand(2,3,256,256,device='cuda')
        reference(x).square().mean().backward()
        inherited={n for n,p in reference.named_parameters() if p.grad is None}
        report['official_classifier_unused_parameters']=sorted(inherited)
        if name=='convneur_m2':
            # Reproduce upstream aliasing loader and compare functional outputs.
            from model.convneur import ConvNeur
            raw=ConvNeur(depths=[2,2,6,2],dims=[48,96,192,384],mem_dim=96,chunk_size=196).cuda().eval()
            state=torch.load(loading['weight'],map_location='cuda',weights_only=True)['model']
            raw.load_state_dict(state,strict=True)
            reference.eval()
            with torch.no_grad():
                raw_outputs=raw.forward_features(x);adapted=reference.forward_features(x)
            errors=[float((a-b).abs().max()) for a,b in zip(raw_outputs,adapted)]
            for a,b in zip(raw_outputs,adapted):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            report['storage_alias_fix_max_feature_errors']=errors
            del raw,state,raw_outputs,adapted
        del reference
        model=build_model(name,pretrained=True).cuda().train()
        from pretrained_unet2d import SharedDecoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);decoder=SharedDecoder()
        assert all(torch.equal(v.cpu(),decoder.state_dict()[k]) for k,v in model.decoder.state_dict().items())
        del decoder
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        prediction=model(x);loss=prediction.square().mean();loss.backward()
        if tuple(prediction.shape)!=(2,1,256,256):raise AssertionError('Wrong segmentation shape')
        missing={n for n,p in model.named_parameters() if p.grad is None}
        expected={'encoder.backbone.'+n for n in inherited if 'encoder.backbone.'+n in dict(model.named_parameters())}
        if missing!=expected:raise AssertionError({'new_missing':sorted(missing-expected),'unexpected_used':sorted(expected-missing)})
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise AssertionError('Nonfinite gradient')
        before={n:p.detach().clone() for n,p in model.align.named_parameters()}
        optimizer.step()
        assert all(not torch.equal(p,before[n]) for n,p in model.align.named_parameters())
        report.update(status='passed',allowed_missing_gradients=sorted(expected),
                      feature_shapes=model.encoder.native_shapes,output=list(prediction.shape),
                      full_segmentation_parameters=sum(p.numel() for p in model.parameters()),
                      encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                      shared_decoder_initialization_equal=True,all_alignment_parameters_updated=True,
                      finite_gradients=True,geometry=SPECS[name]['geometry'])
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,report['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=SPECS)
    main(parser.parse_args().name)
