import importlib
import io
import json
import os
import sys
from pathlib import Path


def main():
    mode = sys.argv[1]
    if mode == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

    import torch
    from candidate import (TrainingCriterion, auxiliary_box_loss,
                           build_candidate, old, soft_rectangle)
    from pretrained_unet2d import prepare_imports
    prepare_imports('tinyvim_s')
    prepare_imports('mambavision_t')
    from experiment import RUN, prepare
    import ham_fast_runner as runner

    torch.set_num_threads(1 if mode == 'cpu' else 4)
    out = {'device': mode, 'checks': {}, 'gradient_steps': [],
           'test_data_accessed': False}
    original_modules = []
    loaders = {}

    def check(key, passed):
        out['checks'][key] = bool(passed)
        assert passed, key
        print('PASS', key, flush=True)

    if mode == 'cpu':
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        class Reference:
            @staticmethod
            def apply(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=False, nrows=1):
                assert u.device.type == 'cpu'
                return selective_scan_ref(
                    u, delta, A, B, C, D, z=None,
                    delta_bias=delta_bias, delta_softplus=delta_softplus)

        for name in ['Models.TinyViM.blocks', 'model.tvimblock']:
            module = importlib.import_module(name)
            original_modules.append((module, module.SelectiveScan))
            module.SelectiveScan = Reference

    try:
        config = prepare()
        api = runner.baseline_api()
        api.seed_all(42)
        device = torch.device('cpu' if mode == 'cpu' else 'cuda:0')
        baseline = old.build_candidate(pretrained=True, arm='E3').eval()
        model = build_candidate(pretrained=True).eval()
        check('original_E3_common_initialization_exact',
              all(torch.equal(value, model.state_dict()[key])
                  for key, value in baseline.state_dict().items()))
        check('official_encoder',
              model.encoder.loading_audit['all_loaded_tensors_equal'])
        out['parameters_original_E3'] = sum(p.numel() for p in baseline.parameters())
        out['parameters'] = sum(p.numel() for p in model.parameters())
        out['added_parameters'] = out['parameters'] - out['parameters_original_E3']
        model = model.to(device)
        baseline = baseline.to(device)
        x = torch.rand(1, 3, 256, 256, device=device)
        model.decoder.constraint_enabled = False
        with torch.no_grad():
            reference_output = baseline(x)
            candidate_output = model(x)
        out['W1_max_abs_error'] = float((reference_output - candidate_output).abs().max())
        check('W1_original_output_equal', torch.equal(reference_output, candidate_output))
        model.decoder.constraint_enabled = True
        del baseline

        shapes = {}
        hooks = []
        for name, module in model.named_modules():
            if (name in ['encoder', 'align', 'decoder.enhance', 'decoder.ups.0',
                         'decoder.stages.0', 'decoder.stages.1',
                         'decoder.stages.2', 'decoder.stages.3',
                         'decoder.box_head'] or name.startswith('align.')):
                def record_shape(mod, args, result, module_name=name):
                    shapes[module_name] = {
                        'input': [list(value.shape) for value in args
                                  if torch.is_tensor(value)],
                        'output': (list(result.shape) if torch.is_tensor(result)
                                   else [list(value.shape) for value in result]),
                    }
                hooks.append(module.register_forward_hook(record_shape))

        if mode != 'cpu':
            from ham_fast_pipeline import make_fast_loaders
            loaders, generator = make_fast_loaders(
                config, api, workers=2, validation_batch=8)
            iterator = iter(loaders['train'])
            torch.cuda.reset_peak_memory_stats()

        model.train()
        segmentation_loss = api.BceDiceLoss().to(device)
        criterion = TrainingCriterion(model, segmentation_loss)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001,
                                      weight_decay=.01)
        for step in range(3):
            if mode == 'cpu':
                x = torch.rand(1, 3, 256, 256)
                y = torch.zeros(1, 1, 256, 256)
                y[:, :, 47:181, 63:197] = 1
                image_ids = ['synthetic']
            else:
                x, y, image_ids, _ = next(iterator)
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            box = model.decoder.last_box
            check('output_shape_' + str(step),
                  tuple(prediction.shape) == (len(x), 1, 256, 256))
            seg_loss = segmentation_loss(prediction, y)
            box_loss = auxiliary_box_loss(box, y)
            coord_grad = torch.autograd.grad(
                seg_loss, box, retain_graph=True)[0]
            check('segmentation_gradient_all_coordinates_' + str(step),
                  bool(torch.isfinite(coord_grad).all() and
                       (coord_grad.abs().sum(0) > 0).all()))
            box_head = [parameter for name, parameter in model.named_parameters()
                        if 'box_head' in name]
            aux_grad = torch.autograd.grad(box_loss, box_head, retain_graph=True)
            check('auxiliary_loss_trains_head_' + str(step),
                  all(torch.isfinite(grad).all() and grad.norm() > 0
                      for grad in aux_grad))
            loss = criterion(prediction, y)
            loss.backward()
            check('all_gradients_finite_' + str(step),
                  all(parameter.grad is not None and
                      torch.isfinite(parameter.grad).all()
                      for parameter in model.parameters()))
            head_gradients = {
                name: float(parameter.grad.double().norm())
                for name, parameter in model.named_parameters()
                if 'box_head' in name
            }
            out['gradient_steps'].append({
                'step': step, 'seg_loss': float(seg_loss.detach()),
                'box_loss': float(box_loss.detach()),
                'coordinate_gradients': coord_grad.detach().cpu().tolist(),
                'head_gradients': head_gradients,
                'image_ids': list(image_ids),
            })
            if step == 2:
                check('all_original_E3_parameters_effective',
                      all(parameter.grad.norm() > 0
                          for name, parameter in model.named_parameters()
                          if name.startswith('decoder.enhance.')))
            optimizer.step()
            print('OPTIMIZER_STEP', step, float(loss.detach()), flush=True)

        for hook in hooks:
            hook.remove()
        out['shapes'] = shapes
        check('decoder_input_to_box',
              [*shapes['decoder.ups.0']['output'][1:]] == [64, 64, 64])
        feature = torch.randn(1, 64, 64, 64, device=device)
        enhanced = model.decoder.enhance(feature)
        box = torch.tensor([[.5, .5, .5, .5]], device=device)
        weight = soft_rectangle(box, 64, 64)
        updated = feature + weight * (enhanced - feature)
        out['spatial'] = {
            'center_weight': float(weight[0, 0, 32, 32]),
            'corner_weight': float(weight[0, 0, 0, 0]),
            'center_correction': float((updated-feature)[:, :, 32, 32].abs().mean()),
            'corner_correction': float((updated-feature)[:, :, 0, 0].abs().mean()),
        }
        check('outside_skip_preserved_softly',
              out['spatial']['corner_weight'] < .001 and
              out['spatial']['center_weight'] > .9)
        model.eval()
        with torch.no_grad():
            expected = model(x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_candidate(pretrained=False)
        restored.load_state_dict(
            torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
        restored = restored.to(device).eval()
        with torch.no_grad():
            actual = restored(x)
        check('strict_roundtrip', torch.equal(expected, actual))
        if mode != 'cpu':
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(loaders['val'].dataset, [0, 1]),
                batch_size=1, num_workers=0)
            validation = api.evaluate(model, val_loader, segmentation_loss, device)
            check('real_validation', validation['images'] == 2)
            out['validation'] = validation
            out.update(
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                formal_batch=8, real_augmented_training=True)
        out.update(passed=True, preflight_updates_discarded=True,
                   protocol_sha256=runner.sha256_file(RUN/'protocol.json'))
    except BaseException as exc:
        out.update(passed=False, error=repr(exc))
        raise
    finally:
        for module, original_scan in original_modules:
            module.SelectiveScan = original_scan
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
        out.update(scan_reference_cpu_only=mode == 'cpu',
                   scan_restored=all(module.SelectiveScan is original_scan
                                     for module, original_scan in original_modules),
                   cuda_initialized=torch.cuda.is_initialized())
        runner.write_json(RUN / (mode + '_acceptance.json'), out)
    print('ACCEPTANCE_PASS', mode, flush=True)


if __name__ == '__main__':
    main()
