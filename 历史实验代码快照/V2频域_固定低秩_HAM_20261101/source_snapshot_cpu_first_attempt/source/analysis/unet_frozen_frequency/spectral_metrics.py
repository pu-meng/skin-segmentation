"""TinyViM-style diagonal curve plus explicitly defined non-DC band statistics."""
import math
import torch


def measure(x):
    x=x.detach().float()
    assert x.ndim==4 and x.shape[-1]==x.shape[-2]
    h,w=x.shape[-2:]
    fft=torch.fft.fft2(x,norm='ortho')
    # TinyViM freq_vis.visualize_amp averages log magnitude over samples/channels,
    # then extracts the positive diagonal and subtracts its DC value.
    logmag=torch.fft.fftshift(fft.abs().clamp_min(1e-12).log(),dim=(-2,-1))
    diagonal=logmag.mean((0,1)).diagonal()[h//2:]
    relative=diagonal-diagonal[0]
    # Centering excludes channel means/DC; band shares are invariant to gain.
    centered=x-x.mean((-2,-1),keepdim=True)
    power=torch.fft.fft2(centered,norm='ortho').abs().square()
    fy=torch.fft.fftfreq(h,device=x.device);fx=torch.fft.fftfreq(w,device=x.device)
    radius=(fy[:,None].square()+fx[None,:].square()).sqrt()
    low=(radius>0)&(radius<=0.125)
    high=radius>=0.25
    energy=power.sum((1,2,3))
    valid=energy>1e-20
    low_energy=power[:,:,low].sum((1,2));high_energy=power[:,:,high].sum((1,2))
    return dict(diagonal_relative_log_amplitude=relative.cpu().tolist(),
                low_non_dc_power_share=(low_energy[valid]/energy[valid]).mean().item() if valid.any() else None,
                high_non_dc_power_share=(high_energy[valid]/energy[valid]).mean().item() if valid.any() else None,
                non_dc_power_mean=energy.mean().item(),dc_power_mean=fft[:,:,0,0].abs().square().sum(1).mean().item(),
                zero_non_dc_images=int((~valid).sum()),samples=x.shape[0])


def self_check():
    n=32;axis=torch.arange(n).float()
    low=torch.cos(2*math.pi*2*axis/n)[None,None,None,:].expand(2,3,n,n)
    high=torch.cos(2*math.pi*12*axis/n)[None,None,None,:].expand(2,3,n,n)
    a,b=measure(low),measure(high)
    assert a['low_non_dc_power_share']>0.999
    assert b['high_non_dc_power_share']>0.999
    assert abs(measure(7*low+3)['low_non_dc_power_share']-a['low_non_dc_power_share'])<1e-5
    assert measure(torch.ones(1,1,n,n))['low_non_dc_power_share'] is None
    assert all(abs(x)<1e-5 for x in measure(torch.nn.functional.one_hot(torch.tensor(0),n*n).float().reshape(1,1,n,n))['diagonal_relative_log_amplitude'])
    return dict(low_sine_pass=True,high_sine_pass=True,gain_and_offset_invariance_pass=True,constant_handling_pass=True,impulse_flat_spectrum_pass=True)


@torch.no_grad()
def band_response(mixer, x, seed=20261025, relative_rms=0.01):
    """Local sensitivity, not evidence of task usefulness. Perturb mixer inputs."""
    b,c,h,w=x.shape
    generator=torch.Generator(device=x.device).manual_seed(seed)
    noise=torch.randn(x.shape,device=x.device,generator=generator)
    freq=torch.fft.fft2(noise,norm='ortho')
    fy=torch.fft.fftfreq(h,device=x.device);fx=torch.fft.fftfreq(w,device=x.device)
    r=(fy[:,None].square()+fx[None,:].square()).sqrt()
    scale=x.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-8)*relative_rms
    tokens=x.flatten(2).transpose(1,2).contiguous();reference=mixer(tokens)
    result={}
    for name,mask in [('low',(r>0)&(r<=.125)),('high',r>=.25)]:
        perturb=torch.fft.ifft2(freq*mask,norm='ortho').real
        perturb=perturb/perturb.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-12)*scale
        delta=mixer((x+perturb).flatten(2).transpose(1,2).contiguous())-reference
        gain=delta.flatten(1).norm(dim=1)/perturb.flatten(1).norm(dim=1).clamp_min(1e-12)
        result[name+'_local_l2_gain']=gain.mean().item()
    return result
