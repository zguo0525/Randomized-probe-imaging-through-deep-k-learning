# pytorch tools used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains pytorch tools to generate approximants and iterative reconstruction for randomized probe imaging."""

from torch.nn.functional import pad
import torch as t
import numpy as np


def propagate(im):
    """Propagates a wavefield to the far field
    """
    shifted = t.fft.ifftshift(im, dim=(-1,-2))
    propagated = t.fft.fft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))

 
def backpropagate(im):
    """Propagates a wavefield to the near field
    """
    shifted = t.fft.ifftshift(im, dim=(-1,-2))
    propagated = t.fft.ifft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))


def expand_probe(probe, im):
    """Upsamples the probe to be big enough in reciprocal space to 
    avoid wrapping effects when the probe and im are multiplied
    """
    fftprobe = propagate(probe)
    fftprobe = pad(fftprobe, (im.shape[1]//2, im.shape[1]//2,
                              im.shape[0]//2, im.shape[0]//2))
    return backpropagate(fftprobe)

 
def interact(im, probe):
    """Upsamples the image to the correct size and multiplies with the probe
    """
    fftim = propagate(im)
    pad0= (probe.shape[0] - im.shape[0])//2
    pad1 = (probe.shape[1] - im.shape[1])//2
    fftim = pad(fftim, (pad1, pad1, pad0, pad0))
    upsampled_im = backpropagate(fftim)
    return upsampled_im * probe

 
def measure(pattern, probe_shape=None, eps=1e-8):
    """Generates intensities corresponding to a subset of the wavefield
    """
    if probe_shape is None:
        return t.abs(pattern)**2 + eps # for numerical stability
    
    pad0 = (pattern.shape[0] - probe_shape[0])//2        
    pad1 = (pattern.shape[1] - probe_shape[1])//2
    if pad0 == 0  and pad1 == 0:
        return t.abs(pattern)**2 + eps # for numerical stability
    elif pad0 == 0:
        return t.abs(pattern[:,pad1:-pad1])**2 + eps # for numerical stability
    elif pad1 == 0:
        return t.abs(pattern[pad0:-pad0,:])**2 + eps # for numerical stability
    else:
        return t.abs(pattern[pad0:-pad0,pad1:-pad1])**2 + eps # for numerical stability


def amplitude_mse(simulated, measured):
    """Normalized amplitude mean square error loss
    """
    return t.sum((t.sqrt(simulated+1e-8) - t.sqrt(measured))**2) / t.sum(measured)


def reconstruct(pattern, probe, resolution, lr, iterations, loss_func=amplitude_mse, background=None, optimizer='Adam', GPU=False, schedule=True):
    """Performs a full reconstruction
    
    because we use the probe here directly, without upsampling it, 
    it is critical to ensure that the probe has been padded sufficiently 
    in reciprocal space to avoid aliasing.
    """
    
    # We start from a uniform image as our initial guess
    im = t.from_numpy(np.ones((resolution, resolution), dtype=np.complex128))

    if GPU:
        im = im.to(device='cuda:0')
        
    # We separately optimize on the real and imaginary parts of the image
    im_real = t.nn.Parameter(t.real(im))
    im_imag = t.nn.Parameter(t.imag(im))

    if 'Adam'.lower() in optimizer.lower():
        t_optimizer = t.optim.Adam([im_real, im_imag], lr=lr)
    elif 'LBFGS'.lower() in optimizer.lower():
        t_optimizer = t.optim.LBFGS([im_real, im_imag], lr=lr, history_size=2, tolerance_grad=1e-11, tolerance_change=1e-11, max_iter=10)
    elif 'sgd'.lower() in optimizer.lower():
        t_optimizer = t.optim.SGD([im_real, im_imag], lr=lr, momentum=True)

    if schedule:
        scheduler = t.optim.lr_scheduler.StepLR(t_optimizer, step_size=100, gamma=0.5)

    
    # To avoid issues arising from the overall scale of the probe and object being mismatched,
    # we start by rescaling the probe such that the object should have an amplitude of around 1.

    reconstruction_norm = t.sum(t.abs(probe)**2)
    pattern_norm = t.sum(pattern)
    scaling_factor = pattern_norm / reconstruction_norm
    # There needs to be a correction for the object's size
    scaling_factor *= ((probe.shape[0] * probe.shape[1]) /
                       (im.shape[0] * im.shape[1]))
        
    # And of course we're talking amplitudes, not intensities
    scaling_factor = np.sqrt(scaling_factor)

    # Scale the probe so a uniform object of mangnitude 1 will lead to the correct detector intensity,
    # this would be our internal probe for the optimization process, later, we would rescale the output
    # object back to its correct scale using the same scaling factor
    probe = probe * scaling_factor

    if GPU:
        probe = probe.to(device='cuda:0')
        pattern = pattern.to(device='cuda:0')
        if background is not None:
            background = background.to(device='cuda:0')

    def closure():
        t_optimizer.zero_grad()
        im = t.complex(im_real, im_imag)
        exit_wave = interact(im, probe)
        simulated = measure(propagate(exit_wave), probe.shape)
        if background is not None:
            simulated = simulated + background
        
        l = loss_func(simulated, pattern)
        l.backward()
        return l

    loss = []
    for i in range(iterations):
        tensor = t_optimizer.step(closure)
        tensor = tensor.detach().cpu().numpy()
        loss.append(tensor[()])
        
        if schedule:
            scheduler.step()

    # The image/object is optimized with the internal probe that has been 
    # multiplied by the scaling factor, we need to scale the image by the 
    # same scaling factor back to match the given probe intensity, instead
    # of the internal probe intensity
    im = t.complex(im_real, im_imag)
    result = im.detach() * scaling_factor
    if GPU:
        result = result.to(device='cpu')


    return result, loss