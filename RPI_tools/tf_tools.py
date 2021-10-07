# tensorflow tools used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains tensorflow tools to generate stimulated diffraction patterns for randomized probe imaging."""

import tensorflow as tf
import numpy as np
import h5py

def generate_blr_probe(shape, band_limiting_radius,
                       beamstop_factor=0.5, dtype=tf.complex64, reseed=True):
    """This function generates a band-limited-random probe

    The shape defines the shape of the array on which the probe is defined,
    the focal radius defines the radius of the focal spot (in pixels),
    the band-limiting radius defines the highest frequency found in the
    focal spot, and the beamstop factor defines the size of the central
    region (in Fourier space) where all the frequency components are zero.

    This central region is not needed for the method, but is generically
    expected to exist in any real BLR probe at x-ray wavelengths due to the
    need for a beamstop in the diffractive optics which generate the probe.
    """

    if reseed:
        # set the seed to 0 so the probe is reproduceable for a given shape
        np.random.seed(0)
        
    Xs, Ys = np.mgrid[:shape[0],:shape[1]]
    Xs = Xs - np.mean(Xs)
    Ys = Ys - np.mean(Ys)
    Rs = np.sqrt(Xs**2 + Ys**2)
    focus = np.exp(2j*np.pi*np.random.rand(*Rs.shape))

    farfield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(focus)))

    farfield[Rs>band_limiting_radius] = 0
    farfield[Rs<int(beamstop_factor*band_limiting_radius)] = 0

    nearfield = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(farfield)))
    return tf.cast(tf.convert_to_tensor(nearfield),dtype)


def propagate(im):
    """Propagates a wavefield to the far-field"""
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im)))


def backpropagate(im):
    """Inverts far-field propagation"""
    return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(im)))


def expand_probe(probe, im_shape):
    """This upsamples a probe to be big enough in reciprocal space to avoid
    any aliasing when it is multiplied with an image of the given shape. 
    The assumption is that the image will be upsampled by padding in
    Fourier space such that it's dimensions match that of the probe, but it
    remains band-limited to a region in Fourier space the size of the original
    image."""
    
    fftprobe = propagate(probe)
    fftprobe = tf.pad(fftprobe, paddings=tf.constant([[im_shape[-2]//2,im_shape[-1]//2],[im_shape[-2]//2,im_shape[-1]//2]]))
    return backpropagate(fftprobe)


def interact(im, probe, known_padding=None):
    """Simulates the probe-sample interaction.

    This impliments the map from the original low-resolution object (im) to
    the exit wave leaving the sample - first, upsampling the object, then
    multiplying by the probe. It is important to be careful and ensure that
    the probe array is sampled finely enough to ensure that no aliasing
    occurs - this can be ensured by the expand_probe function

    Note: The shape of the tensor must be known at compile-time, otherwise
    the padding will fail!
    """
    
    # We start by calculating the padding we'll need in Fourier space
    # so we can upsample the object to match the probe
    pad0 = (probe.shape[-2] - im.shape[-2])//2
    pad1 = (probe.shape[-1] - im.shape[-1])//2
    # the extra padding ensure that padding will be applied to
    # the last and second last dimension to handle batch processing
    extra_padding = [(0,0)] * (len(im.shape) - 2)
    padding = extra_padding + [[pad0,pad0],[pad1,pad1]]

    # Then, we do the Fourier upsampling of the probe
    fftim = propagate(im)
    fftim = tf.pad(fftim, paddings=padding)
    upsampled_im = backpropagate(fftim)

    # And we can do the multiplicative interaction
    return upsampled_im * probe


def measure(pattern, eps=1e-8):
    """Generates intensities corresponding to a subset of the wavefield
    """
    return tf.abs(pattern)**2 + eps # for numerical stability


def generate_imagenet_phase_data(probe, n_photons_per_pix=1000, n_train=1000, n_test=100, chunk_size=100, data_folder_prefix = 'data/'):
    """Generate training data using greyscale images derieved from ImageNet.
    
    The generated images have amplitude of 1 and the images are encoded in the phase channel,
    with 255 (the maximum value contained in the images) corresponding to 2pi.
    """
    
    # First, we load the training and testing data
    # Note that the images are created as phase images with amplitude 1
    filename = data_folder_prefix + 'ImageNet/tr_gd_255_ImageNet.mat'
    with h5py.File(filename, "r") as f:
        train_ims = f['tr_gd'][:n_train]
        # This image is corrupted in the underlying data file. We replace it
        # with final image from the training set.
        if n_train > 3965:
            train_ims[3964] = f['tr_gd'][-1]
        train_ims = np.exp(1j * (train_ims / 255.0)).astype(np.complex64)
        
    filename = data_folder_prefix + 'ImageNet/test_gd_255_ImageNet.mat'
    with h5py.File(filename, "r") as f:
        test_ims = f['test_gd'][:n_test]
        test_ims = np.exp(1j * (test_ims / 255.0)).astype(np.complex64)
    
    # This creates a probe which is sampled finely enough to avoid aliasing
    # when interacted with the training images
    expanded_probe = expand_probe(probe, train_ims.shape)
    
    # Now, we generate diffraction patterns from all of the training images
    train_patterns = []
    for i in range(n_train // chunk_size + 1):
        chunk = train_ims[i*chunk_size:(i+1)*chunk_size]
        patterns = measure(propagate(interact(chunk,expanded_probe)))
        
        # This sets the patterns to the requested scale (in units of photons)
        # np.sum() function compute the total photons in the diffraction pattern
        # dividing that and multiple by the object resolution set it to 1 photon
        # per object pixel. Multiplying it with n_photons_per_pix set the photon
        # number to the correct scale for low photon imaging condition
        patterns = patterns / np.sum(patterns, axis=(-1,-2), keepdims=True)
        patterns = train_ims[0].size * patterns * n_photons_per_pix

        # Once the patterns are set at the appropriate scale, we add poisson noise to the data
        patterns = np.random.poisson(patterns).astype(np.int32)
        
        train_patterns.append(patterns)
    
    train_patterns = (np.concatenate(train_patterns,axis=0))

    # And we do the same for all the testing images
    test_patterns = []
    for i in range(n_test // chunk_size + 1):
        chunk = test_ims[i*chunk_size:(i+1)*chunk_size]
        patterns = measure(propagate(interact(chunk,expanded_probe)))
        
        # This sets the patterns to the requested scale (in units of photons)
        patterns = patterns / np.sum(patterns, axis=(-1,-2), keepdims=True)
        patterns = test_ims[0].size * patterns * n_photons_per_pix

        # Once the patterns are set at the appropriate scale, we add poisson noise to the data
        patterns = np.random.poisson(patterns).astype(np.int32)
        
        test_patterns.append(patterns)

    test_patterns = (np.concatenate(test_patterns,axis=0))

    return expanded_probe, (train_patterns, train_ims), (test_patterns, test_ims)
