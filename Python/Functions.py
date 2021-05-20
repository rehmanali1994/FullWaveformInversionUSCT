import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interpn
import pdb

# Angular Spectrum Method Downward into the Medium from the Transmitters
def downward_continuation(x, z, c_x_z, f, tx_x_f, bckgnd_wf_x_z_f, aawin):

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(aawin*sig, axis=0), axes=0);
    ift = lambda sig: aawin*np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0);

    # Spatial Grid
    dx = np.mean(np.diff(x)); nx = x.size;
    x = dx*np.arange(-(nx-1)/2,(nx-1)/2+1);
    dz = np.mean(np.diff(z));

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(np.fft.fftshift(np.arange(nx)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

    # Convert to Slowness [s/m]
    s_x_z = 1/c_x_z; # Slowness = 1/(Speed of Sound)
    s_z = np.mean(s_x_z, axis=1); # Mean Slowness vs Depth (z)
    ds_x_z = s_x_z - np.tile(s_z[:,np.newaxis],(1,x.size)); # Slowness Deviation

    # Generate Wavefield at Each Frequency
    wf_x_z_f = np.zeros((z.size, x.size, f.size)).astype('complex64');
    for f_idx in np.arange(f.size):
        # Continuous Wave Response By Downward Angular Spectrum
        wf_x_z_f[0,:,f_idx] = tx_x_f[:,f_idx]; # Injection Surface (z = 0)
        for z_idx in np.arange(z.size-1):
            # Create Propagation Filter for this Depth
            kz = np.sqrt((f[f_idx]*s_z[z_idx])**2-kx**2); # Axial Spatial Frequency
            H = np.exp(-1j*2*np.pi*kz*dz); # Propagation Filter in Spatial Frequency Domain
            H[(f[f_idx]*s_z[z_idx])**2-kx**2 <= 0] = 0; # Remove Evanescent Components
            # Create Phase-Shift Correction in Spatial Domain
            dH = np.exp(-1j*2*np.pi*f[f_idx]*ds_x_z[z_idx,:]*dz);
            # Downward Continuation with Split-Stepping
            wf_x_z_f[z_idx+1, :, f_idx] = \
                bckgnd_wf_x_z_f[z_idx+1, :, f_idx] + \
                dH*ift(H*ft(wf_x_z_f[z_idx, :, f_idx]));
    return wf_x_z_f;

# Angular Spectrum Method Upwards into the Medium from the Receivers
def upward_continuation(x, z, c_x_z, f, tx_x_f, aawin):

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(aawin*sig, axis=0), axes=0);
    ift = lambda sig: aawin*np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0);

    # Spatial Grid
    dx = np.mean(np.diff(x)); nx = x.size;
    x = dx*np.arange(-(nx-1)/2,(nx-1)/2+1);
    dz = np.mean(np.diff(z));

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(np.fft.fftshift(np.arange(nx)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

    # Convert to Slowness [s/m]
    s_x_z = 1/c_x_z; # Slowness = 1/(Speed of Sound)
    s_z = np.mean(s_x_z, axis=1); # Mean Slowness vs Depth (z)
    ds_x_z = s_x_z - np.tile(s_z[:,np.newaxis],(1,x.size)); # Slowness Deviation

    # Generate Wavefield at Each Frequency
    wf_x_z_f = np.zeros((z.size, x.size, f.size)).astype('complex64');
    for f_idx in np.arange(f.size):
        # Continuous Wave Response By Downward Angular Spectrum
        wf_x_z_f[-1, :, f_idx] = tx_x_f[:,f_idx]; # Injection Surface (z = 0)
        for z_idx in np.arange(z.size-1,1,-1):
            # Create Propagation Filter for this Depth
            kz = np.sqrt((f[f_idx]*s_z[z_idx])**2-kx**2); # Axial Spatial Frequency
            H = np.exp(1j*2*np.pi*kz*dz); # Propagation Filter in Spatial Frequency Domain
            H[(f[f_idx]*s_z[z_idx])**2-kx**2 <= 0] = 0; # Remove Evanescent Components
            # Create Phase-Shift Correction in Spatial Domain
            dH = np.exp(1j*2*np.pi*f[f_idx]*ds_x_z[z_idx,:]*dz);
            # Downward Continuation with Split-Stepping
            wf_x_z_f[z_idx-1,:,f_idx] = ift(H*ft(dH*wf_x_z_f[z_idx,:,f_idx]));
    return wf_x_z_f;

def soundSpeedPhantom():
    '''SOUNDSPEEDPHANTOM Outputs Sound Speed Phantom
    x, z, c, c_bkgnd = soundSpeedPhantom()
    x, z -- x and z grid [m]
    c -- sound speed map on grid [m/s]
    c_bkgnd -- background sound speed [m/s]'''

    # Load Breast CT Image
    im2double = lambda img: (img.astype('double')/255);
    breastct = im2double(plt.imread('breast_ct.jpg'));

    # Normalize Breast CT Image
    breastct = breastct/np.max(breastct); thr = 0.04;
    breastct[breastct<=thr] = np.mean(breastct[breastct>=thr]);
    breastct = breastct - np.mean(breastct);
    breastct = breastct/np.max(np.abs(breastct));

    # Get Dimensions of Breast CT Image
    Nz, Nx = breastct.shape;
    dx = 0.0005; dz = dx; # Grid Spacing [m]
    x = np.arange(-(Nx-1)/2,(Nx-1)/2+1)*dx;
    z = np.arange(-(Nz-1)/2,(Nz-1)/2+1)*dz;

    # Create Sound Speed Image
    c_bkgnd = 1525; c_std = 60;
    c = c_bkgnd+c_std*breastct;

    return x, z, c, c_bkgnd;

# Python-Equivalent Command for IMAGESC in MATLAB
def imagesc(x, y, img, rng, cmap='gray', numticks=(3, 3), aspect='equal'):
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y)));
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0]));
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1]));
    plt.gca().invert_yaxis();
