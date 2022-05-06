# Importing stuff from all folders in python path
from Functions import *
import scipy.io as sio
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import sys, os, pdb

# Load Simulated Dataset
data_in = sio.loadmat('sim_breast.mat');
P_f = data_in['P_f'][0];
f = data_in['f'][0];
t = data_in['t'][0];
xelem = data_in['xelem'][0];
rotAngle = data_in['rotAngle'][0];
recording_x_t = data_in['recording_x_t'];
array_separation = data_in['array_separation'][0,0];
del data_in;

# Time Axis and Sampling Frequency
freq_bins_used = np.arange(0,(f.size+1)/2,10); # Frequency Bins Used
fused = f[freq_bins_used]; # Frequencies Used
P_fused = P_f[freq_bins_used]; # Frequency Bins in Pulse
dt = np.mean(np.diff(t)); fs = 1/dt; # Sampling Period [s] and Frequency [Hz]
ff,tt = np.meshgrid(fused,t); # Time-Frequency Grid
delays = np.exp(-1j*2*np.pi*ff*tt)*dt; # Fourier Transform Grid
del freq_bins_used; del f; del P_f; del dt; del fs; del ff; del t; del tt;

# Create Frequency-Domain Data for Tomography
recording_x_f = np.zeros((xelem.size,fused.size,rotAngle.size)).astype('complex64');
for rot_idx in np.arange(rotAngle.size):
    random_scaling = (1e3)*np.tile(np.random.randn(1,fused.size) + 
        1j*np.random.randn(1,fused.size), (xelem.size,1)); # Random Scaling at Each View Angle
    recording_x_f[:,:,rot_idx] = np.dot(recording_x_t[:,:,rot_idx],delays);
del recording_x_t;

# Simulation Grid
Nzi = 201; # Number of Grid Points in Axial Dimension
Nxi = 256; # Number of Grid Points in Lateral Dimension
zi = np.linspace(-array_separation/2,array_separation/2,Nzi); # Axial Grid [m]
dxi = np.mean(np.diff(xelem)); # Lateral Grid Spacing [m]
xi = np.arange(-(Nxi-1)/2,(Nxi-1)/2+1)*dxi; # Lateral Grid [m]
Xi, Zi = np.meshgrid(xi, zi); # Create Complete Grid of Imaging Points
Ri = np.sqrt(Xi**2+Zi**2);

# Elements on Simulation Grid
XELEM_GRID, XI = np.meshgrid(xelem, xi);
x_src_idx = np.argmin(np.abs(XI-XELEM_GRID),axis=0); # Indices of Grid Points

# Anti-Aliasing Window
ord = 100; xmax = (np.max(np.abs(xi))+np.max(np.abs(xelem)))/2;
aawin = 1./np.sqrt(1+(xi/xmax)**ord);

# Construct Source for Angular Spectrum Method
tx_x_f = np.zeros((xi.size, fused.size)).astype('complex64');
tx_x_f[x_src_idx,:] = np.tile(P_fused[np.newaxis,:],[x_src_idx.size,1]);

# Create Grid of Points on Which Slowness Model is Iterated
lateral_span = xelem.size*np.mean(np.diff(xelem)); # Lateral XDCR Coverage
radial_span = np.sqrt(array_separation**2+lateral_span**2); # Radial Span
Nx = 601; x = np.linspace(-radial_span/2, radial_span/2, Nx);
Nz = 601; z = np.linspace(-radial_span/2, radial_span/2, Nz);
[X, Z] = np.meshgrid(x, z); # Grid for Slowness Model
R = np.sqrt(X**2 + Z**2);

# Scaling Conversion Factor from Simulation Grid to Reconstruction Grid
dx = np.mean(np.diff(x)); dz = np.mean(np.diff(z)); dzi = dxi;
grid_conv_factor = (dx/dxi) * (dz/dzi);

# Used Background to Form Initial Guess
_, _, _, c_bkgnd = soundSpeedPhantom();
slowness = np.ones(X.shape)/c_bkgnd; # Initial Slowness Model

# (Nonlinear) Conjugate Gradient
Niter = 12; plt.figure(figsize=(16,6)); tpause = 1e-9;
search_dir = np.zeros((Nz,Nx)); # Conjugate Gradient Direction
gradient_img_prev = np.zeros((Nz,Nx)); # Previous Gradient Image
for iter in np.arange(Niter):
    # Step 1: Backprojection to Create Gradient Image
    gradient_img = np.zeros(slowness.shape);
    scaling = np.zeros((rotAngle.size,fused.size)).astype('complex64'); # Source Scaling for Transmission
    ddwf_x_z_f_rot = np.zeros((Nzi,Nxi,fused.size,rotAngle.size)).astype('complex64');
    for rot_idx in np.arange(rotAngle.size):
        # Rotate Slowness Image to Compute Recorded Signals
        Ti = np.arctan2(Zi,Xi)-rotAngle[rot_idx];
        C = interpn((z,x), 1/slowness, (Ri*np.sin(Ti), Ri*np.cos(Ti)), \
            method='linear', bounds_error = False, fill_value=c_bkgnd);
        # Compute Angular Spectrum Over All Points
        dwf_x_z_f = downward_continuation(xi, zi, C, fused, tx_x_f, \
            np.zeros((zi.size,xi.size,fused.size)), aawin);
        # Extract Forward Projected Data at Bottom
        forwardProject_x_f = dwf_x_z_f[-1,x_src_idx,:];
        # Compute Scaling
        for f_idx in np.arange(fused.size):
            REC = recording_x_f[:,f_idx,rot_idx];
            REC_SIM = forwardProject_x_f[:,f_idx]; 
            scaling[rot_idx,f_idx] = np.inner(np.conj(REC_SIM),REC)/np.inner(np.conj(REC_SIM),REC_SIM);
            forwardProject_x_f[:,f_idx] = scaling[rot_idx,f_idx] * forwardProject_x_f[:,f_idx];
            dwf_x_z_f[:,:,f_idx] = scaling[rot_idx,f_idx] * dwf_x_z_f[:,:,f_idx];
        # Compute Residual
        residual_x_f = forwardProject_x_f-recording_x_f[:,:,rot_idx];
        res_x_f = np.zeros((xi.size, fused.size)).astype('complex64');
        res_x_f[x_src_idx, :] = residual_x_f;
        # Compute Angular Spectrum Over All Points
        uwf_x_z_f = upward_continuation(xi, zi, C, fused, res_x_f, aawin);
        # Derivative of Downward Continuation
        ddwf_x_z_f = -1j*2*np.pi*np.mean(np.diff(zi)) * \
            np.tile(np.reshape(fused,(1,1,fused.size)),(Nzi,Nxi,1)) * dwf_x_z_f;
        ddwf_x_z_f_rot[:,:,:,rot_idx] = ddwf_x_z_f;
        # Create Gradient Image for this View
        grad_img = np.real(np.sum(ddwf_x_z_f*np.conj(uwf_x_z_f),axis=2));
        # Accumulate the Gradient from this View Angle
        T = np.arctan2(Z,X)+rotAngle[rot_idx];
        gradient_img = gradient_img + interpn((zi,xi), grad_img, \
            (R*np.sin(T), R*np.cos(T)), method='linear', \
            bounds_error = False, fill_value=0);

    # Step 2: Compute New Conjugate Gradient Search Direction from Gradient
    # Conjugate Gradient Direction Scaling Factor for Updates
    nsteps_reset = 100; # Number of Steps After Which Search Direction is Reset
    if (iter % nsteps_reset) == 0:
        beta = 0;
    else:
        betaPR = np.dot(gradient_img.flatten(), \
            gradient_img.flatten()-gradient_img_prev.flatten()) / \
            np.dot(gradient_img_prev.flatten(),gradient_img_prev.flatten());
        betaFR = np.dot(gradient_img.flatten(),gradient_img.flatten()) / \
            np.dot(gradient_img_prev.flatten(),gradient_img_prev.flatten());
        beta = np.min((np.max((betaPR,0)),betaFR));
    search_dir = beta*search_dir-gradient_img;
    gradient_img_prev = gradient_img;

    # Step 3: Compute Forward Projection of Current Search Direction
    drecording_x_f = np.zeros((xelem.size,fused.size,rotAngle.size)).astype('complex64');
    for rot_idx in np.arange(rotAngle.size):
        # Rotate Slowness Image to Compute Recorded Signals
        Ti = np.arctan2(Zi,Xi) - rotAngle[rot_idx];
        C = interpn((z,x), 1/slowness, (Ri*np.sin(Ti), Ri*np.cos(Ti)), \
            method='linear', bounds_error = False, fill_value=c_bkgnd);
        SEARCH_DIR = interpn((z,x), search_dir, (Ri*np.sin(Ti), Ri*np.cos(Ti)), \
            method='linear', bounds_error = False, fill_value=c_bkgnd);
        # Forward Projection of Search Direction Image
        ddwf_x_z_f = downward_continuation(xi, zi, C, fused, \
            np.zeros((xi.size,fused.size)), ddwf_x_z_f_rot[:,:,:,rot_idx] * \
            np.tile(SEARCH_DIR[:,:,np.newaxis],(1,1,fused.size)), aawin);
        drecording_x_f[:,:,rot_idx] = ddwf_x_z_f[-1,x_src_idx,:];

    # Step 4: Perform a Linear Approximation of Exact Line Search
    perc_step_size = 1; # (<1/2) Introduced to Improve Compliance with Strong Wolfe Conditions
    alpha = -np.dot(gradient_img.flatten(),search_dir.flatten()) / \
        np.dot(drecording_x_f.flatten(),np.conj(drecording_x_f.flatten()));
    slowness = slowness + perc_step_size * np.real(alpha) * grid_conv_factor * search_dir;

    # Compare Gradient Image to Ground Truth Sound Speed
    xg, zg, cg, _ = soundSpeedPhantom(); # Ground Truth Image
    cGNDTRUTH = interpn((zg,xg), cg, (Z,X), method='linear', \
        bounds_error = False, fill_value=c_bkgnd);
    plt.subplot(1,2,2); imagesc(x,z,cGNDTRUTH,(np.min(cg),np.max(cg)));
    plt.xlabel('Lateral [m]'); plt.ylabel('Axial [m]');
    plt.title('Ground-Truth SoS'); plt.colorbar();
    plt.subplot(1,2,1); imagesc(x,z,1/slowness,(np.min(cg),np.max(cg)))
    plt.xlabel('Lateral [m]'); plt.ylabel('Axial [m]');
    plt.title('Reconstructed SoS Iteration '+str(iter)); plt.colorbar();
    plt.draw(); plt.pause(tpause); plt.clf(); # Animate
