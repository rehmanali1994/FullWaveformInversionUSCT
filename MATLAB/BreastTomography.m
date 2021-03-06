clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));
load('sim_breast.mat');

% Time Axis and Sampling Frequency
freq_bins_used = 1:10:(numel(f)+1)/2; % Frequency Bins Used
fused = f(freq_bins_used); % Frequencies Used
P_fused = P_f(freq_bins_used); % Frequency Bins in Pulse
dt = mean(diff(t)); fs = 1/dt; % Sampling Period [s] and Frequency [Hz]
[ff,tt] = meshgrid(fused,t); % Time-Frequency Grid
delays = exp(-1i*2*pi*ff.*tt)*dt; % Fourier Transform Grid

% Create Frequency-Domain Data for Tomography
recording_x_f = zeros(numel(xelem),numel(fused),numel(rotAngle));
for rot_idx = 1:numel(rotAngle)
    random_scaling = (1e3)*repmat(randn(1,numel(fused)) + ...
        1i*randn(1,numel(fused)), [numel(xelem),1]); % Random Scaling at Each View Angle
    recording_x_f(:,:,rot_idx) = random_scaling .* ...
        (recording_x_t(:,:,rot_idx)*delays);
end
clearvars -except recording_x_f xelem rotAngle array_separation fused P_fused

% Simulation Grid
Nzi = 201; % Number of Grid Points in Axial Dimension
Nxi = 256; % Number of Grid Points in Lateral Dimension
zi = linspace(-array_separation/2,array_separation/2,Nzi); % Axial Grid [m]
dxi = mean(diff(xelem)); % Lateral Grid Spacing [m]
xi = (-(Nxi-1)/2:(Nxi-1)/2)*dxi; % Lateral Grid [m]
[Xi, Zi] = meshgrid(xi, zi); % Create Complete Grid of Imaging Points
Ri = sqrt(Xi.^2 + Zi.^2); 

% Elements on Simulation Grid
[XELEM_GRID, XI] = meshgrid(xelem, xi);
[~,x_src_idx] = min(abs(XI-XELEM_GRID)); % Indices of Grid Points

% Anti-Aliasing Window
ord = 100; xmax = (max(abs(xi))+max(abs(xelem)))/2; 
aawin = 1./sqrt(1+(xi/(xmax)).^ord);

% Construct Source for Angular Spectrum Method
tx_x_f = zeros(numel(xi), numel(fused)); 
tx_x_f(x_src_idx, :) = repmat(P_fused, [numel(x_src_idx), 1]); 

% Create Grid of Points on Which Slowness Model is Iterated
lateral_span = numel(xelem)*mean(diff(xelem)); % Lateral XDCR Coverage
radial_span = sqrt(array_separation.^2 + lateral_span.^2); % Radial Span
Nx = 601; x = linspace(-radial_span/2, radial_span/2, Nx);
Nz = 601; z = linspace(-radial_span/2, radial_span/2, Nz);
[X, Z] = meshgrid(x, z); % Grid for Slowness Model
R = sqrt(X.^2 + Z.^2); 

% Scaling Conversion Factor from Simulation Grid to Reconstruction Grid
dx = mean(diff(x)); dz = mean(diff(z)); dzi = dxi;
grid_conv_factor = (dx/dxi) * (dz/dzi);

% Used Background to Form Initial Guess
[~,~,~,c_bkgnd] = soundSpeedPhantom();
slowness = ones(size(X))/c_bkgnd; % Initial Slowness Model

% (Nonlinear) Conjugate Gradient
Niter = 12; 
search_dir = zeros(Nz,Nx); % Conjugate Gradient Direction
gradient_img_prev = zeros(Nz,Nx); % Previous Gradient Image
for iter = 1:Niter

% Step 1: Backprojection to Create Gradient Image
figure;
gradient_img = zeros(size(slowness));
ddwf_x_z_f_rot = zeros(Nzi,Nxi,numel(fused),numel(rotAngle));
scaling = zeros(numel(rotAngle),numel(fused)); % Source Scaling for Transmission
for rot_idx = 1:numel(rotAngle)
    % Rotate Slowness Image to Compute Recorded Signals
    Ti = atan2(Zi, Xi) - rotAngle(rot_idx); 
    C = interp2(X, Z, 1./slowness, Ri.*cos(Ti), Ri.*sin(Ti), 'linear', c_bkgnd);
    % Compute Angular Spectrum Over All Points
    dwf_x_z_f = downward_continuation(xi, zi, C, fused, tx_x_f, ...
        zeros(numel(zi),numel(xi),numel(fused)), aawin);
    % Extract Forward Projected Data at Bottom
    forwardProject_x_f = squeeze(dwf_x_z_f(Nzi,x_src_idx,:));
    % Compute Scaling
    for f_idx = 1:numel(fused)
        REC = recording_x_f(:,f_idx,rot_idx);
        REC_SIM = forwardProject_x_f(:,f_idx); 
        scaling(rot_idx,f_idx) = (REC_SIM(:)'*REC(:))/(REC_SIM(:)'*REC_SIM(:));
        forwardProject_x_f(:,f_idx) = ...
            scaling(rot_idx,f_idx) .* forwardProject_x_f(:,f_idx);
        dwf_x_z_f(:,:,f_idx) = scaling(rot_idx,f_idx) .* dwf_x_z_f(:,:,f_idx);
    end
    % Compute Residual
    residual_x_f = forwardProject_x_f - recording_x_f(:,:,rot_idx);
    res_x_f = zeros(numel(xi), numel(fused));
    res_x_f(x_src_idx, :) = residual_x_f;
    % Compute Angular Spectrum Over All Points
    uwf_x_z_f = upward_continuation(xi, zi, C, fused, res_x_f, aawin);
    % Derivative of Downward Continuation
    ddwf_x_z_f = -1i*2*pi*mean(diff(zi)) * ...
        repmat(reshape(fused,[1,1,numel(fused)]),[Nzi,Nxi,1]) .* dwf_x_z_f;
    ddwf_x_z_f_rot(:,:,:,rot_idx) = ddwf_x_z_f;
    % Create Gradient Image for this View
    grad_img = real(sum(ddwf_x_z_f.* conj(uwf_x_z_f),3));
    % Accumulate the Gradient from this View Angle
    T = atan2(Z, X) + rotAngle(rot_idx); 
    backproj_img = interp2(Xi, Zi, grad_img, R.*cos(T), R.*sin(T), 'linear', 0);
    gradient_img = gradient_img + backproj_img;
    % Show the Accumulation of the Gradient as a Function of Angle
    REC = mean(recording_x_f(:,:,rot_idx), 2);
    REC_SIM = mean(forwardProject_x_f, 2); RESIDUAL = mean(residual_x_f, 2);
    clf; subplot(2,2,[1,2]); plot(xi(x_src_idx), squeeze(real(REC))); hold on;
    plot(xi(x_src_idx), squeeze(real(REC_SIM))); 
    plot(xi(x_src_idx), real(RESIDUAL)); 
    xlabel('Lateral [m]'); ylabel('Real Part of Signal');
    legend('True Signal', 'Projected Signal', 'Error')
    subplot(2,2,3); imagesc(x, z, backproj_img)
    xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
    title(['Backprojection at Angle = ', num2str(rotAngle(rot_idx))]); 
    colorbar; colormap gray;
    subplot(2,2,4); imagesc(x, z, gradient_img)
    xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
    title(['Gradient up to Angle = ', num2str(rotAngle(rot_idx))]); 
    colorbar; colormap gray; drawnow;
end

% Step 2: Compute New Conjugate Gradient Search Direction from Gradient
% Conjugate Gradient Direction Scaling Factor for Updates
nsteps_reset = 100; % Number of Steps After Which Search Direction is Reset
if rem(iter,nsteps_reset) == 1
    beta = 0; 
else 
    betaPR = (gradient_img(:)'*...
        (gradient_img(:)-gradient_img_prev(:))) / ...
        (gradient_img_prev(:)'*gradient_img_prev(:));
    betaFR = (gradient_img(:)'*gradient_img(:)) / ...
        (gradient_img_prev(:)'*gradient_img_prev(:));
    beta = min(max(betaPR,0),betaFR);
end
search_dir = beta*search_dir-gradient_img;
gradient_img_prev = gradient_img;

% Step 3: Compute Forward Projection of Current Search Direction
drecording_x_f = zeros(numel(xelem),numel(fused),numel(rotAngle));
for rot_idx = 1:numel(rotAngle)
    % Rotate Slowness Image to Compute Recorded Signals
    Ti = atan2(Zi, Xi) - rotAngle(rot_idx); 
    C = interp2(X, Z, 1./slowness, Ri.*cos(Ti), Ri.*sin(Ti), 'linear', c_bkgnd);
    SEARCH_DIR = interp2(X, Z, search_dir, Ri.*cos(Ti), Ri.*sin(Ti), 'linear', 0);
    % Forward Projection of Search Direction Image
    ddwf_x_z_f = downward_continuation(xi, zi, C, fused, ...
        zeros(numel(xi),numel(fused)), ddwf_x_z_f_rot(:,:,:,rot_idx).*...
        repmat(SEARCH_DIR,[1,1,numel(fused)]), aawin);
    drecording_x_f(:,:,rot_idx) = squeeze(ddwf_x_z_f(Nzi,x_src_idx,:));
end

% Step 4: Perform a Linear Approximation of Exact Line Search
perc_step_size = 1; % (<=1/2) Introduced to Improve Compliance with Strong Wolfe Conditions 
alpha = -(gradient_img(:)'*search_dir(:))/...
    (drecording_x_f(:)'*drecording_x_f(:));
slowness = slowness + perc_step_size * alpha * grid_conv_factor * search_dir;

% Compare Gradient Image to Ground Truth Sound Speed
[xg,zg,cg,~] = soundSpeedPhantom(); % Ground Truth Image
[XG, ZG] = meshgrid(xg,zg); 
cGNDTRUTH = interp2(XG,ZG,cg,X,Z,'linear',c_bkgnd);
figure; subplot(2,2,1); imagesc(x,z,cGNDTRUTH); 
xlabel('Lateral [m]'); ylabel('Axial [m]'); 
title('Ground-Truth SoS'); colorbar; axis image;
subplot(2,2,2); imagesc(x, z, 1./slowness)
xlabel('Lateral [m]'); ylabel('Axial [m]'); 
caxis([min(cg(:)),max(cg(:))]); colorbar; axis image;
title(['Reconstructed SoS Iteration ', num2str(iter)]); 
subplot(2,2,3); imagesc(x, z, -gradient_img)
xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
title(['Gradient Iteration ', num2str(iter)]); 
colorbar; colormap gray;
subplot(2,2,4); imagesc(x, z, search_dir)
xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
title(['Search Direction Iteration ', num2str(iter)]); 
colorbar; colormap gray;
drawnow;

end
