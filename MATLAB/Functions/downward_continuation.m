function wf_x_z_f = downward_continuation(x, z, c_x_z, f, tx_x_f, bckgnd_wf_x_z_f, aawin)

% Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
ft = @(sig) fftshift(fft(aawin.*sig));
ift = @(sig) aawin.*ifft(ifftshift(sig));

% Spatial Grid
dx = mean(diff(x)); nx = numel(x); 
x = dx*((-(nx-1)/2):((nx-1)/2)); 
dz = mean(diff(z)); 

% FFT Axis for Lateral Spatial Frequency
kx = mod(fftshift((0:nx-1)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

% Convert to Slowness [s/m]
s_x_z = 1./c_x_z; % Slowness = 1/(Speed of Sound)
s_z = mean(s_x_z, 2); % Mean Slowness vs Depth (z)
ds_x_z = s_x_z - repmat(s_z, [1, numel(x)]); % Slowness Deviation

% Generate Wavefield at Each Frequency
wf_x_z_f = zeros(numel(z), numel(x), numel(f));
for f_idx = 1:numel(f)
    % Continuous Wave Response By Downward Angular Spectrum
    wf_x_z_f(1, :, f_idx) = tx_x_f(:,f_idx); % Injection Surface (z = 0)
    for z_idx = 1:numel(z)-1
        % Create Propagation Filter for this Depth
        kz = sqrt((f(f_idx)*s_z(z_idx)).^2 - kx.^2); % Axial Spatial Frequency
        H = exp(-1i*2*pi*kz*dz); % Propagation Filter in Spatial Frequency Domain
        H((f(f_idx)*s_z(z_idx)).^2-kx.^2 <= 0) = 0; % Remove Evanescent Components
        % Create Phase-Shift Correction in Spatial Domain
        dH = exp(-1i*2*pi*f(f_idx)*ds_x_z(z_idx,:)*dz); 
        % Downward Continuation with Split-Stepping
        wf_x_z_f(z_idx+1, :, f_idx) = bckgnd_wf_x_z_f(z_idx+1, :, f_idx) + ...
            dH.*ift(H.*ft(squeeze(wf_x_z_f(z_idx, :, f_idx)))); 
    end
end

end

