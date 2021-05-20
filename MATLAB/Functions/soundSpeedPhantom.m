function [x, z, c, c_bkgnd] = soundSpeedPhantom()
%SOUNDSPEEDPHANTOM Outputs Sound Speed Phantom
% [x, z, c, c_bkgnd] = soundSpeedPhantom()
%   x, z -- x and z grid [m]
%   c -- sound speed map on grid [m/s]
%   c_bkgnd -- background sound speed [m/s]

% Load Breast CT Image
breastct = im2double(imread('breast_ct.jpg'));

% Normalize Breast CT Image
breastct = breastct/max(breastct(:)); thr = 0.04;
breastct(breastct<=thr) = mean(breastct(breastct>=thr));
breastct = breastct - mean(breastct(:));
breastct = breastct/max(abs(breastct(:)));

% Get Dimensions of Breast CT Image
[Nz, Nx] = size(breastct); 
dx = 0.0005; dz = dx; % Grid Spacing [m]
x = ((-(Nx-1)/2):((Nx-1)/2))*dx; 
z = ((-(Nz-1)/2):((Nz-1)/2))*dz;

% Create Sound Speed Image
c_bkgnd = 1525; c_std = 60;
c = c_bkgnd+c_std*breastct;

end