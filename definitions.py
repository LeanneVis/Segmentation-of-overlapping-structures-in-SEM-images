import torch
import torch.nn.functional as F
from math import (
    sqrt,
    factorial
)
from functools import lru_cache


##DEFINITIONS FOR POLAR COORDINATES
def modulo(x, d, offset=0.):
    return x - (x - offset)//d * d

#Note that this is for image sizes where lenght=width
def radial_coordinate(im_size): 
    # ys, xs = torch.meshgrid(torch.arange(im_size), torch.arange(im_size), indexing="ij")
    mod = modulo((im_size-1)/2, 1)
    ys = torch.arange(-((im_size-1)/2) - mod, (im_size-1)/2 - mod + 1)[:, None]
    xs = torch.arange(-((im_size-1)/2) - mod, (im_size-1)/2 - mod + 1)[None, :]
    radialcoordinates = torch.sqrt(ys**2 + xs**2)
    return radialcoordinates

#Note that this can only be used for image sizes where lenght=width
def angular_coordinate(im_size): 
    # ys, xs = torch.meshgrid(torch.arange(im_size), torch.arange(im_size), indexing="ij")
    mod = modulo((im_size - 1)/2, 1)
    ys = torch.arange(-((im_size-1)/2) - mod, (im_size-1)/2 - mod + 1)[:, None]
    xs = torch.arange(-((im_size-1)/2) - mod, (im_size-1)/2 - mod + 1)[None, :]
    angularcoordinates = torch.atan2(-xs, ys)
    return angularcoordinates

def polar_coordinate(im_size):
    mod = modulo((im_size - 1)/2, 1)
    ys = torch.arange(-((im_size-1)/2) - mod ,(im_size-1)/2 - mod + 1)[:, None]
    xs = torch.arange(-((im_size-1)/2) - mod ,(im_size-1)/2 - mod + 1)[None, :]
    angularcoordinates = torch.atan2(ys,xs)
    radialcoordinates = torch.sqrt(ys**2+xs**2)
    return angularcoordinates, radialcoordinates

## GAUSSIAN WINDOW
def g(size, sigma, device="cpu"):
    grid =  torch.arange(-size, size+ 1, device=device)
    blur = torch.exp(-(grid**2) / (2 * sigma**2))
    return blur / (blur.sum() + 1e-12)

## ORIENTATION SCORE TRANSFORM
def radial_window(im_size, rho_inflection, n):
    mod = modulo((im_size-1)/2, 1)
    rho = (1/sqrt((2*(rho_inflection*((im_size-1)/2-mod)**2))/(1+2*n))) * radial_coordinate(im_size)
    ks = torch.arange(n+1)
    factorial_ks = torch.tensor([factorial(k) for k in ks]) 
    rho_pow = rho[:, :, None] ** (2*ks)
    l = torch.exp(-rho**2)[:, :, None] * rho_pow / factorial_ks
    window = torch.sum(l, dim=-1)
    return window

#The code for b-splines is written by Finn Sherry f.m.sherry@tue.nl
def B_spline(n, x):
    """
    Compute degree `n` B-splines.

    In this way, the sum of all cakewavelets in the Fourier domain is
    identically equal to 1 (within the disk M), while each cakewavelet varies
    smoothly in the angular direction in the Fourier domain. See Section 4.6
    in Duits "Perceptual Organization in Image Analysis" (2005).

    For degree `n` <= 3, we use explicit formulae. For higher orders, a
    recursive algorithm is used.
    """
    # if not isinstance(n, int) or n < 0:
    #     raise ValueError("n must be a positive integer")
    match n:
        case 0:
            b = 1 * (-1/2 <= x) * (x < 1/2)
        case 1:
            b = (
                (1. + x) * (-1. <= x) * (x < 0.) +
                (1. - x) * (0. <= x)  * (x < 1.)
            )
        case 2:
            b = (
                ((3/2 + x)**2)/2 * (-3/2 <= x) * (x < -1/2) +
                (3/4 - x**2)     * (-1/2 <= x) * (x < 1/2) +
                ((3/2 - x)**2)/2 * (1/2 <= x)  * (x < 3/2)
            )
        case 3:
            b = (
                ((2. + x)**3)/6         * (-2. <= x) * (x < -1.) + 
                (2/3 - x**2 - (x**3)/2) * (-1. <= x) * (x < -0.) + 
                (2/3 - x**2 + (x**3)/2) * (0. <= x)  * (x < 1.)  + 
                ((2. - x)**3)/6         * (1. <= x)  * (x < 2.)
            )
        case _:
            raise ValueError("n > 3 not implemented")
    return b

def gaussian_kernel_2D(σs, device="cpu"):
    ns = 3 #for images of size 200.  3 * ceil(σs) 
       
    gx, gy = g(ns, σs, device=device), g(ns, σs, device=device)
    kernel = gx[None, None, :] * gy[None, :, None] # H, W
    return kernel

def convolution_gauss_2D(φ, σs): # φ is stack of cake-wavelets and has shape (θ, y, x). We need to copy the gaussian blur for each orientation (so θ-times)
    ns = 3 # for images of size 200. If we don't care about differentiability: 3 * ceil(σs)    
    φ_pad = F.pad(φ, (ns, ns, ns, ns), mode='replicate') # Or, H + (kernel_size-1), W + (kernel_size-1)
    kernel = gaussian_kernel_2D(σs).repeat(φ.shape[-3], 1, 1, 1).to(φ.device) # Or, C=1, kernel_size, kernel_size
    blurred_image = F.conv2d(φ_pad, kernel, groups=φ.shape[-3]) # Or, H, W
    return blurred_image

@lru_cache(maxsize=None)
def cakewavelet_stack_fourier(im_size, No, rho_inflection, n=8):
    dθ = 2. * torch.pi / No
    ks = torch.arange(No)[:, None, None]
    
    angular_coordinates = modulo(angular_coordinate(im_size) - ks * dθ, 2 * torch.pi, offset=-torch.pi)
    mod = modulo(im_size/2, 1)
    centre = int(im_size/2 - mod)

    # Shift the B-spline with 1, then they sum up to 1
    cakewavel_stack_fourier = radial_window(im_size, rho_inflection, n) * B_spline(3, angular_coordinates / dθ) # Or, H, W  
    cakewavel_stack_fourier[:, centre, centre] = torch.mean(cakewavel_stack_fourier[:, centre, centre])
    cakewavel_stack_fourier_blurred = convolution_gauss_2D(cakewavel_stack_fourier, im_size /(2 * torch.pi * ((im_size-1)/4))) # Or, H, W 
    return cakewavel_stack_fourier_blurred

def orientation_score_transform(f, No, rho_inflection):
    im_size = f.shape[-1] # W of image f
    
    cws = cakewavelet_stack_fourier(im_size, No, rho_inflection).to(f.device) # Or, H, W

    f_fourier = torch.fft.fft2(f) # f has shape B, H, W
    # f_fourier has shape B, 1, H, W, and cws has shape 1, Or, H, W
    orientationscore = torch.fft.ifft2(f_fourier[:, None] * torch.fft.ifftshift(cws)[None]) # shape B, Or, H, W
    
    return orientationscore  #B, Or, H, W

##LINE FILTER USING GAUSSIAN DERIVATIVES
def gaussian_kernel_isotropic_x(σs , device="cpu"):
    ns = 25 #3 * ceil(σs) 
       
    gx = g(ns, σs, device=device)
    kernel = gx[None, None, :] 
    return kernel

def gaussian_kernel_isotropic_y(σs , device="cpu"):
    ns = 25 #3 * ceil(σs) 
        
    gy = g(ns, σs, device=device)
    kernel =  gy[None, :, None]
    return kernel

def gaussian_kernel_isotropic_θ(σa, device="cpu"):    
    na = 8 #3 * ceil(σa) 
    
    gθ = g(na, σa, device=device)
    kernel = gθ[:, None, None]
    return kernel

def convolution_gauss(U, σs , σa): # U has shape B, Or, H, W
    device = U.device
    ns = 25 #3 * ceil(σs) 
    na = 8 #3 * ceil(σa) 
    U_pad = F.pad(U[:, None], (ns, ns, ns, ns, 0, 0), mode='replicate') # B, C=1, Or, H, W
    U_pad = F.pad(U_pad, (0, 0, 0, 0, na, na), mode='circular') # B, C=1, Or, H, W
    kernel_x = gaussian_kernel_isotropic_x(σs ,  device=device)[None, None] # B, C=1, 1, 1, kernel_size_x
    kernel_y = gaussian_kernel_isotropic_y(σs ,  device=device)[None, None] # B, C=1, 1, kernel_size_y , 1
    kernel_θ = gaussian_kernel_isotropic_θ(σa, device=device)[None, None] # B, C=1, kernel_size_θ , 1, 1
    
    blurred_os_x = F.conv3d(U_pad, kernel_x)
    blurred_os_xy = F.conv3d(blurred_os_x, kernel_y)
    blurred_os_xyθ = F.conv3d(blurred_os_xy, kernel_θ)[:, 0]
    
    return blurred_os_xyθ # B, Or, H, W (Channel is removed!!)

def left_invariant_derivative(U):
    device = U.device
    dx = torch.gradient(U, dim=-1)[0]
    dy = torch.gradient(U, dim=-2)[0]
    
    dxx = torch.gradient(dx, dim=-1)[0]
    dyy = torch.gradient(dy, dim=-2)[0]
    dxy = torch.gradient(dx, dim=-2)[0]    
    
    θs = torch.arange(U.shape[-3], device=device) 
    θ = θs[:, None, None] * 2 * torch.pi / U.shape[-3]
        
    A11 = torch.cos(θ)**2 * dxx + 2 * torch.cos(θ) * torch.sin(θ) * dxy + torch.sin(θ)**2 * dyy
    A22 = torch.sin(θ)**2 * dxx - 2 * torch.cos(θ) * torch.sin(θ) * dxy + torch.cos(θ)**2 * dyy

    return A11, A22

def line_filter(U, σs , σa):
    U_blurred = convolution_gauss(U, σs , σa) #U has shape B, Or, H, W
    λ1, λ2 =  left_invariant_derivative(U_blurred)
    S = torch.sqrt(λ1**2 + λ2**2) #structure strength
    
    eps = 1e-3 
    R = λ1 / (torch.abs(λ2) + eps) * torch.sign(λ2) #line-ness criterion
    Q = λ2 #convexity criterion
    σ1 = 0.25* torch.max(torch.abs(R)) #0.5
    σ2 = 0.25* torch.max(torch.abs(S))
    
    lineness = torch.exp(-R**2 / (2 * σ1**2)) * (1 - torch.exp(-S**2 / (2 * σ2**2 + eps))) * (Q < 0)    
    
    return lineness  #B, Or, H, W

def cost_function (U, σs, σa, λ, p):
    cost = 1/(1 + λ * line_filter(U, σs , σa)**p)    
    return cost #B, Or, H, W

## NORMALIZE IMAGES
def adaptive_sigmoid_norm(I, scale, newMin=0.0, newMax=1.0): # I has shape B, Or, H, W
        
    # Compute beta = mean over H,W per (B,Or)
    beta = I.mean(dim=(2, 3), keepdim=True)  # (B, Or, 1, 1)

    eps = 1e-3
    # Compute alpha = std over H,W per (B,Or)
    alpha = I.std(dim=(2, 3), keepdim=True) / (scale + eps)
        
    # Apply the sigmoid mapping
    S = 1.0 / (1.0 + torch.exp(-(I - beta) / (alpha + eps)))
    
    return (newMax - newMin) * S + newMin

#BLOB FILTER
def gaussian_kernel_isotropic_x(σs , device="cpu"):
    ns = 25 #3 * ceil(σs) 
       
    gx = g(ns, σs, device=device)
    kernel = gx[None, None, :] 
    return kernel

def gaussian_kernel_isotropic_y(σs , device="cpu"):
    ns = 25 #3 * ceil(σs) 
        
    gy = g(ns, σs, device=device)
    kernel =  gy[None, :, None]
    return kernel

def gaussian_kernel_isotropic_θ(σa, device="cpu"):    
    na = 8 #3 * ceil(σa) 
    
    gθ = g(na, σa, device=device)
    kernel = gθ[:, None, None]
    return kernel

def convolution_gauss_blob(U, σs, device="cpu"): # U has shape B, Or, H, W
    device = U.device
    ns = 25 #3 * ceil(σs) 
    
    f = torch.sum(U, dim = -3, keepdim=True) # f has shape B, 1, H, W

    f_pad = F.pad(f[:, None], (ns, ns, ns, ns, 0, 0), mode='replicate') # B, C=1, 1, H, W
    
    kernel_x = gaussian_kernel_isotropic_x(σs ,  device=device)[None, None] # B, C=1, 1, 1, kernel_size_x
    kernel_y = gaussian_kernel_isotropic_y(σs ,  device=device)[None, None] # B, C=1, 1, kernel_size_y , 1
    
    
    blurred_os_x = F.conv3d(f_pad, kernel_x)
    blurred_os_xy = F.conv3d(blurred_os_x, kernel_y)[:, 0]
    
    
    return blurred_os_xy # B, 1, H, W (Channel is removed!!)

def left_invariant_derivative_blob(f):
    
    dx = torch.gradient(f, dim=-1)[0]
    dy = torch.gradient(f, dim=-2)[0]
    
    dxx = torch.gradient(dx, dim=-1)[0]
    dyy = torch.gradient(dy, dim=-2)[0]
    dxy = torch.gradient(dx, dim=-2)[0]    
    
    return dxx, dyy, dxy

def blob_filter(f, σs):
    device = f.device
    f_blurred = convolution_gauss_blob(f, σs, device=device) #f has shape B, 1, H, W
    dxx, dyy, dxy =  left_invariant_derivative_blob(f_blurred)

    det = (dxx * dyy - (dxy)**2)
    Q1 = (det > 0)
    Q2 = (dxx < 0)
    σ = 0.25 * torch.max(torch.abs(det) * Q1 * Q2)
       
    return  ((1. - torch.exp(-det**2 / (2 * σ**2))) * Q1 * Q2)  #B, Or, H, W (det * Q1 * Q2) +    * Q2

