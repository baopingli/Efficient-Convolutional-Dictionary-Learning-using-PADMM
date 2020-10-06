clear;
close all;
verbose = 'all';  
addpath('./image_helpers');
CONTRAST_NORMALIZE = 'local_cn'; 
ZERO_MEAN = 1;   
COLOR_IMAGES = 'gray';                         
[b] = CreateImages('../datasets/Images/fruit_100_100/',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
b = reshape(b, size(b,1), size(b,2), [] ); 
kernel_size = [11, 11, 100];
lambda_residual = 1.0;
lambda =1;
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2) )
verbose_admm = 'all';
max_it = [13];
tol = 1e-4;
prefix = 'ours';
[ d, z, Dz, obj, iterations]  = admm_learn_conv2D_weighted(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, []);
if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
    figure();    
    pd = 1;
    sqr_k = ceil(sqrt(size(d,3)));
    d_disp = zeros( sqr_k * [kernel_size(1) + pd, kernel_size(2) + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_disp( floor(j/sqr_k) * (kernel_size(1) + pd) + pd + (1:kernel_size(1)) , mod(j,sqr_k) * (kernel_size(2) + pd) + pd + (1:kernel_size(2)) ) = d(:,:,j + 1); 
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title('Final filter estimate');
end
save(sprintf('./filters_%s_obj%3.3g.mat', prefix, obj), 'd', 'z', 'Dz', 'obj', 'iterations');
fprintf('Done sparse coding learning!')

