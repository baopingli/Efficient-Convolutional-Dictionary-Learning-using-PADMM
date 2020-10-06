function [ d_res, z_res, Dz, obj_val, iterations ] = admm_learn_conv2D_weighted(b, kernel_size, ...
                    lambda_residual, lambda_prior, ...
                    max_it, tol, ...
                    verbose, init)

    data_psnr_p1=[];
    data_obj_D=[];
    data_obj_Z=[];
    data_time_Dpre=[];
    data_time_Dliter=[];
    data_time_Zpre=[];
    data_time_Zliter=[];
    psf_s = kernel_size(1);
    k = kernel_size(3);
    n = size(b,3);
                
    psf_radius = floor( psf_s/2 );
    size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, n];
    size_z = [size_x(1), size_x(2), k, n]; 
    size_k = [2*psf_radius + 1, 2*psf_radius + 1, k]; 
    size_k_full = [size_x(1), size_x(2), k]; 
    
    objective = @(z, dh) objectiveFunction( z, dh, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x );
    
   
    [M, Mtb] = precompute_MProx(b, psf_radius); 
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); 

    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;

    ProxKernelConstraint = @(u) KernelConstraintProj( u, size_k_full, psf_radius);
    
    %% Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
    gamma_heuristic = 60 * lambda_prior * 1/max(b(:));
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic]; 
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic]; 
    
    %% Initialize variables for K
    varsize_D = {size_x, size_k_full};
    xi_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    xi_D_hat = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
                                                 
    u_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    v_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    
    %Initial iterates
    if ~isempty(init)
        d = init.d;
    else
        d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');
        d = circshift(d, -[psf_radius, psf_radius, 0] );
    end
    d_hat = fft2(d);
    
    %% Initialize variables for Z
    varsize_Z = {size_x, size_z};
    xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    %Initial iterates
    if ~isempty(init)
        z = init.z;
    else
        z = randn(size_z);
    end
    z_hat = reshape(fft2(reshape(z, size_z(1), size_z(2), [])), size_z);
    
    %% Display it.
    if strcmp(verbose, 'all')
        iterate_fig = figure();
        filter_fig = figure();
        display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, 0);
        pause(0.1);
    end
    
    %Save all objective values and timings
    iterations.obj_vals_d = [];
    iterations.obj_vals_z = [];
    iterations.tim_vals = [];
    iterations.it_vals = [];
    
    %Initial vals
    obj_val = objective(z, d_hat);
    
    %Save all initial vars
    iterations.obj_vals_d(1) = obj_val;
    iterations.obj_vals_z(1) = obj_val;
    iterations.tim_vals(1) = 0;
    d_curr = circshift( d, [psf_radius, psf_radius,0] ); 
    d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1,:);
    iterations.it_vals = cat(4, iterations.it_vals, d_curr );
    
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
    end
    
    %Iteration for local back and forth
    max_it_d = 10;
    max_it_z = 10;
    
    obj_val_filter = obj_val;
    obj_val_z = obj_val;
    
    %Iterate
    for i = 1:max_it
        
        %% Update kernels
        %Timing
        tic;
        
        %Recompute what is necessary for kernel convterm later
        rho = gammas_D(2)/gammas_D(1);
        size(z_hat);
            
        [zhat_mat, zhat_inv_mat,newrho] = precompute_H_hat_D(z_hat, size_z, rho,gammas_D);
        obj_val_min = min(obj_val_filter, obj_val_z);
        
        d_old = d;
        d_hat_old = d_hat;
        d_D_old = d_D;
        obj_val_old = obj_val;
        
        %Timing
        t_kernel = toc;
        data_time_Dpre(end+1)=t_kernel;
        fprintf('D precompute time: %5.5g\n',t_kernel);
        
        for i_d = 1:max_it_d

            tic;
       
            d_hat_old_inD=d_hat;
            v_D{1} = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));                                                                                                                                                                                                                        
            v_D{2} = d;

            %Compute proximal updates 
            u_D{1} = ProxDataMasked( v_D{1} - d_D{1}, lambda(1)/gammas_D(1) );
            u_D{2} = ProxKernelConstraint( v_D{2} - d_D{2});

            for c = 1:2
                %Update running errors
                d_D{c} = d_D{c} - (v_D{c} - u_D{c});

                %Compute new xi and transform to fft
                xi_D{c} = u_D{c} + d_D{c};
                xi_D_hat{c} = fft2(xi_D{c});
                %1:DZ 2:D
            end

           
            z_hatdk_hat=reshape(sum(repmat(d_hat_old_inD,[1,1,1,n]) .* z_hat, 3), size_x);         
            z_hatdk_hat_mat=num2cell( permute( reshape(z_hatdk_hat, size_x(1) * size_x(2), n), [2,1] ), 1);%n
            newrhoDk=newrho*d_hat_old_inD;
            newrhoDk_mat=num2cell( permute( reshape(newrhoDk, size_x(1) * size_x(2), k), [2,1] ), 1);%k
            d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z, z_hatdk_hat_mat, newrhoDk_mat);
            d = real(ifft2( d_hat ));
            
            %Timing
            t_kernel_tmp = toc;
            t_kernel = t_kernel + t_kernel_tmp;
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                data_obj_D(end+1)=obj_val;
                data_time_Dliter(end+1)=t_kernel_tmp;
                fprintf('--> Obj %3.3g time %5.5g \n', obj_val,t_kernel_tmp )
            end
        end
        
        obj_val_filter_old = obj_val_old;
        obj_val_filter = obj_val;
        
        %Debug progress
        d_diff = d - d_old;
        d_comp = d;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            obj_val = objective(z, d_hat);
            fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i, obj_val, norm(d_diff(:),2)/ norm(d_comp(:),2))
        end
        
        %% Update sparsity term
        
        %Timing
        tic;
        
        %Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);
        dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] );
        %dhat_T
        z_old = z;
        z_hat_old = z_hat;
        %d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
        d_Z_old = d_Z;
        obj_val_old = obj_val;
        
        %Timing
        t_vars = toc;
        data_time_Zpre(end+1)=t_vars;
        fprintf('Z precompute time: %5.5g\n',t_vars);
        for i_z = 1:max_it_z
            
            %Timing
            tic;

            %Compute v_i = H_i * z
            v_Z{1} = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
            v_Z{2} = z;

            %Compute proximal updates
            u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
            u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );

            for c = 1:2
                %Update running errors
                d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});

                %Compute new xi and transform to fft
                xi_Z{c} = u_Z{c} + d_Z{c};
                xi_Z_hat{c} = reshape( fft2( reshape( xi_Z{c}, size_x(1), size_x(2), [] ) ), size(xi_Z{c}) );
            end

            %Solve convolutional inverse
            % z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z);
            z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2),[]) )), size_z );
            
            %Timing
            t_vars_tmp = toc;
            t_vars = t_vars + t_vars_tmp;
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                data_obj_Z(end+1)=obj_val;
                data_time_Zliter(end+1)=t_vars_tmp;
                fprintf('--> Obj %3.3g time %5.5g \n', obj_val, t_vars_tmp)
            end
        
        end
        
        obj_val_z_old = obj_val_old;
        obj_val_z = obj_val;
        
        if obj_val_min <= obj_val_filter && obj_val_min <= obj_val_z
            
            z_hat = z_hat_old;
            z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2),[]) )), size_z );
            
            d_hat = d_hat_old;
            d = real(ifft2( d_hat ));
            
            obj_val = objective(z, d_hat);
            break;
        end
        %}
        
        %Display it.
        if strcmp(verbose, 'all')
            p1=display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, i);
            data_psnr_p1(end+1)=p1;
            pause(0.1);
        end

        %Debug progress
        z_diff = z - z_old;
        z_comp = z;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('Iter Z %d, Obj %3.3g, Diff %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2))
        end
        
        %Save current iteration
        iterations.obj_vals_d(i + 1) = obj_val_filter;
        iterations.obj_vals_z(i + 1) = obj_val;
        iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_vars;
        d_curr = circshift( d, [psf_radius, psf_radius,0] ); 
        d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1,:);
        iterations.it_vals = cat(4, iterations.it_vals, d_curr );
    
        %iterations.obj_vals(1:i + 1)   
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol && norm(d_diff(:),2)/ norm(d_comp(:),2) < tol
            break;
        end
    end
    save ./results/data_psnr_p1.mat data_psnr_p1;
    save ./results/data_obj_D.mat data_obj_D;
    save ./results/data_obj_Z.mat data_obj_Z;
    save ./results/data_time_Dpre.mat data_time_Dpre;
    save ./results/data_time_Dliter.mat data_time_Dliter;
    save ./results/data_time_Zpre.mat data_time_Zpre;
    save ./results/data_time_Zliter.mat data_time_Zliter;
    %Final estimate
    z_res = z;
    
    d_res = circshift( d, [psf_radius, psf_radius, 0] ); 
    d_res = d_res(1:psf_radius*2+1, 1:psf_radius*2+1,:);
    
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
    
return;

function [u_proj] = KernelConstraintProj( u, size_k_full, psf_radius)
    
    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius,0] ); 
    u_proj = u_proj(1:psf_radius*2+1, 1:psf_radius*2+1,:);
    
    %Normalize
    u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1] );
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_k_full(1) - (2*psf_radius+1), size_k_full(2) - (2*psf_radius+1), 0], 0, 'post');
    u_proj = circshift(u_proj, -[psf_radius, psf_radius, 0] );

return;

function [M, Mtb] = precompute_MProx(b, psf_radius)
    
    M = padarray(ones(size(b)), [psf_radius, psf_radius, 0], 0, 'both');
    Mtb = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');
    
return;

function [zhat_mat, zhat_inv_mat,newrho] = precompute_H_hat_D(z_hat, size_z, rho,gammas_D)

sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [sy*sx, k, n] ), [3,2,1] ), [1 2] ), [1 sy*sx]); 
percent=0.001;
xx=max(svd(zhat_mat{1}));
for ii=1:round(percent*sx*sy)
    xx=max(xx,max(svd(zhat_mat{ii})));
end
for ii=floor((1-percent)*sx*sy):sx*sy
    xx=max(xx,max(svd(zhat_mat{ii})));
end
newrho=xx^2+0.01;
Imatrix=1/(rho+newrho);
zhat_inv_mat=cell(1,sy*sx);
zhat_inv_mat(:)={Imatrix};
return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
dhat_flat = reshape( dhat, size_x(1) * size_x(2), [] );
size(dhat_flat);
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);
size(dhatTdhat_flat);

return;

function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_hat, rho, size_z,z_hatdk_hat_mat,newrhoDk_mat)

    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    xi_hat_1_cell = num2cell( permute( reshape(xi_hat{1}, sx * sy, n), [2,1] ), 1);
    xi_hat_2_cell = num2cell( permute( reshape(xi_hat{2}, sx * sy, k), [2,1] ), 1);
    x = cellfun(@(Sinv, A, b, c, d, e)(Sinv * (A' * (b - d) + rho * c + e)), zhat_inv_mat, zhat_mat, xi_hat_1_cell, xi_hat_2_cell, z_hatdk_hat_mat, newrhoDk_mat, 'UniformOutput', false);
    d_hat = reshape( permute(cell2mat(x), [2,1]), [sy,sx,k] );

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z )
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    rho = gammas(2)/gammas(1);
    b = dhatT .* permute( repmat( reshape(xi_hat{1}, sy*sx, 1, n), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(xi_hat{2}, sy*sx, k, n), [2,1,3] );
    scInverse = repmat( ones([1,sx*sy]) ./ ( rho * ones([1,sx*sy]) + dhatTdhat.' ), [k,1,n] );
    x = 1/rho *b - 1/rho * scInverse .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    z_hat = reshape(permute(x, [2,1,3]), size_z);

return;

function f_val = objectiveFunction( z, d_hat, b, lambda_residual, lambda, psf_radius, size_z, size_x)
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* zhat, 3), size_x) ));
    
    f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    f_val = f_z + g_z;
    
return;

function [p1] = display_func(iterate_fig, filter_fig, d, d_hat, z_hat, b, size_x, size_z, psf_radius, iter)
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    figure(iterate_fig);
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
    Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);
    p1=psnr(b(:,:,1),Dz(:,:,1));
    p2=psnr(b(:,:,2),Dz(:,:,2));
    p3=psnr(b(:,:,3),Dz(:,:,3));
    subplot(3,2,1), imagesc(b(:,:,1));  axis image, colormap gray, title('Orig');
    subplot(3,2,2), imagesc(Dz(:,:,1)); axis image, colormap gray; title(sprintf('Local iterate %d,PSNR:%5.5g',iter,p1));
    subplot(3,2,3), imagesc(b(:,:,2));  axis image, colormap gray;  
    subplot(3,2,4), imagesc(Dz(:,:,2)); axis image, colormap gray; title(sprintf('PSNR:%5.5g',p2));
    subplot(3,2,5), imagesc(b(:,:,3));  axis image, colormap gray;
    subplot(3,2,6), imagesc(Dz(:,:,3)); axis image, colormap gray; title(sprintf('PSNR:%5.5g',p3));

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = circshift( d(:,:,j + 1), [psf_radius, psf_radius] ); 
        d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title(sprintf('Local filter iterate %d',iter));
    saveas(2,sprintf('./iterations_filters/iter%d.png',iter));
        
return;
