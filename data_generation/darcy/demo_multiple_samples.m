%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(s-1)
s = 421;

%Create mesh (only needed for plotting)
[X,Y] = meshgrid(0:(1/(s-1)):1);

%Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
%Note that we need alpha > d/2 (here d= 2) 
%Laplacian has zero Neumann boundry
%alpha and tau control smoothness; the bigger they are, the smoother the
%function
alpha = 2;
tau = 3;
nsize = 1000;
coeff=zeros(nsize,s,s);
sol = zeros(nsize,s,s);

for t=1:nsize
    
    %Generate random coefficients from N(0,C)
    norm_a = GRF(alpha, tau, s,t+1240000);

    %Exponentiate it, so that a(x) > 0
    %Now a ~ Lognormal(0, C)
%This is done so that the PDE is elliptic
    lognorm_a = exp(norm_a);

%Another way to achieve ellipticity is to threshhold the coefficients
    thresh_a = zeros(s,s);
    thresh_a(norm_a >= 0) = 1;
    thresh_a(norm_a < 0) = 24;

%Forcing function, f(x) = 1 
    f = ones(s,s);

%Solve PDE: - div(a(x)*grad(p(x))) = f(x)
    lognorm_p = solve_gwf(lognorm_a,f);
    thresh_p = solve_gwf(thresh_a,f);
    coeff(t,:,:)= thresh_a;
    sol(t,:,:)= thresh_p;
    
end

save('output1_24_train_1000','coeff','sol')

%Plot coefficients and solutions
subplot(2,2,1)
surf(X,Y,lognorm_a); 
view(2); 
shading interp;
colorbar;
subplot(2,2,2)
surf(X,Y,lognorm_p); 
view(2); 
shading interp;
colorbar;
 subplot(2,2,3)
 surf(X,Y,thresh_a); 
 view(2); 
 shading interp;
 colorbar;
 subplot(2,2,4)
 surf(X,Y,thresh_p); 
 view(2); 
 shading interp;
 colorbar;
