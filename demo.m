%% Bayesian Connectomics DEMO

addpath data\;
addpath functional\;
addpath fusion\;
addpath ggm\;
addpath mex\;
addpath structural\;
addpath utility\;

clear all;
load demodata.mat;

%% estimate P(G,K|X)
% Posterior of conditional independence and partial correlations

[n,p] = size(subcort);
Z = zscore(subcort);
S = Z'*Z;

nsamples = 100;

% fast MEX implementation:

G = eye(p);
tic;
[Gsamples, Ksamples] = ggm_cbf_mex(G,S,n,nsamples);
toc;

figure; imagesc(mean(Gsamples,3)); axis square; colormap hot; caxis([0 1]); % posterior expectation of dependencies
figure; imagesc(prec2parcor(mean(Ksamples,3))); axis square; colormap jet; caxis([-1 1]); % posterior expectation of partial correlations

[pr,~] = sample_dist(Gsamples);
figure; bar(sort(pr)); % distribution of model probabilities

% slow matlab implementation:

G = eye(p); % initial sample
Gsamples = zeros(p,p,nsamples); % conditional independencies
Ksamples = zeros(p,p,nsamples); % precision matrices
Rsamples = zeros(p,p,nsamples); % partial correlations

tic;
for i=1:nsamples
    [G,K] = ggm_gwish_cbf_direct(G,S,n,@gwishrnd3);
    Gsamples(:,:,i) = G;
    Ksamples(:,:,i) = K;
    Rsamples(:,:,i) = prec2parcor(K);
end
toc;

figure; imagesc(mean(Gsamples,3)); axis square; colormap hot; caxis([0 1]); % posterior expectation of dependencies
figure; imagesc(mean(Rsamples,3)); axis square; colormap jet; caxis([-1 1]); % posterior expectation of partial correlations

%% estimate structural connectivity, full distribution P(G|N)

p = length(N);

nsamples = 500;

% fast MEX implementation under construction:

a1 = 1.0; a0 = 0.1; a = 1; b = 1;

tic; 
Gsamples_mex = struct_conn_density_prior_mex(zeros(p), N, a1, a0, a, b, nsamples); 
toc;

figure;
imagesc(mean(Gsamples_mex,3)); colormap hot; axis square; colorbar; title(sprintf('MEX, d1=%0.2f, d0=%0.2f', a1, a0)); % posterior expectation of edge probability

% slow matlab implementation:

Gsamples_mlb = zeros(p,p,nsamples);
G = zeros(p);

tic;
for i=1:nsamples
    G = struct_conn_density_prior(G,N);
    Gsamples_mlb(:,:,i) = G;
end
toc;

figure;
imagesc(mean(Gsamples_mlb,3) + eye(p)); colormap hot; axis square; colorbar; title('MATLAB'); % posterior expectation of edge probability

% probability distribution

[pr,~] = sample_dist(Gsamples);
figure; bar(sort(pr)); % distribution of model probabilities (often extremely flat, as many models are allowed with slight differences)


%% MAP estimate G'= argmax_G [P(G|N)]

prior.a = 1;
prior.b = 4; % expected density = a/(a+b)

nsamples = 100;
G = zeros(p);
T = 10; % initial temperature
Tr = .5^(10/nsamples); % temperature decay

tic;
for i=1:nsamples    
    G = struct_conn_density_prior(G,N,[],prior, T);
    T = T*Tr;
end
toc;

figure;
imagesc(G + eye(p)); colormap hot; axis square; colorbar; % MAP estimate of G


%% estimate functional connectivityas precision K (or scaled to partial correlations R) with structural constraint G: P(K|G,X)

nsamples = 100;
Rsamples = zeros(p,p,nsamples);
S = cov(X);
Sinv = inv(S);
df = 3 + n;

tic;
for i=1:nsamples
    K = gwishrnd_mex(G, Sinv, df); % diagonal must be ones
    % K = gwishrnd3(max(G,eye(p)), S, df); % Matlab alternative is slower,
    % use only if MEX is not available
    Rsamples(:,:,i) = prec2parcor(K);    
end
toc;

figure;
imagesc(mean(Rsamples,3)); colormap jet; axis square; colorbar; caxis([-1 1]); % posterior expectation of partial correlations given G

%% estimate MAP structural connectivity and clustering with nonparametric prior: P(A,Z|N)

mem = crprnd(log(p),p);
Z = mem2cluster(mem);

Ns = {N};
nsubjects = length(Ns);
As = {};

for i=1:nsubjects
    As{i} = zeros(p);
end

nsamples = 1000;

conn_samples = cell(nsamples,nsubjects);
clust_samples = cell(nsamples,1);
num_clusters = zeros(nsamples,1);

T = 10; % initial temperature
Tr = .5^(10/nsamples); % temperature decrease

prior.alpha = log(p); % concentration parameter of Chinese restaurant process
prior.betap = [1 1]; % [1 1] for uninformative prior, [x y], with x>y for modular clusters
prior.betan = fliplr(prior.betap);

for i=1:nsamples    
    [As, Z] = struct_conn_irm_prior(As, Z, N, {}, prior, T);
    for n=1:nsubjects
        conn_samples{i,n} = As{n}+As{n}';
    end
    clust_samples{i} = Z;
    num_clusters(i) = size(Z,1);
    T = T*Tr;
end


figure;
for s=1:nsubjects
    A = conn_samples{end,s};
    Z = clust_samples{end};
    subplot(1,nsubjects,s);
    plot_clustering(A,Z);  
end

figure;
plot(num_clusters);


