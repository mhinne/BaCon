function [a,d] = wishrnd(sigma,df,d)
%WISHRND Generate Wishart random matrix
%   W=WISHRND(SIGMA,DF) generates a random matrix W having the Wishart
%   distribution with covariance matrix SIGMA and with DF degrees of
%   freedom.
%
%   W=WISHRND(SIGMA,DF,D) expects D to be the Cholesky factor of
%   SIGMA.  If you call WISHRND multiple times using the same value
%   of SIGMA, it's more efficient to supply D instead of computing
%   it each time.
%
%   [W,D]=WISHRND(SIGMA,DF) returns D so it can be used again in
%   future calls to WISHRND.
%
%   See also IWISHRND.

%   References:
%   Krzanowski, W.J. (1990), Principles of Multivariate Analysis, Oxford.
%   Smith, W.B., and R.R. Hocking (1972), "Wishart variate generator,"
%      Applied Statistics, v. 21, p. 341.  (Algorithm AS 53)

%   Copyright 1993-2007 The MathWorks, Inc. 
%   $Revision: 1.1.8.1 $  $Date: 2010/03/16 00:18:29 $

% Error checking


n = size(sigma,1);

[d,p] = cholcov(sigma,1);
 


% For small degrees of freedom, generate the matrix using the definition
% of the Wishart distribution; see Krzanowski for example
if (df <= 81+n) && (df==round(df))
   x = randn(df,size(d,1)) * d;

% Otherwise use the Smith & Hocking procedure
else
   % Load diagonal elements with square root of chi-square variates
   a = diag(sqrt(chi2rnd(df-(0:n-1))));

   % Load upper triangle with independent normal (0, 1) variates
   a(itriu(n)) = randn(n*(n-1)/2,1);

   % Desired matrix is D'(A'A)D
   x = a(:,1:size(d,1))*d;
end

a = x' * x;


% --------- get indices of upper triangle of p-by-p matrix
function d=itriu(p)

d=ones(p*(p-1)/2,1);
d(1+cumsum(0:p-2))=p+1:-1:3;
d = cumsum(d);
