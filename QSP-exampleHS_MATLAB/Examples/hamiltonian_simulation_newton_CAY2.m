%% Hamiltonian simulation with QSP (cos + i*sin pieces) and explicit combination
% Steps:
%   1) scales H -> Hs so spec(Hs) ⊂ [-1,1]
%   2) builds high-accuracy Chebyshev approximants to 0.5*cos(T*x) and 0.5*sin(T*x)
%   3) uses QSPPACK to find phase factors for even (cos) and odd (sin) targets
%   4) verifies scalar responses g(x) against targets
%   5) evaluates the matrix polynomials on Hs and combines them to emulate e^{-i*tau*H}
%      (the 0.5 scale is removed at the very end)




%% Setup 
clear; clc;
addpath('C:\Users\CristinaÁlvarezYuste\Documents\GitHub\OptCont-Sch\QSP example HS_MATLAB\Solvers\Optimization');
addpath('C:\Users\CristinaÁlvarezYuste\Documents\GitHub\OptCont-Sch\QSP example HS_MATLAB\Examples\chebfun');

% Pauli matrices
I2 = eye(2);
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];

% Example of a hermitian matrix (as it is made from hermitian terms and real coeffs)
H = 0.40*kron(sx, I2) ...
  + 0.30*kron(I2, sz) ...
  + 0.20*kron(sy, sx) ...
  + 0.10*kron(sz, sz);

% Physical evolution time 
tau = 50;                  % degree grows ~ linearly with tau*alpha

% Spectral scaling so the QSP variable x ∈ [-1,1]
lam = eig(H);
alpha = max(abs(lam));      % scale factor >= spectral radius (then all eigenvalues of A/alpha lie inside the unit interval)
Hs = H / alpha;             % spectrum(Hs) ⊂ [-1,1]
T = tau * alpha;            % we approximate e^{-i * T * x} on x ∈ [-1,1]

fprintf('||H||_spec ≈ %.6f, using T = %.6f\n', alpha, T);

%% Chebyshev targets (even=cos, odd=sin) 
% Approximation of 0.5*cos(tau_eff*x) and 0.5*sin(tau_eff*x).
% The 0.5 keeps the L^∞ norm below 1/2; it is unscaled by 2 at the very end.

eps0 = 1e-12;                             % target uniform accuracy
d    = ceil(1.4*T + log(1/eps0));         % degree heuristic  (QSPPACK tutorial)
fprintf('Chebyshev degree d = %d\n', d);

targ_cos = @(x) 0.5*cos(T.*x);
targ_sin = @(x) 0.5*sin(T.*x);

% Chebyshev expansions via Chebfun
f_cos  = chebfun(targ_cos, d);            % degree-d Chebyshev interpolant on [-1,1]
f_sin  = chebfun(targ_sin, d);

coef_cos_full = chebcoeffs(f_cos);        % a_k for T_k(x), k=0..d
coef_sin_full = chebcoeffs(f_sin);

% Parity-restricted coefficient lists for the QSP solver:
%   parity=0 -> keep T_{0},T_{2},...  
%   parity=1 -> keep T_{1},T_{3},...
coef_even = coef_cos_full(1:2:end);       % even Chebyshev modes for cosine
coef_odd  = coef_sin_full(2:2:end);       % odd Chebyshev modes for sine

%% Solve for QSP phases (even & odd) 
opts = struct();
opts.maxiter   = 100;
opts.criteria  = 1e-12;
opts.useReal   = true;     % speedup (SU(2) multiplication in reals)
opts.targetPre = true;     % make the (1,1) entry's real component the target
opts.method = 'Newton';

% Even / cosine
[phi_even, out_even] = QSP_solver(coef_even, 0, opts);

% Odd / sine
[phi_odd,  out_odd ] = QSP_solver(coef_odd, 1, opts);

%% Check scalar responses g(x) ≈ targets on [-1,1] 
% Check of the phases at scalar level
xlist = linspace(-1,1,1501).';
g_even = QSPGetEntry(xlist, phi_even, out_even);    % should match 0.5*cos(T*x)
g_odd  = QSPGetEntry(xlist, phi_odd,  out_odd );    % should match 0.5*sin(T*x)

E_even = mean(abs(g_even - targ_cos(xlist)));
E_odd  = mean(abs(g_odd  - targ_sin(xlist)));
fprintf('Mean abs error (even/cos): %.3e\n', E_even);
fprintf('Mean abs error (odd/sin) : %.3e\n', E_odd);

figure; plot(xlist, g_even, 'b-', xlist, targ_cos(xlist), 'r--'); grid on;
title('Even piece: QSP vs target'); legend('QSP','0.5 cos(T x)'); xlabel('x');

figure; plot(xlist, g_odd, 'b-', xlist, targ_sin(xlist), 'r--'); grid on;
title('Odd piece: QSP vs target'); legend('QSP','0.5 sin(T x)'); xlabel('x');

%% Build the matrix polynomial approximation to exp(-i*tau*H) 
% We now classically apply the SAME Chebyshev series to Hs
% Using T_0(A)=I, T_1(A)=A, T_{k+1}(A) = 2 A T_k(A) - T_{k-1}(A).
% The final combination removes the 0.5 scale by multiplying by 2:
%     U_approx = 2 * (F_cos(Hs) - 1i * F_sin(Hs))  ≈  e^{-i * tau * H}.


% Evaluate the Chebyshev series on Hs via the standard recursion
Fcos = cheb_eval_matrix(Hs, coef_cos_full);    % sum_k a_k T_k(Hs)
Fsin = cheb_eval_matrix(Hs, coef_sin_full);

U_approx = 2 * (Fcos - 1i * Fsin);
U_exact  = expm(-1i * tau * H);

relFro   = norm(U_approx - U_exact, 'fro') / max(1e-16, norm(U_exact,'fro'));
unit_err = norm(U_approx'*U_approx - eye(size(H)), 'fro');   % unitarity check

fprintf('Relative Fro error vs expm: %.3e\n', relFro);
fprintf('Unitarity residual ||U^*U - I||_F: %.3e\n', unit_err);








%% Save data


meta = struct('tau',tau, 'alpha',alpha, 'T',T, ...
              'parity',[0 1], 'targetPre',opts.targetPre, ...
              'method',opts.method, 'degree_cos', numel(coef_cos_full)-1,'degree_sin', numel(coef_sin_full)-1);

outdir = 'C:\Users\CristinaÁlvarezYuste\Documents\GitHub\OptCont-Sch\QSP example HS_MATLAB\Examples';
if ~exist(outdir,'dir'); mkdir(outdir); end

save(fullfile(outdir,'qsp_phases_tau50.mat'), 'phi_even','phi_odd','meta','H','Hs','-v7'); 





















%% ========================= local helper functions =========================
function F = cheb_eval_matrix(A, a_full)
% Evaluate sum_{k=0}^d a_k T_k(A) using Chebyshev recursion.
    N = numel(a_full)-1;
    I = eye(size(A));
    if N==0, F = a_full(1)*I; return; end
    Tkm1 = I; Tk = A;
    F = a_full(1)*Tkm1 + a_full(2)*Tk;
    for k = 2:N
        Tkp1 = 2*A*Tk - Tkm1;
        F = F + a_full(k+1)*Tkp1;
        Tkm1 = Tk; Tk = Tkp1;
    end
end

