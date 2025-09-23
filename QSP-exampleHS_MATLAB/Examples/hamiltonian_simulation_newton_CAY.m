%% GENERAL IDEA
% We want to implement exp(-i \tau H) = f(x) = cos(\tau x) -i sin(\tau x)
% on x \in [-1,1]
% As it has two different defined parities, it is calculated in two steps
% T_(2k) and T_(2k+1) and later combined to construct the whole f(x)

addpath('C:\Users\CristinaÁlvarezYuste\Documents\GitHub\OptCont-Sch\QSP example HS_MATLAB\Examples\chebfun'); % Adds the folder containing Chebfun helpers to MATLAB's search path so chebfun, chebcoeffs... are callable.

%% Hamiltonian simulation
% In Hamiltonian simulation, the function of interest is $f(x)=e^{-i\tau
% x}$.  In practice, the real and imaginary component of the complex
% polynomial is implemented separately, and are then combined by linear
% combination of unitaries.
% Thus we only need to determine the phase factors corresponding to those polynomials approximating $\cos(\tau x)$ and $\sin(\tau x)$. 








%% Approximating the real component
% Consider the real part $0.5\cos(100 x)$, whose $L^{\infty}$
% norm over $[-1,1]$ is strictly bounded by $\frac{1}{2}$.
parity_r = 0;                     % 0 = par (coeficientes T_{0},T_{2},...)
tau = 100;                        % Evolution time
targ_r = @(x) 0.5*cos(tau.*x);

%%
% The Chebyshev coefficients can be computed using |chebfun|. We truncate the series up
% to $d=1.4| \tau |+\log(1/\epsilon_0)$ such that the approximation error is 
% bounded by $\epsilon_0$.
d = ceil(1.4*tau+log(1e14));   % "ceil" rounds towards positive infinity to the nearest integer
f_r = chebfun(targ_r,d);         % Constructs the degree-d Chebyshev interpolant to targ
coef_r_full = chebcoeffs(f_r);          % Returns the Chebyshev coefficients c_k in the T_k basis (c_0, c_1, c_2..., c_d)

%%
% We only need its Chebyshev coefficients with respect to $T_{2k}$, where
% $k$ is nonegative integer.
coef_r = coef_r_full(parity_r+1:2:end);

%%
% Set up the parameters for the solver.
opts.maxiter = 100;
opts.criteria = 1e-12;

%%
% Set |opts.useReal| to be |true| will increase the computing speed.
opts.useReal = true;    % Avoids complex arithmetic where not needed

%%
% We want the real component of the upper left entry of the QSP unitary
% matrix to be the target function.
opts.targetPre = true; % Tells QSPPACK that our target lives in the real part of the (1,1) entry of the QSP unitary U_d(x,Φ): we want Re U₁₁(x,Φ) = ½ cos(τx). 
                       
%%
% Use the fixed point iteration method to find phase factors
opts.method = 'Newton'; % Can also use coordinate minimization "CM", L-BFGS, or NLFT
[phi_proc_r,out_r] = QSP_solver(coef_r,parity_r,opts);

% QSP_solver ingests the Chebyshev coefficients of the even polynomial and returns:
% - phi_proc: the reduced list of phase angles (only the unique half because symmetry reconstructs the full list),
% - out: info needed to evaluate the QSP unitary (parity, targetPre, and convention data)

%%
% We do the following test to demonstrate that the obtained phase factors 
% satisfy expectation.
xlist = linspace(0, 1, 1000)';                    % Builds 1000 sample points in [0,1] although any subset in [-1,1] is valid
targ_value_r = targ_r(xlist);                         % Evaluates 1/2 cos(\tau x) on those points
QSP_value_r = QSPGetEntry(xlist, phi_proc_r, out_r);    % Reconstructs the full phase list from phi_proc and out and evaluates the QSP (1,1) entry which with the targetPre convention, it returns the real part that should match the target
err_r = norm(QSP_value_r-targ_value_r,1)/length(xlist);
disp('The residual error is');
disp(err_r);

% Overlay the QSP result vs the exact target (should be visually indistinguishable)
figure
hold on
plot(xlist,QSP_value_r,'b-')
plot(xlist,targ_value_r,'r--')
hold off
legend('QSP','Target')
xlabel('$x$', 'Interpreter', 'latex')
% Plot the pointwise error
figure
plot(xlist,QSP_value_r-targ_value_r)
xlabel('$x$', 'Interpreter', 'latex')
ylabel('$g(x,\Phi^*)-f(x)$', 'Interpreter', 'latex')
print(gcf,'hamiltonian_simulation.png','-dpng','-r500');






%% Approximating the imaginary component
% Consider the imaginary part $0.5\sin(100 x)$, whose $L^{\infty}$
% norm over $[-1,1]$ is strictly bounded by $\frac{1}{2}$.
% Evolution time is the same as before (\tau)
parity_i = 1;                     % 1 = par (coeficientes T_{1},T_{3},...)
targ_i = @(x) 0.5*sin(tau.*x);

%%
% Same Chebyshev truncation degree rule
f_i = chebfun(targ_i,d);         % Constructs the degree-d Chebyshev interpolant to targ
coef_i_full = chebcoeffs(f_i);        % Returns the Chebyshev coefficients c_k in the T_k basis (c_0, c_1, c_2..., c_d)

%%
% We only need its Chebyshev coefficients with respect to $T_{2k+1}$, where
% $k$ is nonegative integer.
coef_i = coef_i_full(parity_i+1:2:end);

%%
% Same parameters, opts.useReal, opts.targetPre, opts.method  for the solver.

[phi_proc_i,out_i] = QSP_solver(coef_i,parity_i,opts);

%%
% We do the following test to demonstrate that the obtained phase factors 
% satisfy expectation.
targ_value_i = targ_i(xlist);                         % Evaluates 1/2 cos(\tau x) on those points
QSP_value_i = QSPGetEntry(xlist, phi_proc_i, out_i);    % Reconstructs the full phase list from phi_proc and out and evaluates the QSP (1,1) entry which with the targetPre convention, it returns the real part that should match the target
err_i= norm(QSP_value_i-targ_value_i,1)/length(xlist);
disp('The residual error is');
disp(err_i);

% Overlay the QSP result vs the exact target (should be visually indistinguishable)
figure
hold on
plot(xlist,QSP_value_i,'b-')
plot(xlist,targ_value_i,'r--')
hold off
legend('QSP','Target')
xlabel('$x$', 'Interpreter', 'latex')
% Plot the pointwise error
figure
plot(xlist,QSP_value_i-targ_value_i)
xlabel('$x$', 'Interpreter', 'latex')
ylabel('$g(x,\Phi^*)-f(x)$', 'Interpreter', 'latex')
print(gcf,'hamiltonian_simulation.png','-dpng','-r500');






%% --- Combine even+odd to emulate exp(-i tau x)
QSP_value_full   = 2*(QSP_value_r - 1i*QSP_value_i);   % 2*(1/2 cos - i*1/2 sin)
target_full  = exp(-1i*tau*xlist);
comb_err = norm(QSP_value_full - target_full)/max(1e-16, norm(target_full));
fprintf('Scalar combo error vs exp(-i tau x): %.3e\n', comb_err);

figure; 
plot(xlist, real(QSP_value_full), 'b-', xlist, real(target_full), 'r--'); grid on
legend('Re combined','Re exact'); title('Real part: combined vs exact')

figure; 
plot(xlist, imag(QSP_value_full), 'b-', xlist, imag(target_full), 'r--'); grid on
legend('Im combined','Im exact'); title('Imag part: combined vs exact')
