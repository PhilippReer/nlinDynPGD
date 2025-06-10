%% Proper Generalized Decomposition for the linear pinned-pinned bar
% u_tt - K u_xx = sum_i f_i(x)*f_i(t)
% Philipp Reer


%% Model Parameters
% Solver
numX = 64;                  % Number of discretized nodes in space [-]
numT = 512;                 % Number of discretized nodes in time [-]
maxEnrichmentTerms = 10;    % Maximal number of computed enrichment terms [-]
enrichTol = 10^-3;          % Tolerance for the enrichment terms
iterTol = 10^-3;            % Tolerance for the alternating direction
pmax = 40;                  % Maximal number of alternating dir. iterations 

% Bar
L = 10;             % Length of bar [m]
Tau = 0.02;       	% Maximum simulation time [s]
K = 2.68*10^7;      % specific modulus [m^2/s^2]
x0 = 0.5e-3;        % Initial displacment amplitude [m]
initialDisplacementMode = 1;    % Initial displacment mode [-]

% Multimode harmonic forcing
fij = 1000*[1, 1.2; ...     % Mode 1 excitation amplitude
          0.5, 0; ...       % Mode 2
          3, 0; ...         % Mode 3
          0, 0;];           % Mode 4
Om = 2*pi*[30, 10; ...      % Mode 1 excitation frequency
            100, 0; ...     % Mode 2
            300, 0; ...     % Mode 3
            0.0, 0;];       % Mode 4

%% Computations before the PGD
x = linspace(0, L, numX)';              % discretized space
t = linspace(0, Tau, numT)';            % discretized time

om0 = pi*sqrt(K)/L * (1:size(fij,1));   % natural frequency of each mode

% Placing the forcing into the expected separated representation format
fx = zeros(numX, numel(fij));
ft = zeros(numT, numel(fij));
for i = 1:size(fij,1)
    for j = 1:size(fij,2)
        index = size(fij,2)*(i-1)+j;
        fx(:,index) = sin(i*pi*x/L);
        ft(:,index) = fij(i,j)*cos(Om(i,j)*t);
    end
end

% separated representation of the initial displacement
Gx = [sin(initialDisplacementMode*pi*x/L)];
Gt = [x0*(-t/Tau+1)]; 

% Analytical result derived in A.1
u_anal = x0* sin(initialDisplacementMode*pi/L*x).*cos(om0(initialDisplacementMode)*t)'; % homogeneous vibration due to IC
for i = 1:size(fij,1)
    for j = 1:size(fij,2)
        u_anal = u_anal + sin(i*pi*x/L)*(fij(i,j)/(om0(i)^2-Om(i,j)^2) * (cos(Om(i,j)*t)-cos(om0(i)*t)))';
    end
end

% Finite Difference Matrices for approximation of derivatives
dx = x(2)-x(1);
dt = t(2)-t(1);
D2x = spdiags([ones(numX,1)  -2*ones(numX,1)  ones(numX,1)],...
    [-1 0 1],numX,numX)/dx^2;
D2x(1,:) = D2x(2,:);
D2x(numX,:) = D2x(numX-1,:);
D2t = spdiags([ones(numT,1)  -2*ones(numT,1)  ones(numT,1)],...
    [-1 0 1],numT,numT)/dt^2;
D2t(1,:) = D2t(2,:);
D2t(numT,:) = D2t(numT-1,:);


%% PGD method
% Define size of the matrices saving the enrichment terms
F = zeros(numX,0);
G = zeros(numT,0);

% The boundary contions are included as the first enrichment terms
% according to (3.23)
numBC = size(Gx,2);
F = [F, Gx];
G = [G, Gt];

% Loop for the enrichment terms
for n = numBC+1:maxEnrichmentTerms+numBC
    % arbitrary initialization of the new modeshapes that satisfys the BCs
    R = ones(numX, 1); R(1) = 0; R(end) = 0; 
    S = ones(numT, 1); S(1:2) = 0;
    
    % alternating direction strategy
    for p = 1:pmax
        R_old = R; S_old = S;
        
        % Integrals according to (4.10)
        alpha_x = trapz(t, S.*(D2t*S));
        beta_x = trapz(t, S.^2);
        xi_x = trapz(t, S.*ft);
        if n > 1
            gamma_x_i = trapz(t, S.*(D2t*G));
            delta_x_i = trapz(t, S.*G);
            RHS = fx*xi_x' - F*gamma_x_i' + K*D2x*F*delta_x_i';
        else
            RHS = fx*xi_x';
        end
        LHS = -K*beta_x * D2x + alpha_x * speye(numX);
        % Solve ODE (4.11) using finite difference matrices
        R(2:end-1) = LHS(2:end-1,2:end-1)\RHS(2:end-1);
        % Normalization of spatial mode
        R = R / max(abs(R));
        if sum(R) < 0
            R = -R;
        end
        
        % Integrals according to (4.14)
        alpha_t = trapz(x, R.^2);
        beta_t = trapz(x, R.*(D2x*R));
        xi_t = trapz(x, R.*fx);
        if n > 1
            gamma_t_i = trapz(x, R.*F);
            delta_t_i = trapz(x, R.*(D2x*F));
            RHS = ft*xi_t' - D2t*G *gamma_t_i' + K*G*delta_t_i';
        else
            RHS = ft*xi_t';
        end
        % Solve (4.15) using ode45
        [~, SdS] = solve2ndODE(alpha_t, 0, -K*beta_t, RHS, t, 0, 0);
        % Extract displacement, neglect velocity
        S = SdS(:,1);

        % Stopping criterion alternating direction strategy
        diff = norm(R*S'-R_old*S_old',2)/norm(R_old*S_old',2);
        if(diff < iterTol)
            disp("Enrichment n=" + string(n-numBC)+" finished after iterations p="+string(p))
            break;
        end
    end
    
    % Stopping criterion enrichment
    enrichDiff = norm(R*S',2) / norm(F*G',2);
    if enrichDiff < enrichTol
        disp("Enrichment finished with " + string(n-numBC)+" terms")
        break
    end
    % Add the new enrichment terms to the solution (4.4)
    F = [F, R];
    G = [G, S];
end


%% Plot result
% Shape functions 
figure()
subplot(2,1,1)
hold on;
title("Shapefunctions / domain functions in x domain")
plot(x, F);
xlabel("x")
ylabel("F(x)")
legend(string(1:size(F,2)))
subplot(2,1,2)
hold on;
title("Shapefunctions / domain functions in t domain")
plot(t, G);
xlabel("t")
ylabel("F(t)")
legend(string(1:size(F,2)))

% result
figure()
hold on
result = (F*G')';
surf(x,t,result, 'EdgeColor', 'none')
plot3(x, t(1)*ones(numX,1), result(1,:), Color="black", LineWidth=6)
xlabel("x")
ylabel("t")
zlabel("u")
title("PGD for the 1D bar")
view(50, 22)


%% Additional Functions used in the PGD method
function [t, x] = solve2ndODE(a, b, c, F, tSpan, x0, dx0)
% Solves 2nd Order IVPs of form
% a x'' + b x' + c x = F(t)
% with x0 = x(t=0), dx0 = v(t=0)
% and F(t) discrete values
% Returns the value of x for all t in tSpan
    function xPrime = ode(t, x)
        Fi = linearInterpolation(tSpan, F, t);
        xPrime(1,1) = x(2);
        xPrime(2,1) = 1/(a)*(-b*x(2)-c*x(1)+Fi);
    end
[t,x] = ode45(@ode, tSpan, [x0, dx0]);
end

function x_int = linearInterpolation(t, x, t_int)
% Interpolates the value of x(t_int) from the discrete vector x, and its
% time t. Quicker than the matlab interp1() function

    % Find the interval containing t_int
    idx = find(t <= t_int, 1, 'last');
    if t(idx) == t_int
        x_int = x(idx);
        return;
    end
    % Perform linear interpolation
    t1 = t(idx);
    t2 = t(idx+1);
    x1 = x(idx);
    x2 = x(idx+1);
    x_int = x1 + (x2 - x1) * (t_int - t1) / (t2 - t1);
end
