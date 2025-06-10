%% Proper Generalized Decomposition for the beam with nonlinear spring
% u_tt(x,t) + K u_xxxx(x,t) = p1(x)*p1(t) + fk(x,t)

%% Model Parameters
% discretization for first Beam
numX = 64;              % Number of discretized nodes in space [-]
numT = 512;             % Number of discretized nodes in timr [-]
maxEnrichmentTerms = 20;% Maximal number of computed enrichment terms [-]
enrichTol = 10^-3;      % Tolerance for the enrichment terms [-]
iterTol = 10^-3;        % Tolerance for the alternating direction [-]
pmax = 40;              % Maximal number of alternating dir. iterations [-] 

Tau = 10;           % max time [s]

% Parameter beam
L = 2.5;            % length of beam [m]
K = 40;             % K = EI/(rho A) [m^4/s^2]
g_nlin = 0.10;           % nonlinearity param [1/m^2/s^2]   

% Forcing amplitude and frequency in the first free vibration mode
F0 = 40;            % amplitude of the forcing [m/s^2]
Omega = 0.15*2*pi;  % frequency of harmonic excitation [rad/s]


%% Pre PGD calculations
% Discretization
x = linspace(0, L, numX)';
t = linspace(0, Tau, numT)';
dx = x(2)-x(1);
dt = t(2)-t(1);

% Free Vibration Modes beam
kx1 = 1.8751040687;
k1 = kx1/L;
gamma = (cosh(kx1)+cos(kx1))/ (sinh(kx1)+sin(kx1));
% forcing in first mode 
fx = cosh(k1*x) - gamma*sinh(k1*x) - cos(k1*x) + gamma*sin(k1*x);
ft = F0 * cos(Omega*t); 

% 2nd order derivative approximation operator
D2t = spdiags([ones(numT,1)  -2*ones(numT,1)  ones(numT,1)],...
    [-1 0 1],numT,numT)/dt^2;
D2t(1,:) = D2t(2,:);
D2t(numT,:) = D2t(numT-1,:);
% 4th order derivative approximation operator
D4x = spdiags([1*ones(numX,1) -4*ones(numX,1) 6*ones(numX,1) ...
    -4*ones(numX,1) 1*ones(numX,1)],-2:1:2,numX,numX)/dx^4;
D4x(1,:) = D4x(3,:); D4x(2,:) = D4x(3,:); 
D4x(end,:) = D4x(end-2,:); D4x(end-1,:) = D4x(end-2,:);

% Define size of the matrices saving the enrichment terms
F = zeros(numX,0);
G = zeros(numT,0);

%% PGD method
% Loop for the enrichment terms
for n = 1:maxEnrichmentTerms
    % arbitrary initialization of the new modeshapes
    R = ones(numX, 1);
    S = ones(numT, 1);
    
    % alternating direction strategy
    for p = 1:pmax
        R_old = R; S_old = S;

        % generalized slow introduction of nonlinearity
        % first steps linear, then slow increase of nlin., then fully nlin
        p_lin = 5;
        p_gradualNlin = pmax-5;
        if p < p_lin
            g = 0;
        elseif p <= p_gradualNlin
            g = g_nlin / (p_gradualNlin - p_lin) * (p-p_lin);
        else
            g = g_nlin;
        end
        
        % R(x) from S(t)
        % Integrals according to (4.37)
        alpha_x = trapz(t, S.*(D2t*S));
        beta_x = trapz(t, S.^2);
        gamma_x = trapz(t, S.^2 .* (F*G')'.^2);
        eta_x = trapz(t, S.*ft);
        if n>1
            delta_x_i = trapz(t, S.*(D2t*G));
            epsilon_x_i = trapz(t, S.*G);
            zeta_x = trapz(t, S.*(F*G')'.^3);
            RHS = fx*eta_x - F*delta_x_i' - K*D4x*F*epsilon_x_i' - g*zeta_x';
            % Due to the finite difference approximation of the derivative
            % the RHS has peaks at the end of the domain, which lead to
            % solver instabilities. This remedies it.
            RHS(end) = RHS(end-2); RHS(end-1) = RHS(end-2);
        else 
            RHS = fx*eta_x;
        end
        % Solve ODE (4.36) using Matlab bvp5c method
        [x, R] = solve4thBVP(K*beta_x, alpha_x+3*g*gamma_x', RHS, x);
        R = R';
        R = 1 * R / max(abs(R));
        if sum(R) < 0
            R = -R;
        end      

        % S(t) from R(x)
        % Integrals according to (4.39)
        alpha_t = trapz(x, R.^2);
        beta_t = trapz(x, R.*(D4x*R));
        gamma_t = trapz(x, R.^2 .* (F*G').^2);
        eta_t = trapz(x, R.*fx);
        if n>1
            delta_t_i = trapz(x, R.*F);
            epsilon_t_i = trapz(x, R.*(D4x*F));
            zeta_t = trapz(x, R.*(F*G').^3);
            RHS = ft*eta_t - D2t*G*delta_t_i' - K*G*epsilon_t_i' - g*zeta_t';
        else 
            RHS = ft*eta_t;
        end
        % Solve ODE (4.38) using Matlab bvp5c method
        [~, SdS] = solve2ndODE(alpha_t, 0, K*beta_t+3*g*gamma_t, RHS, t, 0, 0);
        S = SdS(:,1);

        diff = norm(R*S'-R_old*S_old',2)/norm(R_old*S_old',2);
        if(diff < iterTol && p > p_gradualNlin)
           disp("Enrichment n=" + string(n)+" finished after iterations p="+string(p))
           break;
        end
    end
    F = [F, R]; 
    G = [G, S];
    % Stopping criterion enrichment
    enrichDiff = norm(R1*S1',2) / norm(F1*G1',2);
    if enrichDiff < enrichTol
        disp("Enrichment finished with " + string(n)+" terms")
        break
    end
end

%% Plots
dx = x(2)-x(1);
dt = t(2)-t(1);
% 2nd order derivative approximation operator
D2t = spdiags([ones(numT,1)  -2*ones(numT,1)  ones(numT,1)],...
    [-1 0 1],numT,numT)/dt^2;
D2t(1,:) = D2t(2,:);
D2t(numT,:) = D2t(numT-1,:);
% 4th order derivative approximation operator
D4x = spdiags([1*ones(numX,1) -4*ones(numX,1) 6*ones(numX,1) ...
    -4*ones(numX,1) 1*ones(numX,1)],-2:1:2,numX,numX)/dx^4;
D4x(1,:) = D4x(3,:); D4x(2,:) = D4x(3,:); 
D4x(end,:) = D4x(end-2,:); D4x(end-1,:) = D4x(end-2,:);

% Result Beam End
figure()
result = (F*G')';
plot(t, result(:,end))
xlabel("t")
ylabel("u")
title("Free beam end (x=L) for \gamma =" + num2str(g) + ", \tau =" + num2str(Tau)+"s, f=" + num2str(F0))


%% Additional Functions used in the PGD method
function [t, x] = solve2ndODE(a, b, c, F, tSpan, x0, dx0)
% Solves 2nd Order IVPs of form
% a x'' + b x' + c x = F(t)
% with x0 = x(t=0), dx0 = v(t=0)
% and F(t) discrete values
% Returns the value of x for all t in tSpan
    function xPrime = ode(t, x)
        Fi = interp1(tSpan, F, t);
        ci = interp1(tSpan, c, t);
        xPrime(1,1) = x(2);
        xPrime(2,1) = 1/(a)*(-b*x(2)-ci*x(1)+Fi);
    end
[t,x] = ode78(@ode, tSpan, [x0, dx0]);
end

function [t, x] = solve4thBVP(a, b, F, tSpan)
% Solve a 4th Order BVP of the form:
% a x'''' + b x = F(t)
% With predefined BCs

    function yPrime = odefun(t,y)
        Fi = interp1(tSpan, F, t);
        bi = interp1(tSpan, b, t);
        yPrime = [y(2); y(3); y(4); -(bi/a)*y(1) + (1/a)*Fi];
    end

bcfun = @(ya, yb) [ya(1)-0; % R(0) = 0
                   ya(2)-0; % R'(0) = 0
                   yb(3)-0; % R''(L) = 0
                   yb(4)-0]; % R'''(L) = 0
solinit = bvpinit(tSpan, [0 0 0 0]);
sol = bvp5c(@odefun, bcfun, solinit);
x = deval(sol, tSpan);
x = x(1, :);
t = tSpan;
end


