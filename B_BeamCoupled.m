%% Proper Generalized Decomposition for two coupled damped linear beams
% rho1 A1 w1_tt(x,t) + EI w1_xxxx(x,t) = p1(x)*p1(t) + fk1(x,t)
% rho2 A2 w2_tt(x,t) + EI w2_xxxx(x,t) = p2(x)*p2(t) + fk2(x,t)
% fk1(x,t) = k(w2-w1); fk2(x,t) = k(w1-w2);

%% Model Parameters
% Solver parameters
numX1 = 64;             % Number of discretized nodes in space [-]
numX2 = numX1;          % Number of discretized nodes in space [-]
numT = 512;             % Number of discretized nodes in time [-]
maxEnrichmentTerms = 2; % Maximal number of computed enrichment terms [-]
enrichTol = 10^-3;      % Tolerance for the enrichment terms [-]
iterTol = 10^-3;        % Tolerance for the alternating direction [-]
pmax = 40;              % Maximal number of alternating dir. iterations [-]


Tau = 30;           % [s] max time

% Spring Coupling
k_coupling = 20;    % [N/m] coupling spring

% Parameters 1st Beam
L1 = 2.5;           % [m] length of beam
E1 = 210 *10^6;     % [N/m^2]
h1 = 0.04;          % [m]
w1 = 0.05;          % [m]
A1 = h1*w1;         % [m^2]
I1 = w1*h1^3/12;    % [m^4]
rho1 = 7850;        % [kg/m^3]
C1 = 0.15;          % internal damping [1/s]
D1 = 0.1;           % external damping [kg/s/m]

% Parameters 2nd Beam
L2 = L1;            % [m] length of beam
E2 = 210 *10^6;     % [N/m^2]
h2 = 0.06;          % [m]
w2 = 0.1;           % [m]
A2 = h2*w2;         % [m^2]
I2 = w2*h2^3/12;    % [m^4]
rho2 = 7850;        % [kg/m^3]
C2 = 0.15;          % internal damping [1/s]
D2 = 0.05;          % external damping [kg/s/m]

% Forcing amplitude and frequency in the first free vibration mode 
F01 = 1;
Omega1 = 0.3*2*pi;
F02 = 2;
Omega2 = 0.8*2*pi;


%% Pre PGD calculations
% Free vibration mode 1 
kx1 = 1.8751040687;
k1 = kx1/L1;
gamma1 = (cosh(kx1)+cos(kx1))/ (sinh(kx1)+sin(kx1));
k2 = kx1/L2;
gamma2 = (cosh(kx1)+cos(kx1))/ (sinh(kx1)+sin(kx1));

% Forcing Beam 1
fx1= (cosh(k1*x1) - gamma1*sinh(k1*x1) - cos(k1*x1) + gamma1*sin(k1*x1));
ft1 = F01 * cos(Omega1*t) / max(fx1, [], "all");

% Forcing Beam 2
fx2= cosh(k2*x2) - gamma2*sinh(k2*x2) - cos(k2*x2) + gamma2*sin(k2*x2);
ft2 = F02 * cos(Omega2*t) / max(fx2, [], "all");

% Domain discretization
x1 = linspace(0, L1, numX1)';
x2 = linspace(0, L2, numX2)';
t = linspace(0, Tau,numT)';
dx1 = x1(2)-x1(1);
dx2 = x2(2)-x2(1);
dt = t(2)-t(1);

% 2nd order derivative approximation operator
D2t = spdiags([ones(numT,1)  -2*ones(numT,1)  ones(numT,1)],...
    [-1 0 1],numT,numT)/dt^2;
D2t(1,:) = D2t(2,:);
D2t(numT,:) = D2t(numT-1,:);
% 4th order derivative approximation operator
D4x1 = spdiags([1*ones(numX1,1) -4*ones(numX1,1) 6*ones(numX1,1) -4*ones(numX1,1) 1*ones(numX1,1)],-2:1:2,numX1,numX1)/dx1^4;
D4x1(1,:) = D4x1(3,:); D4x1(2,:) = D4x1(3,:); 
D4x1(end,:) = D4x1(end-2,:); D4x1(end-1,:) = D4x1(end-2,:);
% 4th order derivative approximation operator
D4x2 = spdiags([1*ones(numX2,1) -4*ones(numX2,1) 6*ones(numX2,1) -4*ones(numX2,1) 1*ones(numX2,1)],-2:1:2,numX2,numX2)/dx2^4;
D4x2(1,:) = D4x2(3,:); D4x2(2,:) = D4x2(3,:); 
D4x2(end,:) = D4x2(end-2,:); D4x2(end-1,:) = D4x2(end-2,:);
% 1st order derivative
D1t = spdiags([-ones(numT,1)  zeros(numT,1)  ones(numT,1)],...
    [-1 0 1],numT,numT)/(2*dt);


%% PGD method
% Define size of the matrices saving the enrichment terms
F1 = zeros(numX1,0);
G1 = zeros(numT,0);
F2 = zeros(numX2,0);
G2 = zeros(numT,0);

% Loop for the enrichment terms
for n = 1:maxEnrichmentTerms
    % arbitrary initialization of the new modeshapes
    R1 = ones(numX1, 1);
    S1 = ones(numT, 1);
    R2 = ones(numX2, 1);
    S2 = ones(numT, 1);

    % alternating direction strategy
    for p = 1:pmax
        R1_old = R1; S1_old = S1; R2_old = R2; S2_old = S2;
        k = k_coupling;

        % FIRST BEAM
        % R1(x) from S1(t)
        % Integrals according to (4.24)
        alpha_x = trapz(t, S1.*(D2t*S1));
        beta_x = trapz(t, S1.^2);
        mu_x = trapz(t, S1.*ft1);
        phi_x = trapz(t, S1.*(D1t*S1));
        xi1_x = trapz(t, S1.*S2);
        if n>1
            gamma_x_i = trapz(t, S1.*(D2t*G1));
            delta_x_i = trapz(t, S1.*G1);
            epsilon_x_i = trapz(t, S1.*(D1t*G1));
            xi2_x = trapz(t, S1.*G2);
            RHS = fx1*mu_x - k*F1*delta_x_i' + k*(F2*xi2_x'+R2*xi1_x')- rho1*A1*F1*gamma_x_i' ...
                - E1*I1*D4x1*F1*delta_x_i' - C1*E1*I1*D4x1*F1*epsilon_x_i' - D1*F1*epsilon_x_i';
        else 
            RHS = fx1*mu_x + k*R2*xi1_x;
        end
        % Solve ODE (4.23a) using Matlab bvp5c method
        [x1, R1] = solve4thBVP(E1*I1*beta_x+C1*E1*I1*phi_x, rho1*A1*alpha_x+k*beta_x+D1*phi_x, RHS, x1);
        R1 = R1';
        % Normalization of the spatial mode
        R1 = R1 / max(abs(R1));
        if sum(R1) < 0
            R1 = -R1;
        end

        % S1(t) from R1(x)
        % Integrals according to (4.26)
        alpha_t = trapz(x1, R1.^2);
        beta_t = trapz(x1, R1.*(D4x1*R1));
        mu_t = trapz(x1, R1.*fx1);
        xi1_t = trapz(x1, R1.*R2);
        if n>1
            gamma_t_i = trapz(x1, R1.*F1);
            delta_t_i = trapz(x1, R1.*(D4x1*F1));
            xi2_t = trapz(x1, R1.*F2);
            RHS = ft1*mu_t - k*G1*gamma_t_i' + k*(G2*xi2_t'+S2*xi1_t') - rho1*A1*D2t*G1*gamma_t_i' ...
                - E1*I1*G1*delta_t_i' - C1*E1*I1*D1t*G1*delta_t_i' - D1*D1t*G1*gamma_t_i';
        else 
            RHS = ft1*mu_t + k*S2*xi1_t;
        end
        % Solve ODE (4.25a) using Matlab ode78 method
        [~, SdS] = solve2ndODE(rho1*A1*alpha_t, C1*E1*I1*beta_t+D1*alpha_t, E1*I1*beta_t+k*alpha_t, RHS, t, 0, 0);    %IV: undeformed, at rest
        S1 = SdS(:,1);

        % SECOND BEAM
        % R2(x2) from S2(t)
        % Integrals according to (4.24)
        alpha_x = trapz(t, S2.*(D2t*S2));
        beta_x = trapz(t, S2.^2);
        mu_x = trapz(t, S2.*ft2);
        phi_x = trapz(t, S1.*(D1t*S1));
        xi1_x = trapz(t, S2.*S1);
        if n>1
            gamma_x_i = trapz(t, S2.*(D2t*G2));
            delta_x_i = trapz(t, S2.*G2);
            epsilon_x_i = trapz(t, S2.*(D1t*G2));
            xi2_x = trapz(t, S2.*G1);
            RHS = fx2*mu_x - k*F2*delta_x_i' + k*(F1*xi2_x'+R1*xi1_x) - rho2*A2*F2*gamma_x_i' ...
                - E2*I2*D4x2*F2*delta_x_i' - C2*E2*I2*D4x2*F2*epsilon_x_i' - D2*F2*epsilon_x_i';
        else 
            RHS = fx2*mu_x + k*R1*xi1_x;
        end
        % Solve ODE (4.23b) using Matlab bvp5c method
        [x2, R2] = solve4thBVP(E2*I2*beta_x+C2*E2*I2*phi_x, rho2*A2*alpha_x+k*beta_x+D2*phi_x, RHS, x2);
        R2 = R2';
        if any(R2)
            R2 = R2 / max(abs(R2));
        end

        % S2(t) from R2(x)
        % Integrals according to (4.26)
        alpha_t = trapz(x2, R2.^2);
        beta_t = trapz(x2, R2.*(D4x2*R2));
        mu_t = trapz(x2, R2.*fx2);
        xi1_t = trapz(x2, R2.*R1);
        if n>1
            gamma_t_i = trapz(x2, R2.*F2);
            delta_t_i = trapz(x2, R2.*(D4x2*F2));
            xi2_t = trapz(x2, R2.*F1);
            RHS = ft2*mu_t - k*G2*gamma_t_i' + k*(G1*xi2_t'+S2*xi1_t) - rho2*A2*D2t*G2*gamma_t_i' ...
                - E2*I2*G2*delta_t_i' - C2*E2*I2*(D1t*G2)*delta_t_i' - D2*(D1t*G2)*gamma_t_i';
        else
            RHS = ft2*mu_t + k*S1*xi1_t;
        end
        % Solve ODE (4.25a) using Matlab ode78 method
        [~, SdS] = solve2ndODE(rho2*A2*alpha_t, C2*E2*I2*beta_t+D2*alpha_t, E2*I2*beta_t+k*alpha_t, RHS, t, 0, 0);    %IV: undeformed, at rest
        S2 = SdS(:,1);
        
        % Stopping criterion alternating direction strategy
        diff1 = norm(R1*S1'-R1_old*S1_old',2)/norm(R1_old*S1_old',2);
        diff2 = norm(R2*S2'-R2_old*S2_old',2)/norm(R2_old*S2_old',2);
        if(diff1 < iterTol && diff2 < iterTol)
            disp("Enrichment n=" + string(n)+" finished after iterations p="+string(p))
            break;
        end
    end
    
    F1 = [F1, R1]; 
    G1 = [G1, S1];
    F2 = [F2, R2]; 
    G2 = [G2, S2];
    % Stopping criterion enrichment
    enrichDiff = norm(R1*S1',2) / norm(F1*G1',2);
    if enrichDiff < enrichTol
        disp("Enrichment finished with " + string(n)+" terms")
        break
    end

end



%% Plotting of the results
% Space-time plot of the solution u(x,t)
result1 = (F1*G1')';
result2 = (F2*G2')';
figure()
hold on
subplot(1,2,1)
surf(x1,t,result1, 'EdgeColor', 'none')
xlabel("x_1")
ylabel("t")
zlabel("u_1")
title("BEAM 1: PGD result")
view(50, 22)
subplot(1,2,2)
surf(x2,t,result2, 'EdgeColor', 'none')
xlabel("x_2")
ylabel("t")
zlabel("u_2")
title("BEAM 2: PGD result")
view(50, 22)

% Overlayed displacement of both beam ends
figure()
plot(t, result1(:,end), t, result2(:,end))
xlabel("t")
ylabel("u(x=L,t)")
legend("Beam 1", "Beam 2")
title("Displacement end of beam")

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
if a == 0 && b == 0
    t = tSpan;
    x = zeros(size(tSpan));
    return
end
[t,x] = ode78(@ode, tSpan, [x0, dx0]);
end


function [t, x] = solve4thBVP(a, b, F, tSpan)
% Solve a 4th Order BVP of the form:
% a x'''' + b x = F(t)

    function yPrime = odefun(t,y)
        Fi = linearInterpolation(tSpan, F, t);
        yPrime = [y(2); y(3); y(4); -(b/a)*y(1) + (1/a)*Fi];
    end
if a == 0 && b == 0
    t = tSpan;
    x = zeros(size(F))';
    return
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

function x_int = linearInterpolation(t, x, t_int)

    % Find the interval containing t_int
    idx = find(t <= t_int, 1, 'last');
    
    % If t_int exactly matches a value in t, return the corresponding x
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