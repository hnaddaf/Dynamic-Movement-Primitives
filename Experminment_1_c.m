% System:
k=100;
c=0.1;
m=1;
A=[0 1; -k/m -c/m];
B=[0;1/m];
T=0.01;
time=5;
%Get path (position, time)
[t, q] = GetLinearPath(time);

q=q';
%Calculate the position, velocity, and actions using the system:
q_dot=zeros(length(q),1);
q_ddot=zeros(length(q),1);
for n = 1:1:(length(q)-1)
     q_dot(n)=(q(n+1)-q(n))/T;
end

for n = 1:1:(length(q)-1)
     q_ddot(n)=(q_dot(n+1)-q_dot(n))/T;
end




% Set up data
dt = 0.01;  % time step
N = time/dt;  % number of time steps
t = dt * (0:N-1);  % time vector

y0 = 0;  % initial state
g = q(length(q));  % goal state

% Parameters for the demonstration trajectory
omega = 10;
A = 0.5;
ph = -0.5 * pi;
yd = q';  % demonstration trajectory
yddot = q_dot';
ydddot = q_ddot';

% Run the DMP learning and plot the results
run_dmp(y0, g, yd, yddot, ydddot, 20, [], 0.5, 1, dt, N);


function run_dmp(y0, g, yd, yddot, ydddot, alphaz, betaz, alphax, tau, dt, N)
    if isempty(betaz)
        betaz = alphaz / 4;
    end

    % Time vector
    t = dt * (0:N-1);

    % Basis functions
    c = 0:0.1:1;
    sigma = 0.05 * ones(size(c));

    % Learn the parameters
    w = learn(t, yd, yddot, ydddot, alphaz, betaz, alphax, tau, g, y0, dt, N, c, sigma);

    % Set up the forcing term function
    f = @(x) predict(x, w, c, sigma);

    % Reproduce the demonstration via the DMP
    [y, ~, ~] = dmp(y0, alphaz, betaz, g, tau, alphax, f, dt, N);

    % Plot the results
    figure;
    plot(t, yd, 'DisplayName', 'Demonstration');
    hold on;
    plot(t, y, '--', 'DisplayName', 'DMP Reproduction');
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 14);
    legend;
    title('DMP Learning from Demonstration', 'FontSize', 16);
    grid on;
end

function [y, z, x] = dmp(y0, alphaz, betaz, g, tau, alphax, f, dt, N)
    z = zeros(1, N);
    y = zeros(1, N);
    x = zeros(1, N);
    y(1) = y0;
    x(1) = 1;
    for n = 1:N-1
        x(n+1) = x(n) + dt * (-alphax / tau) * x(n);
        w = f(x(n));
        z(n+1) = z(n) + dt * ((alphaz / tau) * (betaz * (g - y(n)) - z(n)) + w * x(n) * (g - y0));
        y(n+1) = y(n) + dt * (z(n) / tau);
    end
end

function phi = phi_b(x, sigma, c)
    phi = exp(-((x - c).^2) / (2 * sigma^2));
end

function w = learn(t, yd, yddot, ydddot, alphaz, betaz, alphax, tau, g, y0, dt, N, c, sigma)
    x = zeros(1, N);
    x(1) = 1;
    for n = 1:N-1
        x(n+1) = x(n) + dt * (-alphax / tau) * x(n);
    end
    s = x * (g - y0);
    ft = (tau^2) * ydddot - alphaz * (betaz * (g - yd) - tau * yddot);
    w = zeros(1, length(c));
    for i = 1:length(c)
        phi = arrayfun(@(n) phi_b(x(n), sigma(i), c(i)), 1:N);
        Gamma = diag(phi);
        w(i) = (s * Gamma * ft') / (s * Gamma * s');
    end
end

function f = predict(x, w, c, sigma)
    phi = arrayfun(@(ci, si) phi_b(x, si, ci), c, sigma);
    f = (phi * w') / sum(phi);
end


