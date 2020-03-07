%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  16833 Robot Localization and Mapping  % 
%  Assignment #2                         %
%  EKF-SLAM                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

%==== TEST: Setup uncertainity parameters (try different values!) ===
sig_x = 0.25;
sig_y = 0.1;
sig_alpha = 0.1;
sig_beta = 0.01;
sig_r = 0.08;

%==== Generate sigma^2 from sigma ===
sig_x2 = sig_x^2;
sig_y2 = sig_y^2;
sig_alpha2 = sig_alpha^2;
sig_beta2 = sig_beta^2;
sig_r2 = sig_r^2;

%==== Open data file ====
fid = fopen('../data/data.txt');

%==== Read first measurement data ====
tline = fgets(fid);
arr = str2num(tline);
measure = arr';
t = 1;
 
%==== Setup control and measurement covariances ===
control_cov = diag([sig_x2, sig_y2, sig_alpha2]);
measure_cov = diag([sig_beta2, sig_r2]);

%==== Setup initial pose vector and pose uncertainty ====
pose = [0 ; 0 ; 0];
pose_cov = diag([0.02^2, 0.02^2, 0.1^2]);

%==== TODO: Setup initial landmark vector landmark[] and covariance matrix landmark_cov[] ====
%==== (Hint: use initial pose with uncertainty and first measurement) ====

% Write your code here...
% num of landmarks
k = size(measure, 1) / 2;
% landmark poses in global frame
landmark = zeros(2*k, 1);
landmark_cov = zeros(2*k, 2*k);
for i = 0 : (k-1)
    beta = measure(2*i+1);
    r = measure(2*i+2);
    x = pose(1);
    y = pose(2);
    theta = pose(3);
    % calculate landmark position in global frame use robot pose and
    % measurement
    landmark(2*i+1) = x + r * cos(theta + beta); % lx
    landmark(2*i+2) = y + r * sin(theta + beta); % ly
    % calculate covariance 
    % denote l_t = g(p_t, z_t)
    % where p_t = robot pose
    %       z_t = measurement beta and r
    
    % H_t = partial(g) / partial(p_t)
    H = [1  0  -r * sin(theta + beta);
         0  1  r * cos(theta + beta)];
    % M_t = partial(g) / partial(z_t)
    M = [-r * sin(theta + beta)  cos(theta + beta);
          r * cos(theta + beta)  sin(theta + beta)];
    % landmark_cov is a 12*12 matrix, however different landmarks are
    % independent, we update the "diagonal" submatrices one by one
    landmark_cov(2*i+1:2*i+2, 2*i+1:2*i+2) = H * pose_cov * H' + M * measure_cov * M';
end
%==== Setup state vector x with pose and landmark vector ====
x = [pose ; landmark];

%==== Setup covariance matrix P with pose and landmark covariances ====
P = [pose_cov zeros(3, 2*k) ; zeros(2*k, 3) landmark_cov];

%==== Plot initial state and conariance ====
last_x = x;
drawTrajAndMap(x, last_x, P, 0);

%==== Read control data ====
tline = fgets(fid);
while ischar(tline)
    arr = str2num(tline);
    d = arr(1);
    alpha = arr(2);
    
    %==== TODO: Predict Step ====
    %==== (Notice: predict state x_pre[] and covariance P_pre[] using input control data and control_cov[]) ====
    
    % Write your code here...
    x_pre = x;
    P_pre = P;
    
    x_pre(1) = x(1) + d * cos(x(3));
    x_pre(2) = x(2) + d * sin(x(3));
    x_pre(3) = x(3) + alpha;
    
    % denote p_t = f(p_t-1, u_t-1, w_t-1)
    % where w is process noise e_x, e_y and e_alpha
    
    % partial(f) / partial(p)
    F_t = [1  0  -d * cos(x(3));
           0  1   d * sin(x(3));
           0  0   1];
       
    % partial(f) / partial(w)
    L_t = [cos(x(3))  -sin(x(3))  0;
           sin(x(3))   cos(x(3))  0;
           0           0          1];
       
    % update covariance of robot pose only
    P_pre(1:3, 1:3) = F_t * pose_cov * F_t' + L_t * control_cov * L_t';
    
    %==== Draw predicted state x_pre[] and covariance P_pre[] ====
    drawTrajPre(x_pre, P_pre);
    
    %==== Read measurement data ====
    tline = fgets(fid);
    arr = str2num(tline);
    measure = arr';
    
    %==== TODO: Update Step ====
    %==== (Notice: update state x[] and covariance P[] using input measurement data and measure_cov[]) ====
    
    % Write your code here...
    
    % we incrementally update for each landmark
    % in each step, we update robot pose and the corresponding landmark
    % pose
    for i = 0:(k-1)
        beta_t = measure(2*i+1);
        r_t = measure(2*i+2);
        
        lx_est = x_pre(3+2*i+1);
        ly_est = x_pre(3+2*i+2);
        dx = lx_est - x_pre(1);
        dy = ly_est - x_pre(2);
        dxdy2 = dx^2 + dy^2;
        dxdy2sqrt = sqrt(dxdy2);
        
        beta_est = wrapToPi(atan2(dy, dx) - x_pre(3));
        r_est = dxdy2sqrt;

        residual = [beta_t; r_t] - [beta_est; r_est];
        
        H_t = zeros(2, 3+2*k);
        H_t(1:2, 1:3) = [dy/dxdy2        -dx/dxdy2      -1;
                         -dx/dxdy2sqrt   -dy/dxdy2sqrt   0];
        H_t(1:2, 3+2*i+1:3+2*i+2) = [-dy/dxdy2      dx/dxdy2;
                                      dx/dxdy2sqrt  dy/dxdy2sqrt];
        
        S_t = H_t * P_pre * H_t' + measure_cov;
        K_t = P_pre * H_t' * inv(S_t);
        x_pre = x_pre + K_t * residual;
        P_pre = (eye(3+2*k) - K_t * H_t) * P_pre;

    end
    
    x = x_pre;
    P = P_pre;
    
    %==== Plot ====   
    drawTrajAndMap(x, last_x, P, t);
    last_x = x;
    
    %==== Iteration & read next control data ===
    t = t + 1;
    tline = fgets(fid);
end

%==== EVAL: Plot ground truth landmarks ====

% Write your code here...
landmark_ground_truth = [3 6 3 12 7 8 7 14 11 6 11 12];
hold on;
for i = 0:(k-1)
    plot(landmark_ground_truth(i*2+1), landmark_ground_truth(i*2+2), '*b');
end



%==== Close data file ====
fclose(fid);
