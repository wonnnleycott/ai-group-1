addpath('provided')

%% All angles at 0 - the starting pose
a = zeros(1,24);
plot_spider_pose(a)
title('All angles = 0')
pause(2)

%% a(1) = coxa of leg L1
% Observation: L1 moved slightly - the coxa swings the whole leg around the body.
a = zeros(1,24);
a(1) = 0.5;
plot_spider_pose(a)
title('a(1) = 0.5  (L1 coxa)')
pause(2)

%% a(2) = femur of leg L1
% Observation: L1 moved back down - the femur tilts the leg up/down.
a = zeros(1,24);
a(2) = 0.5;
plot_spider_pose(a)
title('a(2) = 0.5  (L1 femur)')
pause(2)

%% (3) = tibia of leg L1
% Observation: the red part of L1 angled slightly - the tibia bends the lower leg.
a = zeros(1,24);
a(3) = 0.5;
plot_spider_pose(a)
title('a(3) = 0.5  (L1 tibia)')
pause(2)

%% Negative numbers move the joint the opposite way
a = zeros(1,24);
a(2) = -0.5;
plot_spider_pose(a)
title('a(2) = -0.5  (L1 femur, opposite direction)')

%% Conclusion
% Every leg works the same way (a(4:6) = L2, a(7:9) = L3, ...).
% To walk, each leg needs to swing forwards/backwards (coxa) and
% lift/lower (femur + tibia), at different times from the other
% legs so the spider stays balanced. This led to the sine-wave
% chromosome idea: each joint = offset + amplitude * sin(time + phase).