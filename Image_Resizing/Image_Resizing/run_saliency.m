
function run_saliency(file_name)
%% A script for running saliency computation
%%clear all;
%%close all;

%% load parameters and images
addpath('Saliency');

%%file_names{1} = 'bird.jpg';
%%file_names{2} = '003.jpg';
% file_names{1} = 'butterfly.jpg';
file_names{1} = file_name;
% disp(file_names);

saliency(file_names);
exit();

% MOV = saliency(file_names);

%% display results
% N = length(MOV);
% for i=1:N
%     figure(i); clf;
%     subplot(1,2,1); imshow(MOV{i}.Irgb); title('Input','fontsize',16);
%     subplot(1,2,2); imshow(MOV{i}.SaliencyMap); title('Saliency map','fontsize',16);
% end    
    
    
    