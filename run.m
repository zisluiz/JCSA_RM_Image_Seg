clear all; close all; clc;
addpath('rgbd/')
addpath('features/')

if ~exist('results', 'dir')
   mkdir('results')
end

fid = fopen(strcat('results/run_', num2str(posixtime(datetime('now')) * 1e6), '.txt'), 'wt');
fprintf(fid, '=== Start time: %s\n', datestr(datetime('now')));
MethodType = 'rb_jcsa_rm'; % 'rb_jcsd_rm'; 'rb_jcsa_rm'; % 'rb_jcsa'; % 'rb_jcsd'; 
scaleImage = 2;
%method=rb_jcsa_rm
%1-5,1-1,0.5-5,1.5-5,1-3
%1-1.5,0.5-3,1-5,0.5-2, 1-2
multiplicador = 1;
multiplicador2 = 2;

% Load options and the threshold values
opt.sc = 1; %1,2 ou 4
opt.kMax = 20 * multiplicador;
opt.showLLH = 0;
opt.showIt = 0;
opt.numiter = 20 * multiplicador;

thOptions.thDivNormalMax = ceil(2 * multiplicador2);
thOptions.thDivNormalMin = ceil(1 * multiplicador2);
thOptions.planarityTh = 0.9 * multiplicador2;
thOptions.thKappa = ceil(5 * multiplicador2);
thOptions.edgeStrengthTh = 0.2 * multiplicador2; 

ShowImages = false;

datasetdir = 'datasets/selection';
images = [dir(fullfile(datasetdir,'**/rgb/*.jpg')); dir(fullfile(datasetdir,'**/rgb/*.png'))];
images = images(~[images.isdir]);  %remove folders from list
filesCount = 0;
cpuTimes = [0.0, 0.0, 0.0, 0.0];

gpuTimes = 0.0;
gpuMemTimes = 0.0;
maxNumThreads = 0;
memUsageTimes = 0;

tStart = datetime('now');

pid = int16(feature('getpid'));

%parpool(4);
%parfor k = 1:length(images), 4
for k = 1:length(images)
    if k <= 40
        continue;
    end
    clearvars -except pid fid tStart filesCount ShowImages opt thOptions images k scaleImage MethodType methods i allInfo rgbd_data rgb normal pcloud imageName ptCloudSingle multiplicador multiplicadores multiplicador2 multiplicadores2; 
    imageRgb = images(k);
    display(strcat('Processing image ', imageRgb.name));
    
    imageRgbFolders = strsplit(imageRgb.folder, '/');
    datasetName = imageRgbFolders{end-1};
    
    datasetPath = strjoin(imageRgbFolders(1:end-1), '/');
    depthFileName = strrep(imageRgb.name, 'jpg', 'png');
    imageName = strrep(imageRgb.name, '.png', '');
    
    % Load image and other information
    imageRgbOriginal = imread(strcat(imageRgb.folder, '/', imageRgb.name));
    imageDepthOriginal = imread(strcat(datasetPath, '/depth/', depthFileName));

    if datasetName == "active_vision" || datasetName == "putkk"
        imageRgbOriginal = imcrop(imageRgbOriginal, [240 1 1439 1080]);
        imageDepthOriginal = imcrop(imageDepthOriginal, [240 1 1439 1080]);
    end

    rgb = imresize(imageRgbOriginal, size(imageRgbOriginal(:,:,1))/scaleImage, 'nearest'); 
    depth = imresize(imageDepthOriginal, size(imageDepthOriginal(:,:,1))/scaleImage, 'nearest');
        
    %depthDouble=double(depth)/1000;
    depthDouble=im2double(depth);
        
    if datasetName == "active_vision" || datasetName == "putkk"
        topleft = [0 0];
        center = [953 531];
        focal = 1078.7;    
    else
        topleft = [0 0];
        center = [540 540];
        focal = 759.681;    
    end

    [pcloud, distance] = DepthtoCloud(depthDouble, topleft, center, focal);    
    ptCloud = pointCloud(pcloud);
    ptCloudSingle = pointCloud(single(ptCloud.Location),...
                               'Color',ptCloud.Color,...
                               'Normal',ptCloud.Normal,...
                               'Intensity',ptCloud.Intensity);
    
    pcloud(isnan(pcloud)) = 0;
    normal=pcnormal(pcloud,0.05,8);
    normal=fix_normal_orientation( normal, pcloud );

    rgbd_data.rgbImg = rgb;
    rgbd_data.depImg = depthDouble;
    rgbd_data.imgNormals = normal;
    rgbd_data.pcloud = pcloud;
    
    allInfo = rgbd_data;

    img = fnProcessImages(opt, thOptions, rgbd_data.rgbImg, rgbd_data.depImg, rgbd_data.imgNormals, allInfo, ShowImages, MethodType);
    
    [status,cmdout] = unix(strcat('top -n 1 -p ', num2str(pid)));

    fprintf(fid, "%s\n", cmdout);
    
    segres = label2rgb(assignRandomLabel(img));

    if (ShowImages)
        display('Displaying segmentation results ...');    
        subplot(2, 3, [4:6]); imshow(segres); title('Segmented Image');
    end

    if ~exist(strcat('results/', datasetName), 'dir')
       mkdir(strcat('results/', datasetName));
    end    
    
    imwrite(segres, strcat('results/', datasetName, '/', depthFileName));
    %imwrite(normal, strcat('results/', datasetName, '/', imageName, '_normals.png'));
    %pcwrite(ptCloudSingle, strcat('results/', datasetName, '/', imageName, '.pcd'))
    filesCount = filesCount + 1;

    display(strcat('Processed image ', num2str(k)));
end

tEnd = datetime('now');
tElapsed = between(tStart, tEnd);

fprintf(fid, "=== Total image predicted: %d\n", filesCount);
fprintf(fid, "=== Seconds per image: %d\n", (seconds(time(tElapsed)) / filesCount));

fprintf(fid, '=== End time: %s\n', datestr(datetime('now')));
fclose(fid);
