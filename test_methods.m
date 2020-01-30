clear all; close all; clc;
addpath('rgbd/');
addpath('features/');

scaleImage = 2;
datasetName = "semantics3d_raw";%"active_vision";
imageName = "4a7bfe0577f74a1a891683cf5b435f93_52";%'000210000020101';
imageRgbOriginal = imread(strcat('datasets/selection/',datasetName,'/rgb/',imageName,'.png'));
imageDepthOriginal = imread(strcat('datasets/selection/',datasetName,'/depth/',imageName,'.png'));

if datasetName == "active_vision" || datasetName == "putkk"
    imageRgbOriginal = imcrop(imageRgbOriginal, [420 1 1079 1080]);
    imageDepthOriginal = imcrop(imageDepthOriginal, [420 1 1079 1080]);
end

rgb = imresize(imageRgbOriginal, size(imageRgbOriginal(:,:,1))/scaleImage); 

depth = imresize(imageDepthOriginal, size(imageDepthOriginal(:,:,1))/scaleImage);

depthDouble=im2double(depth);

topleft = [1 1];
center = [952.6592286 530.7386644];
focal = 1078.68499;    

[pcloud, distance] = DepthtoCloud(depthDouble, topleft, center, focal);
ptCloud = pointCloud(pcloud);
ptCloudSingle = pointCloud(single(ptCloud.Location),...
                           'Color',ptCloud.Color,...
                           'Normal',ptCloud.Normal,...
                           'Intensity',ptCloud.Intensity);
                       
pcloud(isnan(pcloud)) = 0;
multiplicador = 1;
normal=pcnormal(pcloud,0.05*multiplicador,ceil(8*multiplicador)); %0.05,8
normal=fix_normal_orientation( normal, pcloud );

rgbd_data.rgbImg = rgb;
rgbd_data.depImg = im2double(depth);
rgbd_data.imgNormals = normal;
rgbd_data.pcloud = pcloud;
allInfo = rgbd_data;

methods = ["rb_jcsa_rm"];

multiplicadores = [0.5,1,1.5];
multiplicadores2 = [0.5,1,1.5,2,3,4,5];

%multiplicadores = [3,4];

for i=1:length(multiplicadores)
    multiplicador = multiplicadores(i);
    
    for j=1:length(multiplicadores2)
        multiplicador2 = multiplicadores2(j);    
    
        display(strcat('Processing image ', num2str(i))); 
        MethodType = "rb_jcsa_rm"; % 'rb_jcsa_rm'; % 'rb_jcsa'; % 'rb_jcsd';     

        clearvars -except j scales MethodType methods i allInfo rgbd_data rgb normal pcloud imageName ptCloudSingle multiplicador multiplicadores multiplicador2 multiplicadores2; 
        scale = 1;
        % Load options and the threshold values
        opt.sc = scale; %1,2 ou 4
        opt.kMax = 20 * multiplicador;
        opt.showLLH = 0;
        opt.showIt = 0;
        opt.numiter = 20 * multiplicador;

        thOptions.thDivNormalMax = ceil(2 * multiplicador2);
        thOptions.thDivNormalMin = ceil(1 * multiplicador2);
        thOptions.planarityTh = 0.9 * multiplicador2;
        thOptions.thKappa = ceil(5 * multiplicador2);
        thOptions.edgeStrengthTh = 0.2 * multiplicador2;    

        img = fnProcessImages(opt, thOptions, rgbd_data.rgbImg, rgbd_data.depImg, rgbd_data.imgNormals, allInfo, false, MethodType);

        segres = label2rgb(assignRandomLabel(img));

        if ~exist(strcat('tests/'), 'dir')
           mkdir(strcat('tests/'));
        end    

        imwrite(segres, strcat('tests/', imageName, '_mult_opt_', num2str(multiplicador), '_mult_thopt_', num2str(multiplicador2), '.png'));

        display(strcat('Processed image i=', num2str(i), ' j=', num2str(j))); 
    end
end

imwrite(rgb, strcat('tests/', imageName, '_rgb.png'));
imwrite(normal, strcat('tests/', imageName, '_normals.png'));
pcwrite(ptCloudSingle, strcat('tests/', imageName, '.pcd'));