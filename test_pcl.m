addpath('features/');


imageRgbOriginal = imread('datasets/selection/active_vision/rgb/000210000010101.jpg');
imageDepthOriginal = imread('datasets/selection/active_vision/depth/000210000010101.png');
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
                  
imwrite(imageRgbOriginal, strcat('rgb_pcl_1.png'));
pcwrite(ptCloudSingle,'pcl1.pcd');

imageRgbOriginal = imread('datasets/selection/semantics3d_raw/rgb/4a7bfe0577f74a1a891683cf5b435f93_3.jpg');
imageDepthOriginal = imread('datasets/selection/semantics3d_raw/depth/4a7bfe0577f74a1a891683cf5b435f93_3.png');
depthDouble=im2double(depth);

%1073.45 1074.07 628.659 522.306
topleft = [0 0];
center = [628.659 522.306];
focal = 1073.45;    

[pcloud, distance] = DepthtoCloud(depthDouble, topleft, center, focal);
ptCloud = pointCloud(pcloud);
ptCloudSingle = pointCloud(single(ptCloud.Location),...
                           'Color',ptCloud.Color,...
                           'Normal',ptCloud.Normal,...
                           'Intensity',ptCloud.Intensity);
                  
imwrite(imageRgbOriginal, strcat('rgb_pcl_2.png'));
pcwrite(ptCloudSingle,'pcl2.pcd');
