addpath('features/');

scaleImage = 6;
imageRgbOriginal = imread('datasets/selection/active_vision/rgb/000210000010101.jpg');
rgb = imresize(imageRgbOriginal, size(imageRgbOriginal(:,:,1))/scaleImage); 
imageDepthOriginal = imread('datasets/selection/active_vision/depth/000210000010101.png');
depth = imresize(imageDepthOriginal, size(imageDepthOriginal(:,:,1))/scaleImage);

%seg = imresize(imread('datasets/selection/active_vision/gt/000210000010101.png'), [270 480]);
%depthDouble=double(depth)*100000;
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
%rgbd_data = rmfield(rgbd_data,'imgNormals');
save('teste.mat', 'rgbd_data');
imwrite(normal, strcat('normals_m', num2str(multiplicador),'.png'));

pcwrite(ptCloudSingle,'teste.pcd')