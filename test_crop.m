scaleImage = 2;
imageRgbOriginal = imread('datasets/selection/active_vision/rgb/000210000010101.jpg');
imageDepthOriginal = imread('datasets/selection/active_vision/depth/000210000010101.png');

%1440	1080
imageRgb = imcrop(imageRgbOriginal, [240 1 1439 1080]);
imageDepth = imcrop(imageDepthOriginal, [240 1 1439 1080]);

rgb = imresize(imageRgb, size(imageRgb(:,:,1))/scaleImage, 'nearest'); 
depth = imresize(imageDepth, size(imageDepth(:,:,1))/scaleImage, 'nearest');

imwrite(imageRgbOriginal, strcat('rgb_original.png'));
imwrite(imageRgb, strcat('rgb_cropped.png'));
imwrite(rgb, strcat('rgb_cropped_resized.png'));
imwrite(depth, strcat('d_cropped_resized.png'));
