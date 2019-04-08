
f_path = 'F:/demo/py/pydemo/data/baboon.bmp';

fprintf('the value of pi is %s\n', f_path);

img_raw = imread(f_path);


img_raw2 = img_raw(:,:,2);
fprintf('0 image data %d %d %d %d %d %d\n', img_raw2(1, 1), img_raw2(1, 2), img_raw2(1, 3), img_raw2(1, 4), img_raw2(1, 5), img_raw2(1, 6));
fprintf('0 image data %d %d %d %d %d %d\n', img_raw2(2, 1), img_raw2(2, 2), img_raw2(2, 3), img_raw2(2, 4), img_raw2(2, 5), img_raw2(2, 6));

if size(img_raw, 3) == 3
  img_raw = rgb2ycbcr(img_raw);
  img_raw = img_raw(:,:,1);
end

fprintf('image data %d %d %d %d %d %d\n', img_raw(1, 1), img_raw(1, 2), img_raw(1, 3), img_raw(1, 4), img_raw(1, 5), img_raw(1, 6));
fprintf('image data %d %d %d %d %d %d\n', img_raw(2, 1), img_raw(2, 2), img_raw(2, 3), img_raw(2, 4), img_raw(2, 5), img_raw(2, 6));

img_raw = im2double(img_raw);

img_size = size(img_raw);
width = img_size(2);
height = img_size(1);

fprintf('width : %d  height : %d\n', width, height);

img_raw = img_raw(1:height-mod(height,12), 1:width-mod(width,12), :);
img_size = size(img_raw);
fprintf('width : %d  height : %d\n', img_size(2), img_size(1));

save('F:/demo/py/pydemo/data/baboon.mat', 'img_raw');