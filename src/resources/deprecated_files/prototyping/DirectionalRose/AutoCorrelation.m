
addpath('~/Documents/HandWritingRecognition/Matlab prototyping/')

% import image and throw away channels
image = imread('data/7.png');
image = image(:,:,1);
[height, width] = size(image);

% perform AutoCorrelation R = FFT-1(FFT(image)*FFTconj(image))
fftimage = fft2(image);
corr = fftimage .* conj(fftimage);
corr = ifft2(corr);
corr = abs(corr);
corr = fftshift(corr);

minV = min(corr);
maxV = max(corr);
corr = (corr-minV)./(maxV-minV);

%% New Line finder

f = 10;
dim = height;

allCorrSums = [];
for angle = 1:180
    
    values = [];
    
    lut = zeros(dim, dim);
    
    rad = angle * pi / 180;
    slope = cos(rad)/sin(rad);

    for x = 1:dim*f
       lineY = slope * (x-(dim*f/2));
       for y = 1:dim*f

           offsetY = y - (dim*f/2);
           offsetX = x - (dim*f/2);

           if lineY >= (y-(dim*f/2)) && lineY <= (y-(dim*f/2)+1)
              foundX = ceil(x/f);
              foundY = ceil(y/f);
              if lut(foundY, foundX) == 0
                  values = [values corr(foundY,foundX)];
                  lut(foundY, foundX) = 1;
              end
              
           end
       end
    end
    
    normalized = (sum(values)-min(corr))/(max(corr)-min(corr));
    allCorrSums = [allCorrSums normalized];
end

allCorrSums
%% How many directions do we want?

directions = 8;
[value, i] = maxk(allCorrSums, directions);
% sort(i)

% data = [value; i]';
% [idx, C] = kmeans(data, 8);

% figure;
% scatter(value,i);
% hold on
% scatter(C(:,1), C(:,2), 'kx');
 
%% plot each significant direction on the image

% bestDirections = round(C(:,2))
% bestDirections = [148, 156, 157, 158, 161, 162, 163, 165];
bestDirections = i;
for k = 1:length(bestDirections)

%     F = abs(180-bestDirections(k));
    F = bestDirections(k);
    slope = cos(F * pi / 180)/sin(F * pi / 180);

    for x = 1:dim*f
       lineY = slope * (x-(dim*f/2));
       for y = 1:dim*f

           offsetY = y - (dim*f/2);
           offsetX = x - (dim*f/2);

           if lineY >= (y-(dim*f/2)) && lineY <= (y-(dim*f/2)+s)
              foundX = ceil(x/f);
              foundY = ceil(y/f);
              image(foundY, foundX) = 200;
           end
       end
    end
end
   
imshow(image)