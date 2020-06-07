
addpath('~/Library/Mobile Documents/com~apple~CloudDocs/iCloud/Study/University/6-4 Handwriting Recognition/Project/HandWritingRecognition/CharacterIdentification/')

% import image and throw away channels
image = imread('Test7.tiff');
image = image(:,:,1);
[height, width] = size(image);

% perform AutoCorrelation R = FFT-1(FFT(image)*FFTconj(image))
fftimage = fft2(image);
congfft = conj(fftimage); %conjugate
ans1 = fftimage*congfft;
corr = fftshift(abs(ifft2(ans1)));

%% New Line finder

f = 100;
dim = height;
s = 1;

allCorrSums = [];
for angle = 1:179
    
    values = [];
    
    lut = zeros(dim, dim);
    
    rad = angle * pi / 180;
    slope = cos(rad)/sin(rad);

    for x = 1:dim*f
       lineY = slope * (x-(dim*f/2));
       for y = 1:dim*f

           offsetY = y - (dim*f/2);
           offsetX = x - (dim*f/2);

           if lineY >= (y-(dim*f/2)) && lineY <= (y-(dim*f/2)+s)
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

%% How many directions do we want?

directions = 32;
[value, i] = maxk(allCorrSums, directions);

data = [value; i]';
[idx, C] = kmeans(data, 8);

% figure;
% scatter(value,i);
% hold on
% scatter(C(:,1), C(:,2), 'kx');

% plot each significant direction on the image
bestDirections = round(C(:,2))
for k = 1:length(bestDirections)

    F = abs(180-bestDirections(k));
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