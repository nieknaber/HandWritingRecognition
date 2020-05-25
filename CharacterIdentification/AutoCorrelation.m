
addpath('~/Library/Mobile Documents/com~apple~CloudDocs/iCloud/Study/University/6-4 Handwriting Recognition/Project/HandWritingRecognition/CharacterIdentification/')

% import image and throw away channels
image = imread('Test9.tiff');
image = image(:,:,1);
[height, width] = size(image);

% perform AutoCorrelation R = FFT-1(FFT(image)*FFTconj(image))
fftimage = fft2(image);
congfft = conj(fftimage); %conjugate
ans1 = fftimage*congfft;
corr = fftshift(abs(ifft2(ans1)));

% sum the correlation along a line
allCorrSums = [];
for deg = 1:180
    slope = cos(deg * pi / 180)/sin(deg * pi / 180);
    values = [];
    for x = 1:width
        y = slope * (x-width/2);
        if y > -width/2
            if y < width/2
                rounded = round(y+width/2);
                if rounded == 0
                    rounded = 1;
                end
                values = [values corr(rounded,x)];
            end
        end
    end
    normalized = (sum(values)-min(corr))/(max(corr)-min(corr));
    allCorrSums = [allCorrSums normalized];
end

%% How many directions do we want?

directions = 16;
[value, i] = maxk(allCorrSums, directions)

% plot each significant direction on the image
for k = 1:directions

    F = abs(180-i(k));
    slope = cos(F * pi / 180)/sin(F * pi / 180);

    for x = 1:width
        y = s * (x-width/2);
        if y > -width/2
            if y < width/2
                rounded = round(y+width/2);
                image(rounded,x) = 200; % grey color
            end
        end
    end
end
   
imshow(image)