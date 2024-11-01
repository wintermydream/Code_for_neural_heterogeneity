classdef Fun_FixNet < handle
    methods (Static = true)

        function [Rfilter_train]=gaussian_filter(filter_sigma,dt1,sx_train)
            %{
                Function:
                    gaussian filter for 1-dimentional time series
                Input:
                    filter_sigma: std of gaussial fiulter
                    
            %}
            bin=2; % gaussian filter bin :ms
            taus=-bin*filter_sigma:dt1:bin*filter_sigma;
            Gfilter=exp(-taus.^2./(2*filter_sigma^2));
            Gfilter=Gfilter./sum(Gfilter(:));
            Rfilter_train = zeros(size(sx_train));
            for i = 1:size(sx_train,2)
                Rfilter_train(:,i) = conv(sx_train(:,i),Gfilter,'same');
            end
        end

        function [Rfilter_train] = GaussianSmooth(filter_sigma, filter_bin, dt1, rateSeries)
            %{
        Function:
            gaussian smoothing for time series: Trange x NumNeuron
        Input:
            filter_sigma: std of gaussian filter
            filter_bin: width of filter window
            dt1: temporal resolution
            rateSeries: firing rate series
            %}

            % Ensure rateSeries is a column vector if it's a 1D array
            if isvector(rateSeries)
                rateSeries = rateSeries(:);
            end

            % Gaussian windows
            taus = -filter_bin*filter_sigma:dt1:filter_bin*filter_sigma;

            % Gaussian kernel
            Gfilter = exp(-taus.^2./(2*filter_sigma^2));
            Gfilter = Gfilter ./ sum(Gfilter(:));

            % Initialize Rfilter_train
            Rfilter_train = zeros(size(rateSeries));

            % Smoothed firing rate series
            for i = 1:size(rateSeries, 2)
                Rfilter_train(:, i) = conv(rateSeries(:, i), Gfilter, 'same');
            end
        end


        function [Co_idx,sigma_R,mu_R] = coherence_index(spike,t_bin,dt)
            % cite: Wang, Xiao-Jing. "Pacemaker neurons for the theta rhythm and their
            % synchronization in the septohippocampal reciprocal loop." Journal
            % of Neurophysiology 87.2 (2002): 889-900.
            % clear,clc,close all
            % load 'spike.mat'; % 2-d spiking data with N neurons and t_length time steps
            % t_bin = 2; dt = 0.1;
            t_length = size(spike,2)-t_bin/dt; N0 = size(spike,1);
            R = zeros(t_length,1);
            for t = 1:t_length
                R(t) = sum(sum(spike(:,t:t+t_bin/dt)))/N0/t_bin;
            end
            mu_R = mean(R);
            sigma_R = sqrt(mean((R-mu_R).^2));
            Co_idx = sigma_R/mu_R;
        end


        function parsave(parsave_file_name, varargin)
            %{
            Function:
                注意varargin直接赋值变量名，不用赋值为变量名的字符串
                使用save函数save(path,x) x应赋值为变量名的字符串
            %}
            var_names={};
            for i=1:length(varargin)
                eval(sprintf('%s=varargin{%d};',inputname(i+1),i));
                var_names{i}=inputname(i+1);
            end
            save(parsave_file_name,var_names{:});
        end

        function images = loadMNISTImages(filename)
            %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
            %the raw MNIST image
            %imshow image: imshow(reshape(images(:,1 ), 28, 28))

            fp = fopen(filename, 'rb');
            assert(fp ~= -1, ['Could not open ', filename, '']);

            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2051, ['Bad magic number in ', filename, '']);

            numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
            numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
            numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

            images = fread(fp, inf, 'unsigned char');
            images = reshape(images, numCols, numRows, numImages);
            images = permute(images, [2, 1, 3]);

            fclose(fp);

            % Reshape to #pixels x #examples
            images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
            % Convert to double and rescale to [0, 1]
            images = double(images) / 255;

        end

        function labels = loadMNISTLabels(filename)
            %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
            %the labels for the MNIST images

            fp = fopen(filename, 'rb');
            assert(fp ~= -1, ['Could not open ', filename, '']);

            magic = fread(fp, 1, 'int32', 0, 'ieee-be');
            assert(magic == 2049, ['Bad magic number in ', filename, '']);

            numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

            labels = fread(fp, inf, 'unsigned char');

            assert(size(labels, 1) == numLabels, 'Mismatch in label count');

            fclose(fp);

        end

        function datasetTrainTest = GenDataBySVD(ImagesSamples, SVDRank)
            % Function:
            %   decompose 2-D image data by using SVD method, and get the U and V vectors
            % Input:
            %   ImagesSamples: 2-D image data 768 x samplesize
            %   SVDRank:  maximun number of  selected eigenvalues of SVD
            % Output:
            %   datasetTrainTest:  2*SVDRank x ImageSize matrix
            SamplesNumber = size(ImagesSamples, 2);
            ImageSize = floor(sqrt(size(ImagesSamples,1)));

            datasetTrainTest = zeros(2*SVDRank,ImageSize*SamplesNumber);

            for indexSamples = 1 : SamplesNumber
                [U,S,V] = svd(reshape(ImagesSamples(:, indexSamples), ImageSize, ImageSize));
                for i = 1 : SVDRank
                    datasetTrainTest(i*2-1, (indexSamples-1)*ImageSize+1:indexSamples*ImageSize) = U(:, i) .* sqrt(S(i, i));
                    datasetTrainTest(i*2, (indexSamples-1)*ImageSize+1:indexSamples*ImageSize) = V(:, i) .* sqrt(S(i, i));
                end
            end

        end

        function [x,t]=FunMackeyGlass(N,tau,dt)
            %Link: https://blog.csdn.net/itnerd/article/details/113852957
            % Function:
            %   Mackey-Glass混沌延迟微分方程
            %   $\dot{\mathrm{X}}(\mathrm{t})=\frac{\mathrm{aX}(\mathrm{t}-\tau)}{1+[\mathrm{X}(\mathrm{t}-\tau)]^c}-\mathrm{bX}(\mathrm{t})$
            %   a = 0.2 , b = 0.1 , c = 10 a=0.2, b=0.1,c=10a=0.2,b=0.1,c=10

            % Input:
            %   N为输出点数，tau为延迟时间
            %   dt is the time step
            % Output:
            %   x为序列返回值，t为时间返回值，h为采样间隔
            %Example:
            %   [x,t] = MackeyGlass(10000,17,0.1);
            %   plot(t,x)

            t=zeros(N,1);
            x=zeros(N,1);
            x(1)=1.2; t(1)=0;
            a=0.2;
            b=0.1;
            c=10;
            h=dt/10;

            Df = @(x) a*x./(1+x.^c);
            for k=1:N-1
                t(k+1)=t(k)+h;
                if t(k)<tau
                    k1=-b*x(k);
                    k2=-b*(x(k)+h*k1/2);
                    k3=-b*(x(k)+k2*h/2);
                    k4=-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                else
                    n=floor((t(k)-tau-t(1))/h+1);
                    k1=Df(x(n))-b*x(k);
                    k2=Df(x(n))-b*(x(k)+h*k1/2);
                    k3=Df(x(n))-b*(x(k)+2*k2*h/2);
                    k4=Df(x(n))-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                end
            end
        end

        function [x,t]=FunMG_match(StimSize,tau,dt)
            %Link: https://blog.csdn.net/itnerd/article/details/113852957
            % Function:
            %   Mackey-Glass混沌延迟微分方程
            %   $\dot{\mathrm{X}}(\mathrm{t})=\frac{\mathrm{aX}(\mathrm{t}-\tau)}{1+[\mathrm{X}(\mathrm{t}-\tau)]^c}-\mathrm{bX}(\mathrm{t})$
            %   a = 0.2 , b = 0.1 , c = 10 a=0.2, b=0.1,c=10a=0.2,b=0.1,c=10

            % Input:
            %   StimSize为输出点数，tau为延迟时间
            %   dt is the time step
            % Output:
            %   x为序列返回值，t为时间返回值，h为采样间隔
            %Example:
            %   [x,t] = MackeyGlass(10000,17,0.1);
            %   plot(t,x)

            % modification in Feb 28th 2023
            %   Consider the first value approximate to the end value in MG
            %   series
            Ntemp=StimSize*50;
            t=zeros(Ntemp,1);
            x=zeros(Ntemp,1);
            x(1)=1.2; t(1)=0;
            a=0.2;
            b=0.1;
            c=10;
            h=dt/10;

            Df = @(x) a*x./(1+x.^c);
            for k=1:Ntemp-1
                t(k+1)=t(k)+h;
                if t(k)<tau
                    k1=-b*x(k);
                    k2=-b*(x(k)+h*k1/2);
                    k3=-b*(x(k)+k2*h/2);
                    k4=-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                else
                    n=floor((t(k)-tau-t(1))/h+1);
                    k1=Df(x(n))-b*x(k);
                    k2=Df(x(n))-b*(x(k)+h*k1/2);
                    k3=Df(x(n))-b*(x(k)+2*k2*h/2);
                    k4=Df(x(n))-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                end
            end
            % add algorithm for matching of first and end
            x=zscore(x);
            error=1;
            i=1;
            sign1=1;
            sign2=-1;

            while(error>0.05||sign1~=sign2||locsign1==locsign2||min(x(i:i+StimSize-1))<-3)
                % difference of value is less than error
                error=abs(x(i)-x(i+StimSize-1));
                % same monotonicity
                sign1=sign(x(i)-x(i+1));
                sign2=sign(x(i+StimSize-1)-x(i+StimSize));
                % different sign of nearest peak
                [pks1,locs1] = findpeaks(x(i:i+StimSize-1));
                [pks2,locs2] = findpeaks(-x(i:i+StimSize-1));
                j=1;
                while(pks1(j)<1)
                    j=j+1;
                end
                k=1;
                while(pks2(k)<1)
                    k=k+1;
                end
                [locstemp,locsign1]=min([locs1(j),locs2(k)]);

                j=length(locs1);
                while(pks1(j)<1)
                    j=j-1;
                end
                k=length(locs2);
                while(pks2(k)<1)
                    k=k-1;
                end
                [locstemp,locsign2]=max([locs1(j),locs2(k)]);
                i=i+1;
            end
            x=x(i:i+StimSize-1);
        end

        function [x,t]=FunMG_match_noZS(StimSize,tau,dt)
            %Link: https://blog.csdn.net/itnerd/article/details/113852957
            % Function:
            %   Mackey-Glass混沌延迟微分方程
            %   $\dot{\mathrm{X}}(\mathrm{t})=\frac{\mathrm{aX}(\mathrm{t}-\tau)}{1+[\mathrm{X}(\mathrm{t}-\tau)]^c}-\mathrm{bX}(\mathrm{t})$
            %   a = 0.2 , b = 0.1 , c = 10 a=0.2, b=0.1,c=10a=0.2,b=0.1,c=10

            % Input:
            %   StimSize为输出点数，tau为延迟时间
            %   dt is the time step
            % Output:
            %   x为序列返回值，t为时间返回值，h为采样间隔
            %Example:
            %   [x,t] = MackeyGlass(10000,17,0.1);
            %   plot(t,x)

            % modification in Feb 28th 2023
            %   Consider the first value approximate to the end value in MG
            %   series
            % Mar 22th 2023
            %   canceling zscore of MG signal
            Ntemp=StimSize*50;
            t=zeros(Ntemp,1);
            x=zeros(Ntemp,1);
            x(1)=1.2; t(1)=0;
            a=0.2;
            b=0.1;
            c=10;
            h=dt/10;

            Df = @(x) a*x./(1+x.^c);
            for k=1:Ntemp-1
                t(k+1)=t(k)+h;
                if t(k)<tau
                    k1=-b*x(k);
                    k2=-b*(x(k)+h*k1/2);
                    k3=-b*(x(k)+k2*h/2);
                    k4=-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                else
                    n=floor((t(k)-tau-t(1))/h+1);
                    k1=Df(x(n))-b*x(k);
                    k2=Df(x(n))-b*(x(k)+h*k1/2);
                    k3=Df(x(n))-b*(x(k)+2*k2*h/2);
                    k4=Df(x(n))-b*(x(k)+k3*h);
                    x(k+1)=x(k)+(k1+2*k2+2*k3+k4)*h/6;
                end
            end
            % add algorithm for matching of first and end
            y=zscore(x);
            error=1;
            i=1;
            sign1=1;
            sign2=-1;

            while(error>0.05||sign1~=sign2||locsign1==locsign2||min(y(i:i+StimSize-1))<-3)
                % difference of value is less than error
                error=abs(y(i)-y(i+StimSize-1));
                % same monotonicity
                sign1=sign(y(i)-y(i+1));
                sign2=sign(y(i+StimSize-1)-y(i+StimSize));
                % different sign of nearest peak
                [pks1,locs1] = findpeaks(y(i:i+StimSize-1));
                [pks2,locs2] = findpeaks(-y(i:i+StimSize-1));
                j=1;
                while(pks1(j)<1)
                    j=j+1;
                end
                k=1;
                while(pks2(k)<1)
                    k=k+1;
                end
                [locstemp,locsign1]=min([locs1(j),locs2(k)]);

                j=length(locs1);
                while(pks1(j)<1)
                    j=j-1;
                end
                k=length(locs2);
                while(pks2(k)<1)
                    k=k-1;
                end
                [locstemp,locsign2]=max([locs1(j),locs2(k)]);
                i=i+1;
            end
            x=x(i:i+StimSize-1);
        end

        function FunPlot_fft(xn,fs)
            % Function
            %   ploting the power spectrum of signal
            % Input
            %   xn: 1-dimentional signal
            %   fs: sampling rate
            %

            y=fft(xn);    %对信号进行快速Fourier变换
            mag=abs(y);     %求得Fourier变换后的振幅
            f=(0:length(xn)-1)*fs/length(xn);    %频率序列
            plot(f,mag);   %绘出随频率变化的振幅
            xlabel('Frequency (Hz)')
            ylabel('Power')

        end

        function FunPlot_periodogram(xn,fs)
            % Function
            %   ploting the power spectrum of signal
            % Input
            %   xn: 1-dimentional signal
            %   fs: sampling rate
            %

            nfft=2^min(8,nextpow2(length(xn)));
            Fs=1000; %采样频率
            n=0:1/Fs:1;
            %产生含有噪声的序列
            xn=cos(2*pi*40*n)+3*cos(2*pi*100*n)+randn(size(n));
            cxn=xcorr(xn,'unbiased'); %计算序列的自相关函数
            CXk=fft(cxn,nfft);
            Pxx=abs(CXk);
            index=0:round(nfft/2-1);
            f=index*fs/nfft;

            plot_Pxx=10*log10(Pxx(index+1));
            plot(f,plot_Pxx);
            xlabel('Frequency (Hz)')
            ylabel('Power')

        end

        function FunPlot_pwelch(xn,fs)
            % Function
            %   ploting the power spectrum of signal
            % Input
            %   xn: 1-dimentional signal
            %   fs: sampling rate
            %

            nfft=2^min(8,nextpow2(length(xn)));
            window_len = max(nfft,length(xn)/fs);
            window1=hamming(window_len); %海明窗
            noverlap=ceil(0.5*window_len); %数据无重叠  % 33%~50% window_len
            range='half'; %频率间隔为[0 fs/2]，只计算一半的频率

            [Pxx1,f]=pwelch(xn,window1,noverlap,nfft,fs,range);

            plot_Pxx1=10*log10(Pxx1);
            plot(f,plot_Pxx1);
            xlabel('Frequency (Hz)')
            ylabel('Power')
        end


    end
end