//最开始洗数据
function [] = PeaksSnap_SaveAll_Debug_800(str, folder1)
    % --------------
    % 1. 初始化参数
    % --------------

    s = 20;
    siz = 100000 / s;      % => 100000/20 = 5000
    binsize = s * 0.125;   % => 20*0.125 = 2.5

    % 8 个探测器, 每个探测器 144 通道, 每通道有 siz=5000 个采样点
    specs = zeros(8, 144, siz);  % [8 x 144 x 5000]

    % --------------
    % 1.1 读入 8 个 histogram 文件
    % --------------
    for i = 1:8
        infile = fullfile(str, "histogram" + num2str(i) + ".txt");
        disp("Reading file: " + infile);

        f = fopen(infile, "r");
        if f < 0
            warning("Cannot open file: " + infile);
            continue;  
        end

        % 期望文件有 5000 列 × 144 行 => fscanf 读为 [5000 x 144]
        sizeA = [siz, 144];  % => [5000 x 144]
        A = fscanf(f, "%f", sizeA);
        fclose(f);

        if numel(A) < prod(sizeA)
            warning("File %s does not have enough data! Expected %d floats, got %d.", ...
                infile, prod(sizeA), numel(A));
        end

        % 转置 => [144 x 5000], 存入 specs(i,:,:)
        A = A';   
        specs(i, :, :) = A;  
    end

    % x 轴: 从 (binsize/2) 步长 binsize 到 binsize*siz
    % binsize=2.5, siz=5000 => x 有 5000 点, 最后约12500
    xx_full = binsize/2 : binsize : binsize * siz;

    % 仅取前800点 => x(1) ~ x(800) 大约到 ~2000
    nPlot = 800;  
    if nPlot > length(xx_full)
        error("nPlot=800 超过了 x 向量长度 %d.", length(xx_full));
    end
    xx = xx_full(1:nPlot);

    % --------------
    % 2. 创建输出文件夹
    % --------------
    matlabDir = "Matlab";
    if ~exist(matlabDir, 'dir')
        mkdir(matlabDir);
    end

    outDir = fullfile(matlabDir, folder1);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % --------------
    % 3. 处理每个探测器
    % --------------
    for i = 1:8
        specDir = fullfile(matlabDir, "Spec" + num2str(i));
        if ~exist(specDir, 'dir')
            mkdir(specDir);
        end

        figDir = fullfile(outDir, "Figures" + num2str(i));
        if ~exist(figDir, 'dir')
            mkdir(figDir);
        end

        % 用 cell 存储每个通道点击的峰位置
        peakloca = cell(144, 1);

        % 仅示例：处理前128个通道
        for j = 1:128
            % ---------- 3.1 数据获取 -----------
            sp_full = squeeze(specs(i, j, :))';  % [1 x 5000] 行向量
            sp = sp_full(1:nPlot);               % 只取前 800 点

            disp("============");
            disp("Detector i = " + num2str(i) + ", Channel j = " + num2str(j));
            disp("Min(sp)=" + num2str(min(sp)) + " Max(sp)=" + num2str(max(sp)));
            disp("============");

            if all(sp == 0)
                warning("All zeros in sp(1:800) => Possibly empty. Will still plot.");
            end

            % ---------- 3.2 绘制光谱 并自动最大化 ----------
            h = figure('Visible','on','Units','normalized','OuterPosition',[0 0 1 1]);
            semilogy(xx, sp, 'LineWidth', 2);
            hold on;
            title("Detector " + num2str(i) + " - Channel " + num2str(j));
            xlabel("Time (ns approx)");
            ylabel("Counts");

            zoom on;
            pan on;

            % ---------- 3.3 用户点击峰值 ----------
            disp("请在图上点击若干峰值，然后按回车结束选点...");
            [xs, ys] = getpts();
            disp("   => Clicked " + num2str(length(xs)) + " points.");

            % 如果用户没点任何峰 => 跳过
            if isempty(xs)
                disp("   => No points clicked, save as is.");
                specFileTiff = fullfile(specDir, "spec" + num2str(j) + ".tiff");
                specFileFig  = fullfile(figDir,  "spec" + num2str(j) + ".fig");
                saveas(h, specFileTiff);
                saveas(h, specFileFig);
                close(h);
                continue;
            end

            % ---------- 3.4 自动吸附 ----------
            for m = 1:length(xs)
                [~, idx] = min(abs(xx - xs(m)));
                xs(m) = xx(idx);
                ys(m) = sp(idx);
            end

            % ---------- 3.5 记录点击的峰 ----------
            peakloca{j} = xs;

            % 在图上标示这些峰
            for m = 1:length(xs)
                xline(xs(m), '-r', 'LineWidth', 2);
            end

            % ---------- 3.6 保存并关闭 ----------
            specFileTiff = fullfile(specDir, "spec" + num2str(j) + ".tiff");
            specFileFig  = fullfile(figDir,  "spec" + num2str(j) + ".fig");
            saveas(h, specFileTiff);
            saveas(h, specFileFig);
            close(h);
        end

        % ---------- 4. 写出峰值信息 ----------
        peakinfoFile = fullfile(outDir, "peakinfo" + num2str(i) + ".txt");
        disp("Writing peak info: " + peakinfoFile);
        f = fopen(peakinfoFile, "w");
        for k = 1:144
            fprintf(f, "%d ", k);
            if ~isempty(peakloca{k})
                pkXs = peakloca{k};
                for pX = 1:length(pkXs)
                    fprintf(f, "%f ", pkXs(pX));
                end
            end
            fprintf(f, "\n");
        end
        fclose(f);
    end

    disp("All done.");
end
