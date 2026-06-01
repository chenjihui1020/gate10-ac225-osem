//能量校正
function Ecalfit_gh_144()
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1) 定义能量峰 & 通道数量
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Epeak = [31; 81; 122; 356; 511; 662];
    numChannels = 144;
    numMeasurements = 8;
    
    % 初始化日志文件
    logFileName = 'partial_data_log.txt';
    logFileID = fopen(logFileName, 'w');
    if logFileID == -1
        error('无法创建日志文件: %s', logFileName);
    end
    fprintf(logFileID, '--- 部分数据缺失通道记录 ---\n');
    fprintf(logFileID, '时间: %s\n\n', datestr(now));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2) 从各能量文件夹读取数据
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TOT = zeros(length(Epeak), numChannels, numMeasurements);
    
    % (内部函数 safe_load_peakinfo 无需修改)
    function data = safe_load_peakinfo(filename, numCh)
        fid = fopen(filename, 'r');
        if fid == -1, warning("无法打开文件: %s", filename); data = zeros(numCh, 2); return; end
        raw = textscan(fid, '%f %f', 'Delimiter', ' ', 'EmptyValue', NaN);
        fclose(fid);
        data = nan(numCh, 2);
        valid_rows = ~isnan(raw{2});
        raw_data = [raw{1}(valid_rows), raw{2}(valid_rows)];
        numRows = min(numCh, size(raw_data,1));
        data(1:numRows, :) = raw_data(1:numRows, :);
        data(isnan(data(:,2)), 2) = 0;
    end

    % -- 依次读取 8 次测量 --
    for i = 1:numMeasurements
        disp("=========== Loading measurement " + i + " ===========");
        folderNames = ["31", "81", "122", "356", "511", "662"];
        for e_idx = 1:length(folderNames)
            file_path = fullfile(folderNames(e_idx), "peakinfo" + i + ".txt");
            disp("Loading: " + file_path);
            data = safe_load_peakinfo(file_path, numChannels);
            TOT(e_idx, :, i) = data(:, 2);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3) 对每次测量的数据做拟合
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:numMeasurements
        values = zeros(numChannels, 3);
        folderName = "Ecalib" + i;
        if ~exist(folderName, 'dir'), mkdir(folderName); end
        
        paramFileName = "calibinfo" + i + ".txt";
        paramFile = fopen(paramFileName, 'w');
        
        for j = 1:numChannels
            current_ch_data = TOT(:, j, i);

            % *** 修改后的核心功能：详细记录缺失的能量点 ***
            is_not_all_zero = any(current_ch_data);
            has_some_zeros = any(current_ch_data == 0);
            
            if is_not_all_zero && has_some_zeros
                % 1. 找到所有值为0的行的索引
                missing_indices = find(current_ch_data == 0);
                % 2. 根据索引从Epeak数组中找到对应的能量值
                missing_energies = Epeak(missing_indices);
                % 3. 将能量值数组转换成逗号分隔的字符串，用于打印
                missing_energies_str = strjoin(string(missing_energies'), ', ');
                % 4. 写入详细的日志信息
                fprintf(logFileID, '记录: 测量 %d, 通道 %d 存在数据缺失。缺失的能量点(keV): %s\n', i, j, missing_energies_str);
            end

            % (后续的拟合逻辑保持不变)
            isMonotonic = all(diff(current_ch_data) >= 0);
            sumVal = sum(current_ch_data);
            
            if isMonotonic && (sumVal > 0)
                try
                    qq = fit(Epeak, current_ch_data, 'b+a*(1-exp(-c*x))', 'StartPoint',[1000,250,0.001]);
                    c1 = -1/qq.c; c2 = -1/qq.a; c3 = 1 + qq.b / qq.a;
                    values(j,:) = [c1, c2, c3];
                    TOT_meas = current_ch_data; E_calc = c1 * log(c2.*TOT_meas + c3);
                    figure('Visible','off');
                    plot(TOT_meas, Epeak, 'o-', 'LineWidth',2); hold on;
                    plot(TOT_meas, E_calc, '--', 'LineWidth',2); hold off;
                    xlabel('TOT (ns)'); ylabel('Energy (keV)');
                    title("Measurement=" + i + ", Ch=" + j);
                    saveas(gcf, fullfile(folderName, i + "-" + j + ".tiff"));
                    close(gcf);
                catch ME
                    warning("拟合失败: meas=%d, ch=%d: %s", i,j,ME.message);
                end
            else
                values(j,:) = [0,0,0];
                figure('Visible','off');
                plot(Epeak, zeros(size(Epeak)), '--o', 'LineWidth',2);
                xlabel('TOT (ns)'); ylabel('Energy (keV)');
                title("数据无效或非单调: M="+i+", Ch="+j);
                saveas(gcf, fullfile(folderName, i + "-" + j + ".tiff"));
                close(gcf);
            end
        end
        
        for ch = 1:numChannels
            fprintf(paramFile, "%f %f %f\n", values(ch,1), values(ch,2), values(ch,3));
        end
        fclose(paramFile);
    end
    
    fclose(logFileID);
    
    disp("完成所有测量的统一能量刻度 (144通道)！");
    disp("详细的缺失数据日志已记录在 " + logFileName);
end
