#
# Multiprocessing looper for .data
# cutting injection
# 
# (2026) NanaVan@github
# Inspired by loopr.py by (2025) xaratustrah@github
#

import numpy as np
import pyfftw, multiprocessing, os, sys, re, time, signal
from scipy.signal import windows
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from preprocessing import Preprocessing


def handle_windows(window_length, window=None, beta=None):
    '''
    handling various windows

    window_length:      length of the tapering window
    window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    if window is None:
        window_sequence = np.ones(window_length)
    elif window == "kaiser":
        if beta is None:
            raise ValueError("additional argument beta is empty!")
        else:
            window_sequence = np.kaiser(window_length, beta)
    else:
        window_func = getattr(np, window)
        window_sequence = window_func(window_length)
    return window_sequence

def file_cutInjection(input_file, output_dir, window_length, n_average, overlap_ratio, n_hop=None, window=None, beta=None, last_file=None, todo=['data_spectrogram', 'png_spectrogram', 'data_spectrum', 'png_spectrum']):
    '''
    Cutting puyuan's raw data based on injection, for each injection saved as a .npz file
    
    input_file:         .data file from 4-channel puyuan devices
    output_dir:         output files' dirname
    last_file:          .npz file of the end of the last file
    todo:               'png_spectrum': saving .png figure for averaged spectrum for one data file
                        'png_spectrogram': saving .png figure for waterfall spectrogram for one data file
                        'data_spectrum': saving .npz data for averaged spectrum for one data file
                        'data_spectrogram': saving .npz data for waterfall spectrogram for one data file
    window_length:      length of the tapering window, a.k.a. L
    n_average:          length of average, len k
    overlap_ratio:      the overlap ratio for the L-D points and if K sequences cover the entire N data points
                        tot_N = ( L + D * ( K - 1 ) ) * n_frame
                        overlap_ratio = 1 - D / L
    n_hop:              number of points skipped between each frame, default tot_N
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser", etc.] (from scipy.signal.windows)
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected

    ---
    return
                        .npz/.png file with format, 
                        timestamp stands for the beginning of psd_arrays
                        completed:  8249_PY82ch1_0010_trigger_18_2026-04-08T02-00-44_{spectrogram/spectrum}.{npz/png} (complete injection inside PY82ch1_10.data)
                                    8249_PY82ch1_0010-0011_trigger_2026-04-08T02-00-44_{spectrogram/spectrum}.{npz/png} (complete injection between PY82ch1_10.data & PY82ch1_11.data)
                                    {spectrogram/spectrum}.{npz/png} depends on the todo's choices
                                    spectrogram:    frequencies, times, psd_arrays
                                    spectrum:       frequencies, psds
                                    for completed files, frequencies range within span
                        incomplete: 8249_PY82ch1_0010_trigger_0_incomplete_2026-04-08T02-00-44_spectrogram.npz (PY82ch1_10.data has no trigger inside)
                                    8249_PY82ch1_0010_trigger_2_incomplete_2026-04-08T02-00-44_spectrogram.npz (end of file PY82ch1_10.data)
                                    8249_PY82ch1_0010-0011_trigger_incomplete_2026-04-08T02-00-44_spectrogram.npz (new injection begain from PY82ch1_10.data, no trigger inside PY82ch1_11.data)
                                    spectrogram:    frequencies, times, psd_arrays, addition_data
                                    for incompleted files, frequencies range within sampling rate
    '''
    
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = window_length
    if window_length == int (window_length * overlap_ratio): overlap_ratio = 0.5
    D = int((1 - overlap_ratio) * window_length) 
    N = int(window_length + D * (n_average - 1))
    # modify hop number
    if n_hop == 0 or n_hop == None: n_hop = N
    if n_hop < 0:
        print("Input Error: n_hop must >= 0!")
        sys.exit(1)

    _current_fileIndex = int(os.path.basename(input_file).split('_')[1].split('.')[0])
    _prefix = os.path.dirname(input_file).split('/')[-1].split('_')[0] + '_' + os.path.basename(input_file).split('_')[0]  # 8249_PY82ch1

    # create an FFT plan
    dummy = pyfftw.empty_aligned((n_average, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)

    bud = Preprocessing(input_file, puyuan_new=True, abs_trigger=False)
    ThisFileTimestamp = bud.date_time + np.timedelta64(8, 'h') # convert to '+08' timezone

    if last_file is None:
        additional_x, lastTriggerData_remain, trigger_frame, offset = np.array([]), 0, 0, 0
    else:
        trigger_inLastFile = False
        _macth = re.search(r'trigger_\d+', os.path.basename(last_file))
        if _macth: # 前序文件存在触发信号，trigger_inLastFile = True，反之亦然
            trigger_inLastFile = True if int(_macth.group().split('_')[-1]) > 0 else False
        with np.load(last_file) as _f:
            if trigger_inLastFile:
                additional_x, lastTriggerData_remain, psd_array = _f['addition_data'], len(_f['addition_data']), _f['psd_arrays']
                trigger_frame, offset = len(_f['times'])-1, 0
                _match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', os.path.basename(last_file))
                if _match:
                    ThisDataTimestamp = np.datetime64(_match.group(1)[:11] + _match.group(1)[11:].replace('-', ':'), 's')
            else:
                additional_x, lastTriggerData_remain = _f['addition_data'], len(_f['addition_data'])
                trigger_frame, offset = 0, 0
                _match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', os.path.basename(last_file))
                if _match:
                    ThisDataTimestamp = np.datetime64(_match.group(1)[:11] + _match.group(1)[11:].replace('-', ':'), 's') + np.timedelta64(int(_f['times'][-1]), 's')


        
    # 如果整个文件中都不存在触发信号，那么执行以下策略：1.前序文件含有触发信号，就将前序文件末尾的新注入数据与本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中；2.前序文件不含有触发信号，仅将前序文件addition_data中的数据与本文件可组成频谱的内容合并未新文件（incomplete），未组合成频谱部分放置于addition_data中。3.无前序文件（见于PY8*ch*_0.data），将本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中。
    if len(bud.trigger_timestamp) == 0:
        print('[!] 当前文件内无触发信号，将按普通频谱处理！')
        ThisTriggerData_remain = bud.n_sample
        while True:
            if ThisTriggerData_remain > N:
                x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain, offset)[1]))
                _signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                if trigger_frame == 0:
                    psd_array = np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                else:
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                ThisTriggerData_remain -= n_hop
                trigger_frame += 1
                if lastTriggerData_remain > 0:
                    if lastTriggerData_remain - n_hop > 0:
                        offset = 0
                        additional_x = additional_x[n_hop:]
                        lastTriggerData_remain = len(additional_x)
                    else:
                        offset = n_hop - lastTriggerData_remain
                        additional_x = np.array([])
                        lastTriggerData_remain = 0
                else:
                    offset += n_hop
            else:
                additional_x = np.hstack((additional_x, bud.load(ThisTriggerData_remain, offset)[1]))
                lastTriggerData_remain = len(additional_x)
                offset = 0
                frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
                if n_point % 2 ==1: frequencies += bud.sampling_rate / (2*n_point)
                x_frequency = frequencies + bud.center_frequency # Hz
                y_time = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
                z_psd_array = psd_array
                if trigger_inLastFile or (last_file is None):
                    print('[*] 当前无触发文件的前序文件也无触发或不存在，保留时频谱 ...')
                    np.savez(os.path.join(output_dir, '{:}_{:04}_trigger_0_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x) # 前序不含有触发信号或者前序文件不存在，生成格式为 8249_PY82ch1_0010_trigger_0_incomplete_2026-04-08T22-12-23_spectrogram.npz
                    print('[√] 文件生成：{:}'.format('{:}_{:04}_trigger_0_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                else:
                    print('[*] 当前无触发文件将与前序文件剩余部分合并，保留时频谱 ...')
                    np.savez(os.path.join(output_dir, '{:}_{:04d}-{:04}_trigger_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x) # 前序文件含有触发信号，生成文件格式为 8249_PY82ch1_0010-0011_trigger_incomplete_2026-04-08T22-12-23_spectrogram.npz
                    print('[√] 文件生成：{:}'.format('{:}_{:04d}-{:04}_trigger_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                break
        return

    for trigger_i, trigger_timestamp in enumerate(bud.trigger_timestamp):
        ThisTriggerData_remain = trigger_timestamp * bud.data_len + lastTriggerData_remain - offset
        if trigger_i == 0 and trigger_frame == 0:
            print('[!] 当前含新注入的文件为文件夹中起始文件，生成文件将从首次注入开始 ...')
            offset = trigger_timestamp * bud.data_len
            ThisDataTimestamp = ThisTriggerData_remain + np.timedelta64(int(offset/bud.sampling_rate), 's')
            continue
        print('[*] 开始对本文件中第{:}次注入，进行处理...'.format(trigger_i))
        while True:
            if ThisTriggerData_remain > N:
                x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain, offset)[1]))
                _signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                if trigger_frame == 0:
                    psd_array = np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                else:
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                ThisTriggerData_remain -= n_hop
                trigger_frame += 1
                if lastTriggerData_remain > 0:
                    if lastTriggerData_remain - n_hop > 0:
                        offset = 0
                        additional_x = additional_x[n_hop:]
                        lastTriggerData_remain = len(additional_x)
                    else:
                        offset = n_hop - lastTriggerData_remain
                        additional_x = np.array([])
                        lastTriggerData_remain = 0
                else:
                    offset += n_hop
            else:
                frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1)
                if n_point % 2 ==1: frequencies += bud.sampling_rate / (2*n_point)
                freq_idx_0, freq_idx_1 = np.searchsorted(frequencies, [-bud.span/2, bud.span/2])
                x_frequency = frequencies[freq_idx_0:freq_idx_1+1] + bud.center_frequency # Hz
                y_time = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
                z_psd_array = psd_array[:,freq_idx_0:freq_idx_1]
                if trigger_i == 0:
                    print('[*] 正在处理注入在 file {:} - {:} 的数据'.format(_current_fileIndex-1, _current_fileIndex))
                    if 'data_spectrum' in todo:
                        np.savez(os.path.join(output_dir, '{:}_{:04d}-{:04d}_trigger_{:}_spectrum.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency[:-1], psd=np.mean(z_psd_array, axis=0))
                    if 'png_spectrum' in todo:
                        fig, ax = plt.subplots(figsize=(10,6))
                        ax.plot(x_frequency[:-1]*1e-3, np.mean(z_psd_array, axis=0))
                        ax.set_yscale('log')
                        ax.set_title('Average Spectrum\nFile: {:}'.format('{:}_{:04d}-{:04d}_trigger_{:}_spectrum.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                        ax.set_xlabel('Frequency [kHz]')
                        ax.set_ylabel('Power Spectral Density [arb. unit]')
                        ax.set_xlim((x_frequency[0]*1e-3, x_frequency[-1]*1e-3))
                        ax.grid(True, which='both', ls='--', alpha=0.5)
                        plt.savefig(os.path.join(output_dir, '{:}_{:04d}-{:04d}_trigger_{:}_spectrum.png'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), transparent=False)
                        plt.close()
                    if 'data_spectrogram' in todo:
                        np.savez(os.path.join(output_dir, '{:}_{:04d}-{:04d}_trigger_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array)
                    if 'png_spectrogram' in todo:
                        norm = colors.LogNorm(vmin=z_psd_array.min(), vmax=z_psd_array.max())
                        fig, ax = plt.subplots(figsize=(12,10))
                        waterfall = ax.pcolormesh(x_frequency*1e-3, # kHz
                                    y_time*1e3, # ms
                                    z_psd_array,
                                    shading = 'flat', cmap = 'viridis', norm = norm)
                        ax.set_title('Waterfall Plot\nFile: {:}'.format('{:}_{:04d}-{:04d}_trigger_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                        ax.set_xlabel('Frequency [kHz]')
                        ax.set_ylabel('Time [ms]')
                        cbar = fig.colorbar(waterfall, ax=ax)
                        cbar.set_label('Power Spectral Density [arb. unit]')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, '{:}_{:04d}-{:04d}_trigger_{:}_spectrogram.png'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), transparent=False)
                        plt.close()
                    print('[√] 当前数据文件处理结束。')
                else:
                    print('[*] 正在处理 file: {:}, trigger: {:}'.format(_current_fileIndex, trigger_i))
                    if 'data_spectrum' in todo:
                        np.savez(os.path.join(output_dir, '{:}_{:04d}_trigger_{:}_{:}_spectrum.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, psd=np.mean(z_psd_array, axis=0))
                    if 'png_spectrum' in todo:
                        fig, ax = plt.subplots(figsize=(10,6))
                        ax.plot(x_frequency[:-1]*1e-3, np.mean(z_psd_array, axis=0))
                        ax.set_yscale('log')
                        ax.set_title('Average Spectrum\nFile: {:}'.format('{:}_{:04d}_trigger_{:}_{:}_spectrum.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                        ax.set_xlabel('Frequency [kHz]')
                        ax.set_ylabel('Power Spectral Density [arb. unit]')
                        ax.set_xlim((x_frequency[0]*1e-3, x_frequency[-1]*1e-3))
                        ax.grid(True, which='both', ls='--', alpha=0.5)
                        plt.savefig(os.path.join(output_dir, '{:}_{:04d}_trigger_{:}_{:}_spectrum.png'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), transparent=False)
                        plt.close()
                    if 'data_spectrogram' in todo:
                        np.savez(os.path.join(output_dir, '{:}_{:04d}_trigger_{:}_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array)
                    if 'png_spectrogram' in todo:
                        norm = colors.LogNorm(vmin=z_psd_array.min(), vmax=z_psd_array.max())
                        fig, ax = plt.subplots(figsize=(12,10))
                        waterfall = ax.pcolormesh(x_frequency*1e-3, # kHz
                                    y_time*1e3, # ms
                                    z_psd_array,
                                    shading = 'flat', cmap = 'viridis', norm = norm)
                        ax.set_title('Waterfall Plot\nFile: {:}'.format('{:}_{:04d}_trigger_{:}_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
                        ax.set_xlabel('Frequency [kHz]')
                        ax.set_ylabel('Time [ms]')
                        cbar = fig.colorbar(waterfall, ax=ax)
                        cbar.set_label('Power Spectral Density [arb. unit]')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, '{:}_{:04d}_trigger_{:}_{:}_spectrogram.png'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), transparent=False)
                        plt.close()
                    print('[√] 当前数据文件处理结束。')
                additional_x = np.array([])
                lastTriggerData_remain = 0
                offset = trigger_timestamp * bud.data_len
                ThisDataTimestamp = ThisFileTimestamp + np.timedelta64(int(offset/bud.sampling_rate), 's')
                trigger_frame = 0
                break

    trigger_i += 1
    ThisTriggerData_remain = bud.n_sample + lastTriggerData_remain - offset
    print('[*] 正在处理本文件剩余数据...')
    while True:
        if ThisTriggerData_remain > N:
            x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain, offset)[1]))
            _signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
            if trigger_frame == 0:
                psd_array = np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
            else:
                psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
            ThisTriggerData_remain -= n_hop
            trigger_frame += 1
            offset += n_hop
        else:
            additional_x = np.hstack((additional_x, bud.load(ThisTriggerData_remain, offset)[1]))
            lastTriggerData_remain = len(additional_x)
            offset = 0
            frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
            if n_point % 2 ==1: frequencies += bud.sampling_rate / (2*n_point)
            x_frequency = frequencies + bud.center_frequency # Hz
            y_time = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
            z_psd_array = psd_array
            np.savez(os.path.join(output_dir, '{:}_{:04d}_trigger_{:}_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
            print('[√] 本文件剩余部分保存：{:}'.format('{:}_{:04d}_trigger_{:}_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))))
            break


class FileProcessor:
    def __init__(self, SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL):
        self.SOURCE_DIR = SOURCE_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        self.FILE_PREFIX = FILE_PREFIX
        self.EXPECTED_SIZE = EXPECTED_SIZE
        self.CHECK_INTERVAL = CHECK_INTERVAL
        self.running = True
        self.current_index = self._get_start_index()
        # 注册信号捕获，处理 Ctrl+C
        signal.signal(signal.SIGINT, self._handle_exit)

    def _get_start_index(self):
        '''
        启动时检测输出文件夹，寻找最大的 index
        '''
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            return 0
        
        existing_files = os.listdir(self.OUTPUT_DIR)
        indices = []
        for f in existing_files:
            indices.append(int(os.path.basename(f).split('_')[2].split('-')[-1]))
        if not indices:
            return 0

        last_index = max(indices)
        print(f"检测到已处理至索引：{last_index}，将从{last_index+1}开始。")
        return last_index + 1

    def _find_last_file(self, index):
        '''检索出最后生成的不完整文件'''
        index = index - 1
        target_files = []
        for filename in os.listdir(self.OUTPUT_DIR):
            if not filename.endswith('.npz'):
                continue
            if 'incomplete' in filename:
                try:
                    if int(filename.split('_')[2].split('-')[-1]) == index:
                        target_files.append(filename)
                except:
                    continue
        try:
            return os.path.basename(target_files[0])
        except:
            return None

    def _delete_assigned_incomplete_file(self, last_file):
        '''
        检查前序不完整文件是否是当前生成文件中已覆盖的：
        前序文件存在触发，当前生成文件使用了前序文件末尾数据合并成了频谱
        删除该不完整文件（特征是含有trigger_i_incomplete，其中i>0）
            8249_PY82ch1_0010_trigger_i_incomplete_2026-04-08T02-00-44_spectrogram.npz
        '''
        _match = re.search(r'trigger_(\d+)_incomplete', last_file)
        if _match:
            trigger_i = int(_match.group(1))
            if trigger_i > 0:
                os.remove(os.path.join(self.OUTPUT_DIR, last_file))
                print('[!] 删除掉前序不完整文件：{:}'.format(last_file))

    def _handle_exit(self, signum, frame):
        '''捕获退出信号，但不立即停止，而是标记状态'''
        print("\n[!] 接收到退出信号，正在处理当前文件，请稍后...")
        self.running = False

    def is_file_ready(self, path):
        '''完整性校验：检查大小且确认不再增长'''
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < self.EXPECTED_SIZE:
            return False
        # 再次确认文件没有再写入
        s1 = os.path.getsize(path) # Bytes
        time.sleep(0.2)
        if s1 != os.path.getsize(path):
            return False
        return True

    def run(self, WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO):
        '''处理逻辑'''
        print("程序启动，当前文件索引：{:}".format(self.current_index))

        while True:
            target_file = os.path.join(self.SOURCE_DIR, '{:}_{:}.data'.format(self.FILE_PREFIX, self.current_index))

            if self.is_file_ready(target_file):
                # 即使收到了退出信号，也会先执行完这段代码
                last_file = self._find_last_file(self.current_index)
                if last_file is not None:
                    lastfile_path = os.path.join(self.OUTPUT_DIR, self._find_last_file(self.current_index))
                else:
                    lastfile_path = None
                file_cutInjection(target_file, self.OUTPUT_DIR, WIN_LEN, N_AVER, OVERLAPR, n_hop=N_HOP, window='kaiser', beta=14, last_file=lastfile_path, todo=TODO)
                if last_file is not None:
                    self._delete_assigned_incomplete_file(last_file)
                self.current_index += 1
                # 处理完当前文件后，检查是否需要退出
                if not self.running:
                    print('[!] 当前任务已安全结束，程序正常退出。')
                    break
            else:
                # 文件未就绪，且用户按了 Ctrl+C
                if not self.running:
                    print('[!] 无待处理任务，程序正常退出。')
                    break
                time.sleep(self.CHECK_INTERVAL)

if __name__ == "__main__":
    processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
