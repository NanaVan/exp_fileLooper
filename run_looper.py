from looper_cut import *

# --- 配置参数 ---
SOURCE_DIR = '/mnt/nas82_2/raw_data/puyuan82_data/Data/8243_TestModePY82_26-04-07_22-12-25'      # 原始文件存放路径
OUTPUT_DIR = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/test_looper/8243_TestModePY82_26-04-07_22-12-25/file_cutInjection2'   # 处理后生成文件的路径
FILE_PREFIX = 'PY82ch1'
EXPECTED_SIZE = 1024 * 1024 * 1024      # 原始文件固定大小, Bytes
CHECK_INTERVAL = 0.5              # 检查频率, sec

WIN_LEN = 262144                  # 频谱窗口长度
N_AVER = 4                        # 单帧平均次数
OVERLAPR = 0.60881                # 数据重叠率
N_HOP = 250108                    # 单帧数据间隔
TODO = ['data_spectrogram', 'data_spectrum', 'png_spectrogram', 'png_spectrum']


# --- 运行程序: looper_cut ---
processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL)
processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)


