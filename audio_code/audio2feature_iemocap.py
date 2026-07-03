import os
import subprocess
import time
import logging

# --- 尝试导入显存监控库 ---
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except Exception:
    HAS_GPU = False

# --- 路径配置 ---
config_path = r"audio_video_code/audio_code/opensmile-3.0-win-x64/opensmile-3.0-win-x64/config/is09-13/IS13_ComParE.conf"
smile_extract_path = r"audio_video_code/audio_code/opensmile-3.0-win-x64/opensmile-3.0-win-x64/bin/SMILExtract.exe"

input_root = r"audio_video_code\audio_code\audio_denoise\IEMOCAP_Splits_1011"
output_root = r"audio_video_code\audio_code\audio2feature\IEMOCAP_Splits_1011"

log_file = os.path.join(output_root, "extraction_IEMOCAP_log.txt")

# --- 创建输出目录 ---
if not os.path.exists(output_root):
    os.makedirs(output_root)

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_gpu_memory():
    """获取当前显存占用情况 (MB)"""
    if not HAS_GPU:
        return "N/A"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"{info.used / 1024**2:.2f} MB"
    except:
        return "Error"

def process_tasks():
    total_start_time = time.time()
    total_files = 0
    total_success = 0

    logging.info("开始处理 WAV 文件...")

    # 直接遍历 input_root 下的文件
    for filename in os.listdir(input_root):
        if filename.lower().endswith(".wav"):
            total_files += 1

            input_file = os.path.join(input_root, filename)
            output_file = os.path.join(
                output_root,
                filename.replace(".wav", ".csv")
            )

            start_time = time.time()
            mem_before = get_gpu_memory()

            command = [
                smile_extract_path,
                "-C", config_path,
                "-I", input_file,
                "-O", output_file
            ]

            try:
                subprocess.run(command, check=True, capture_output=True)

                duration = time.time() - start_time
                mem_after = get_gpu_memory()

                logging.info(
                    f"[成功] {filename} | 用时: {duration:.2f}s | 显存: {mem_before} -> {mem_after}"
                )
                total_success += 1

            except subprocess.CalledProcessError as e:
                logging.error(f"[失败] {filename} | 错误原因: {e}")

    # --- 最终统计 ---
    total_duration = time.time() - total_start_time
    logging.info("="*60)
    logging.info("所有任务处理完成！")
    logging.info(f"总计文件数: {total_files}")
    logging.info(f"成功数: {total_success}")
    logging.info(f"失败数: {total_files - total_success}")
    logging.info(f"总耗时: {total_duration/60:.2f} 分钟")
    logging.info("="*60)

if __name__ == "__main__":
    try:
        process_tasks()
    finally:
        if HAS_GPU:
            pynvml.nvmlShutdown()