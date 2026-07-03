import os
import subprocess
import time
import logging
from datetime import datetime

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

# 输入根目录和输出根目录

input_root = r"audio_video_code\audio_code\audio_denoise\MELD.DProcess"
output_root = r"audio_video_code\audio_code\audio2feature\MELD.DProcess"


# 定义文件夹映射关系： { "源文件夹名": "目标文件夹名" }
folder_mapping = {
    "dev_reduce_noise": "dev_audio_feature",
    "train_reduce_noise": "train_audio_feature",
    "test_reduce_noise": "test_audio_feature"
}

log_file = os.path.join(output_root, "extraction_MELD_log.txt")

# --- 配置日志 ---
if not os.path.exists(output_root):
    os.makedirs(output_root)

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
    if not HAS_GPU: return "N/A"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"{info.used / 1024**2:.2f} MB"
    except: return "Error"

def process_tasks():
    total_start_time = time.time()
    total_files = 0
    total_success = 0

    # 遍历映射字典
    for src_sub, dst_sub in folder_mapping.items():
        src_path = os.path.join(input_root, src_sub)
        dst_path = os.path.join(output_root, dst_sub)

        if not os.path.exists(src_path):
            logging.warning(f"源文件夹不存在，跳过: {src_path}")
            continue

        logging.info(f"正在处理分区: {src_sub} -> {dst_sub}")

        # 使用 os.walk 处理该分区下的所有文件（包括子文件夹）
        for root, dirs, files in os.walk(src_path):
            for filename in files:
                if filename.endswith(".wav"):
                    total_files += 1
                    
                    # 1. 确定输入完整路径
                    input_file = os.path.join(root, filename)
                    
                    # 2. 保持子目录结构（如果有）
                    # 计算当前文件相对于 src_path 的相对路径
                    rel_dir = os.path.relpath(root, src_path)
                    current_target_dir = os.path.join(dst_path, rel_dir)
                    
                    if not os.path.exists(current_target_dir):
                        os.makedirs(current_target_dir)

                    # 3. 确定输出完整路径
                    output_file = os.path.join(current_target_dir, filename.replace(".wav", ".csv"))

                    # --- 执行提取 ---
                    start_time = time.time()
                    mem_before = get_gpu_memory()

                    command = [
                        smile_extract_path,
                        "-C", config_path,
                        "-I", input_file,
                        "-O", output_file
                    ]

                    try:
                        # 运行命令，capture_output=True 可以保持控制台整洁
                        subprocess.run(command, check=True, capture_output=True)
                        
                        duration = time.time() - start_time
                        mem_after = get_gpu_memory()
                        
                        logging.info(f"[成功] {filename} | 用时: {duration:.2f}s | 显存: {mem_before} -> {mem_after}")
                        total_success += 1
                    except subprocess.CalledProcessError as e:
                        logging.error(f"[失败] {filename} | 错误原因: {e}")

    # --- 最终统计 ---
    total_duration = time.time() - total_start_time
    logging.info("="*60)
    logging.info(f"所有任务处理完成！")
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