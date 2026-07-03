# -*- coding: utf-8 -*-
import os
import glob
import re
import csv
import pickle
import time
import logging
import psutil
from collections import defaultdict

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# ==================== 路径配置 ====================
BASE_DIR = "audio_video_code/audio_code/audio2feature/MELD.DProcess"

CSV_DIRS = [
    os.path.join(BASE_DIR, "train_audio_feature"),
    os.path.join(BASE_DIR, "dev_audio_feature"),
    os.path.join(BASE_DIR, "test_audio_feature"),
]

OUT_PATH = os.path.join(BASE_DIR, "AudioFeatures.pkl")
LOG_PATH = os.path.join(BASE_DIR, "csv2pkl.log")


# ==================== 日志配置 ====================
logger = logging.getLogger("AudioFeatures")
logger.setLevel(logging.INFO)

# 同时输出到文件和终端
file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
console_handler = logging.StreamHandler()

formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ==================== 资源监控 ====================
def get_ram_usage_mb():
    """获取当前进程的内存(RAM)占用，单位 MB"""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


def get_gpu_usage_mb():
    """获取所有 GPU 的显存占用，单位 MB，返回列表"""
    if not HAS_NVML:
        return None
    gpu_info = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / (1024 * 1024)
        total_mb = mem_info.total / (1024 * 1024)
        gpu_info.append((i, used_mb, total_mb))
    return gpu_info


def log_resource(tag=""):
    """记录当前内存和显存"""
    ram_mb = get_ram_usage_mb()
    msg = f"[资源] {tag} | RAM: {ram_mb:.1f} MB"

    gpu_info = get_gpu_usage_mb()
    if gpu_info:
        for gpu_id, used, total in gpu_info:
            msg += f" | GPU{gpu_id}: {used:.1f}/{total:.1f} MB"
    else:
        msg += " | GPU: 不可用或未检测到"

    logger.info(msg)


# ==================== 核心逻辑 ====================
def parse_dia_utt(filename):
    """提取 dia_id 和 utt_id"""
    match = re.search(r"dia(\d+)_utt(\d+)", filename)
    if not match:
        raise ValueError(f"无法解析文件名: {filename}")
    return int(match.group(1)), int(match.group(2))


def extract_6373_features(csv_path):
    """
    专门针对 openSMILE 的 IS13-ComParE 提取
    跳过头信息，提取掐头去尾的 6373 维特征，'?' 转为 0.0
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 100:
                raw_feats = row[1:-1]
                feats = []
                for x in raw_feats:
                    if x == '?':
                        feats.append(0.0)
                    else:
                        try:
                            feats.append(float(x))
                        except ValueError:
                            feats.append(0.0)
                return feats
    return None


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("开始构建 AudioFeatures.pkl")
    logger.info(f"输出路径: {OUT_PATH}")
    logger.info(f"日志路径: {LOG_PATH}")
    logger.info("=" * 60)

    log_resource("程序启动")

    # 结构: dia_id -> {utt_id: feature_list}
    dialog2utts = defaultdict(dict)

    total_files = 0
    success_files = 0
    skip_files = 0

    for csv_dir in CSV_DIRS:
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        split_name = os.path.basename(csv_dir)
        logger.info(f"扫描目录: {split_name}, 共找到 {len(csv_files)} 个 csv 文件")

        dir_start_time = time.time()

        for csv_path in csv_files:
            total_files += 1
            filename = os.path.basename(csv_path)

            try:
                dia_id, utt_id = parse_dia_utt(filename)
                feats = extract_6373_features(csv_path)

                if feats and len(feats) == 6373:
                    dialog2utts[dia_id][utt_id] = feats
                    success_files += 1
                else:
                    logger.warning(f"特征维度不是 6373 或提取失败: {filename}")
                    skip_files += 1

            except Exception as e:
                logger.warning(f"跳过 {filename}: {e}")
                skip_files += 1

            # 每处理 500 个文件记录一次资源
            if total_files % 500 == 0:
                log_resource(f"已处理 {total_files} 个文件")

        dir_elapsed = time.time() - dir_start_time
        logger.info(f"{split_name} 处理完毕，耗时 {dir_elapsed:.2f}s")

    log_resource("特征提取完成")

    # 组装最终格式
    logger.info("正在组装最终数据结构...")
    final_audio_features = {}

    for dia_id, utt_dict in dialog2utts.items():
        sorted_utts = sorted(utt_dict.items(), key=lambda x: x[0])
        final_audio_features[dia_id] = [feat for _, feat in sorted_utts]

    log_resource("数据结构组装完成")

    # 保存
    logger.info("正在保存为 PKL 文件...")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_audio_features, f, protocol=4)

    log_resource("PKL 保存完成")

    # 总耗时
    total_elapsed = time.time() - start_time

    # 最终统计
    logger.info("=" * 60)
    logger.info("转换完成！最终统计：")
    logger.info(f"  扫描文件总数:       {total_files}")
    logger.info(f"  成功提取文件数:     {success_files}")
    logger.info(f"  跳过/失败文件数:    {skip_files}")
    logger.info(f"  生成的对话(dia)数:  {len(final_audio_features)}")
    logger.info(f"  单条特征维度:       6373")
    logger.info(f"  输出文件路径:       {OUT_PATH}")
    logger.info(f"  输出文件大小:       {os.path.getsize(OUT_PATH) / (1024*1024):.2f} MB")
    logger.info(f"  总运行时间:         {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
    logger.info("=" * 60)

    # 关闭 pynvml
    if HAS_NVML:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()