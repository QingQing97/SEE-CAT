import os
import subprocess
import time
import threading
from datetime import datetime

import psutil  # pip install psutil


class ResourceMonitor:
    """后台线程监控 GPU 显存 和 CPU 内存占用"""

    def __init__(self, interval=2.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

        self.gpu_mem_samples = []
        self.gpu_available = False
        self.cpu_mem_samples = []
        self.process = psutil.Process(os.getpid())

    def _check_gpu_available(self):
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_gpu_memory_used(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            total_used = sum(int(line.strip()) for line in lines if line.strip())
            return total_used
        except Exception:
            return 0

    def _get_cpu_memory_used(self):
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception:
            return 0

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            cpu_mem = self._get_cpu_memory_used()
            self.cpu_mem_samples.append(cpu_mem)

            if self.gpu_available:
                gpu_mem = self._get_gpu_memory_used()
                self.gpu_mem_samples.append(gpu_mem)

            self._stop_event.wait(self.interval)

    def start(self):
        self.gpu_available = self._check_gpu_available()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_summary(self):
        lines = []
        lines.append("----- 资源使用统计 -----")

        if self.cpu_mem_samples:
            avg_cpu = sum(self.cpu_mem_samples) / len(self.cpu_mem_samples)
            peak_cpu = max(self.cpu_mem_samples)
            lines.append(f"CPU 内存 - 采样次数: {len(self.cpu_mem_samples)}")
            lines.append(f"CPU 内存 - 峰值占用: {peak_cpu:.2f} MB")
            lines.append(f"CPU 内存 - 平均占用: {avg_cpu:.2f} MB")
        else:
            lines.append("CPU 内存 - 无采样数据")

        lines.append("")

        if not self.gpu_available:
            lines.append("GPU 显存 - 未检测到 NVIDIA GPU（nvidia-smi 不可用）")
        elif self.gpu_mem_samples:
            avg_gpu = sum(self.gpu_mem_samples) / len(self.gpu_mem_samples)
            peak_gpu = max(self.gpu_mem_samples)
            min_gpu = min(self.gpu_mem_samples)
            lines.append(f"GPU 显存 - 采样次数: {len(self.gpu_mem_samples)}")
            lines.append(f"GPU 显存 - 峰值占用: {peak_gpu} MB")
            lines.append(f"GPU 显存 - 最低占用: {min_gpu} MB")
            lines.append(f"GPU 显存 - 平均占用: {avg_gpu:.2f} MB")
        else:
            lines.append("GPU 显存 - 无采样数据")

        lines.append("------------------------")
        return "\n".join(lines)


def extract_audio_from_videos(input_dirs, output_dirs, log_file, sample_rate=44100, channels=2):
    """
    从多个输入目录中提取 mp4 视频音频为 wav 格式，
    输入目录与输出目录一一对应。
    """

    assert len(input_dirs) == len(output_dirs), \
        f"输入目录数量({len(input_dirs)})与输出目录数量({len(output_dirs)})不匹配！"

    # 启动资源监控
    monitor = ResourceMonitor(interval=2.0)
    monitor.start()

    start_time = time.time()
    start_datetime = datetime.now()

    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write("            音频提取任务日志\n")
        log.write("=" * 60 + "\n\n")
        log.write(f"开始时间 : {start_datetime}\n")
        log.write(f"采样率   : {sample_rate} Hz\n")
        log.write(f"声道数   : {channels}\n\n")

        log.write("目录映射关系:\n")
        for i, (in_dir, out_dir) in enumerate(zip(input_dirs, output_dirs)):
            log.write(f"  [{i+1}] {in_dir}\n")
            log.write(f"   -> {out_dir}\n")
        log.write("\n" + "-" * 60 + "\n\n")

        total_files = 0
        success_files = 0
        failed_files = 0

        for input_dir, output_dir in zip(input_dirs, output_dirs):

            # 确保对应的输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            log.write(f"[扫描] 输入: {input_dir}\n")
            log.write(f"       输出: {output_dir}\n")
            print(f"\n[扫描] 输入: {input_dir}")
            print(f"       输出: {output_dir}")

            if not os.path.isdir(input_dir):
                log.write(f"  ⚠ 输入目录不存在，跳过\n\n")
                print(f"  ⚠ 输入目录不存在，跳过")
                continue

            dir_count = 0

            for root, dirs, files in os.walk(input_dir):
                # 计算相对路径，保留子目录结构
                rel_path = os.path.relpath(root, input_dir)
                current_output_dir = os.path.join(output_dir, rel_path) if rel_path != "." else output_dir
                os.makedirs(current_output_dir, exist_ok=True)

                for file in sorted(files):
                    if not file.lower().endswith(".mp4"):
                        continue

                    total_files += 1
                    dir_count += 1

                    input_path = os.path.join(root, file)
                    output_filename = os.path.splitext(file)[0] + ".wav"
                    output_path = os.path.join(current_output_dir, output_filename)

                    file_start = time.time()

                    command = [
                        "ffmpeg",
                        "-y",
                        "-i", input_path,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", str(sample_rate),
                        "-ac", str(channels),
                        output_path,
                    ]

                    try:
                        subprocess.run(
                            command,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        file_duration = time.time() - file_start
                        log.write(f"  [成功] {input_path}\n")
                        log.write(f"         -> {output_path}  ({file_duration:.2f}s)\n")
                        print(f"  [成功] {file}  ({file_duration:.2f}s)")
                        success_files += 1

                    except subprocess.CalledProcessError as e:
                        file_duration = time.time() - file_start
                        log.write(f"  [失败] {input_path}  ({file_duration:.2f}s)\n")
                        log.write(f"         错误: {e}\n")
                        print(f"  [失败] {file}")
                        failed_files += 1

            log.write(f"  该目录共处理: {dir_count} 个文件\n\n")
            print(f"  该目录共处理: {dir_count} 个文件")

        # 停止监控
        monitor.stop()

        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time

        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

        log.write("=" * 60 + "\n")
        log.write("            任务统计\n")
        log.write("=" * 60 + "\n\n")
        log.write(f"结束时间     : {end_datetime}\n")
        log.write(f"总运行时间   : {duration_str}  ({total_duration:.2f} 秒)\n\n")
        log.write(f"总文件数     : {total_files}\n")
        log.write(f"成功数量     : {success_files}\n")
        log.write(f"失败数量     : {failed_files}\n\n")

        resource_summary = monitor.get_summary()
        log.write(resource_summary + "\n")

        print("\n" + resource_summary)
        print(f"\n总运行时间: {duration_str}")
        print(f"日志已保存: {log_file}")


if __name__ == "__main__":

    # ============================================================
    #   输入目录列表（视频源）
    # ============================================================
    input_directories = [
        r"audio_video_code/audio_code/Datasets/MELD.Raw/dev_splits",
        r"audio_video_code/audio_code/Datasets/MELD.Raw/test_splits",
        r"audio_video_code/audio_code/Datasets/MELD.Raw/train_splits",
        r"audio_video_code/audio_code/Datasets/IEMOCAP_Splits_1011",
    ]

    # ============================================================
    #   输出目录列表（一一对应）
    # ============================================================
    output_directories = [
        r"audio_video_code/audio_code/video2audio/MELD.Raw/dev_splits",
        r"audio_video_code/audio_code/video2audio/MELD.Raw/test_splits",
        r"audio_video_code/audio_code/video2audio/MELD.Raw/train_splits",
        r"audio_video_code/audio_code/video2audio/IEMOCAP_Splits_1011",
    ]

    # ============================================================
    #   日志文件路径
    # ============================================================
    log_path = r"audio_video_code/audio_code/video2audio/audio_extract_log.txt"

    # 确保日志所在目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    extract_audio_from_videos(input_directories, output_directories, log_path)