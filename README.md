# IME Lab Module 2 ECG 采集与分析

本目录包含 MATLAB ECG 采集 App、Python 分析脚本

## 目录

- `app/`：当前 App、其 MATLAB 函数、心脏图片和声音资源。
- `analysis/`：raw 频谱、首个 QRS 波形和 R 峰幅值分析脚本。

## 启动 App

在 MATLAB 中将当前文件夹切换到 `app/`，在命令行运行 `app1`，或通过 App Designer 打开 `app/app1.mlapp`。App 需要 MATLAB Data Acquisition Toolbox、Signal Processing Toolbox、NI 硬件及相应驱动；声音回放需要可用的音频设备。图片和声音需与 `app1.mlapp` 同目录。

采样率大于 100 Hz 时使用现有预处理和 R 峰检测；较低采样率下保存 raw 信号和频谱，filtered 和心率不可用。低采样率对比可能受混叠和采集硬件滤波影响。实际可选采样率由 NI 设备决定。
