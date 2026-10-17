# 运行手册

## 环境

训练环境使用 Python 3.11。`features` 安装 Torch、librosa、pyworld、pyopenjtalk；`training` 安装训练和日志依赖，作者 `pytorch-trainer` 源码由 `fetch-sources` 下载，训练器优先导入该快照。不要用 PyPI 的同名包代替。框架把 `WANDB_MODE=disabled` 传给训练子进程，日志保存在 run 目录。服务使用的 Python 3.13 环境不作为训练解释器。

首次准备源码：

```sh
uv pip install --python .venv/bin/python -e '.[dev,features,training,onnx]'
.venv/bin/new-vv-train fetch-sources
```

`fetch-sources` 从 `upstream.lock.json` 下载固定提交并记录每个文件哈希。源码快照含少量 API 适配补丁：Torch 2.4 的 `torch._six`、Python 3.11 的旧 NumPy distutils 检查、当前 wandb ID API，以及作者旧训练器的声学 `inference_mode` 问题。补丁不改变网络层和模型输入输出语义。

### 独立克隆后的外部资产配置

`configs/project.yaml` 的所有相对路径从其 `root`（默认仓库根目录）解析。默认 `../vv_tts_ws_server/` 是原开发工作区的服务目录，不包含在这个 Git 仓库中。独立机器上应编辑配置中的以下字段：

- `frontend.dictionary`：含 `sys.dic` 的 Open JTalk 字典目录；前端不自动下载字典。
- `core.python`：安装目标 `voicevox_core` 0.17.0 的 Python 解释器。
- `core.assets`、`core.existing_models`：实际 Runtime/字典资产根目录和待共存 VVM 目录。
- `core.runtime`、`core.dictionary`：可显式指定 Runtime 二进制与字典。默认 Runtime 文件名针对原开发机的 macOS ARM64，Linux 需配置本机 `.so` 路径。
- `core.service_path`：执行 `validate-service` 时，指向含 `local_engine.py` 的已有服务目录；该检查针对本项目原有服务 adapter，不是任意 TTS 服务的通用入口。

训练和 ONNX 导出可与服务部署分开运行。先执行 `.venv/bin/new-vv-train doctor` 查看缺失输入；没有真实录音时 `not-ready` 和非零退出码是预期行为。

运行自动测试前先安装 `dev,features,training,onnx` 并执行 `fetch-sources`；日语数据集成测试还需要上述字典。使用 `.venv/bin/python -m pytest -q` 运行。CPU 冒烟使用 `.venv/bin/python scripts/smoke_cpu.py`，无需 Core，但需要字典和全部作者源码。

## 语料准备

`init-corpus` 只创建模板，不生成假录音。真实录音放在 corpus `raw/`，manifest 的 `audio` 路径相对于语料根目录。每个 utterance 必须进入且只进入 `train`、`valid`、`test` 之一；`test` 会被 materialize 排除。建议固定说话人、设备和距离，保留原始 WAV，不做逐句 loudness 归一化。

前端使用锁定的 Open JTalk 字典产生 full-context labels 和作者音素表。必须人工抽查读音、长音、促音、拨音和重音。对齐文件是三列秒制 `start end phoneme`，首端从零连续到音频长度；HTK 100 ns 文件需在配置中设置 `alignment.unit: htk`。

派生特征是 24 kHz、hop 256、93.75 Hz：F0 是自然对数 Hz、无声为 0；频谱直接复用作者 HiFi-GAN 的自然对数 Mel 函数；phoneme 是作者顺序的 45 维 one-hot。框架自己的中间文件是无 pickle 的 NPZ，materialize 时才转换成作者 loader 需要的 NPY dict。

## 训练和导出

`training.mode: scratch` 从随机权重开始；`finetune` 只接受配置、音素表、speaker 表和 profile 完全匹配的本地 PyTorch 基座；VVM/ONNX 不能作为训练基座。每个模块必须产生真实 checkpoint，框架会检查 state dict、哈希和新鲜度。

`train duration|intonation|acoustic` 可以在 CPU 上做小规模链路验证，但真实训练建议 Linux/NVIDIA GPU。作者 hifi-gan 训练入口要求 CUDA，当前 Mac CPU 冒烟不会运行它。`scripts/smoke_cpu.py` 可重新运行合成音调测试，并将结果写入 `runs/diagnostics/cpu-smoke.json`。

导出得到 `predict_duration.onnx`、`predict_intonation.onnx` 和只返回 wave 的 `decode.onnx`。NSF 的随机激励只做 shape/有限值检查，不做逐样本波形 parity。VVM 打包前会检查所有模型哈希、speaker 映射、UUID、现有模型 public ID 冲突和 Core 0.17.0 talk 输入输出契约。
