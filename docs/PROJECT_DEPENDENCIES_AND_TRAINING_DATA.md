# 项目外部依赖与训练数据清单

审计日期：2026-10-28。范围是 Git 仓库 `new_vv_train/` 当前源码、配置及本机已经生成的 JSUT BASIC5000 数据流程。本文件分析依赖和数据边界，不包含语料文件或训练模型。

## 结论

本项目有四类不同的外部输入：Python 包、固定提交的上游训练/推理源码、最终集成时才需要的本机 VOICEVOX 运行资产，以及训练语料。它们不是同一种“依赖”：例如 JSUT WAV 是训练数据，不是软件包；Open JTalk 字典是前端资源，不是模型 checkpoint；VOICEVOX Engine 服务目录仅供最终服务集成测试，不是训练器依赖。

JSUT 音频、导入后的 `corpora/`、上游源码快照 `vendor/`、派生特征、日志、checkpoint、ONNX/VVM 和注册后的基座均被 Git 忽略，不会随普通 Git 提交同步。当前 v2 语料可以在本机训练，但这份文档的提交不会把它上传到 `origin`。

```text
Python工具与训练器源码
          |
原始JSUT音频 + pinned标签 + Open JTalk字典
          |
manifest / 音素时间标签 / full-context
          |
24kHz音频 + F0 + Mel + 音素张量
          |
duration / intonation / acoustic / NSF HiFi-GAN
          |
三个ONNX -> VVM
          |
可选：VOICEVOX Core / 现有服务目录集成验证
```

## 1. Python 与系统依赖

`pyproject.toml` 声明 Python `>=3.11,<3.13`。服务集成可以使用单独的服务解释器；不要把服务使用的 Python 3.13 环境误认为训练解释器。

| 安装组 | 声明依赖 | 作用与边界 |
|---|---|---|
| 基础安装 | `numpy`、`PyYAML`、`soundfile`、`scipy` | 配置、manifest、WAV读写、重采样与基本数据处理。运行数据导入/校验所需。 |
| `features` | `torch==2.4.1`、`librosa==0.10.2.post1`、`pyworld==0.3.5`、`pyopenjtalk==0.4.1`、`setuptools<81` | Open JTalk前端、Harvest/StoneMask F0、作者Mel特征及数据阶段。这里的 `pyopenjtalk` 是实际由 pip extras 安装并导入的实现。 |
| `training` | Torch/Torchvision、`torch-optimizer`、TensorBoard/TensorBoardX、WandB、`six`、`matplotlib`、`more-itertools` | 上游 duration、intonation、acoustic、HiFi-GAN 训练器及日志。框架为训练子进程设置 `WANDB_MODE=disabled`。 |
| `onnx` | `onnx`、`onnxruntime` | 导出检查及本机 ONNX Runtime CPU 动态长度测试。 |
| `dev` | `pytest` | 自动化测试。 |

完整版本范围见 [`pyproject.toml`](../pyproject.toml)。该文件提供若干直接依赖的版本范围/固定版本，但仓库目前没有完整的 Python 传递依赖 lock 文件；因此不同平台的间接依赖解析结果不保证完全相同。

另有工具和硬件依赖：

- `uv` 出现在安装文档中，是建议使用的环境管理器，不是 Python 运行时依赖。
- `fetch-sources` 调用系统 `patch` 应用仓库内的校验补丁。
- 正式上游 HiFi-GAN 训练路径使用 CUDA 设备；全量配置需要 NVIDIA GPU、兼容驱动和 CUDA 版 PyTorch。本机 CPU GAN smoke 是专用短程验证入口，不等于上游正式训练可在 CPU/MPS 上运行。
- `soundfile`、`pyworld`、`pyopenjtalk` 等包可能包含平台相关二进制组件；需要在目标操作系统/架构的 Python 环境验证安装。

## 2. 固定版本的上游源码

`upstream.lock.json` 锁定 9 个 GitHub 源码快照。`fetch-sources` 按提交下载到 `vendor/`、应用列明的本地 patch，并保存文件清单哈希；`inspect_sources` 会拒绝提交、patch 或快照文件不匹配的情况。源码快照目录不提交到本仓库，锁文件与 patch 文件会提交。

| 上游项目 | 锁定提交 | 本项目用途 | 本地 patch |
|---|---|---|---|
| [`vv_core_inference`](https://github.com/Hiroshiba/vv_core_inference) | `df60099ccd58a54352bf0239b60cb29794385473` | Full-context解析、音素词表兼容、作者PyTorch forwarder与ONNX导出包装。 | 无 |
| [`yukarin_s`](https://github.com/Hiroshiba/yukarin_s) | `50556896d74acd6252b84dc16e58af8cb15cc7d0` | 音素时长预测器及训练入口。 | adapter、Torch 2.4、WandB API补丁；各自SHA见锁文件。 |
| [`yukarin_sa`](https://github.com/Hiroshiba/yukarin_sa) | `2ad1065f94d61e53006e7311086294a8b1407b6f` | mora音高/intonation预测器及训练入口。 | adapter、Torch 2.4、WandB API补丁。 |
| [`yukarin_sosoa`](https://github.com/Hiroshiba/yukarin_sosoa) | `22aa731d56bb5f1c9fe4bfab12b661035cb4a0e2` | 条件声学/Log-Mel预测器及训练入口。 | adapter、Torch 2.4、WandB API补丁。 |
| [`hifi-gan`](https://github.com/Hiroshiba/hifi-gan) | `265da0f0f71923336615d6657f195ffdc6df25e4` | NSF HiFi-GAN声码器、特征函数和GAN训练入口。 | Mel/STFT API适配。上游仓库文件列表中的LJSpeech示例索引不是本项目训练语料。 |
| [`old_yukarin_sosoa`](https://github.com/Hiroshiba/old_yukarin_sosoa) | `38dcd81805549658d2a8a6e871cbc75513154c7a` | 兼容旧声学源码导入路径。 | 无 |
| [`pyopenjtalk`](https://github.com/VOICEVOX/pyopenjtalk) | `74703b034dd90a1f199f49bb70bf3b66b1728a86` | 锁文件中保留的VOICEVOX fork源码快照。 | 无 |
| [`espnet_pytorch_library`](https://github.com/Hiroshiba/espnet_pytorch_library) | `fe0183b78b3495324e6769e340c3bd80f91e7f8e` | 作者训练源码使用的模型/网络辅助库。 | 无 |
| [`pytorch-trainer`](https://github.com/Hiroshiba/pytorch-trainer) | `b4b3d648868e4cec33c69e18fc3877c103a8d438` | 作者duration/intonation/acoustic训练循环。 | NumPy兼容补丁。 |

**pyopenjtalk 实际导入说明：**虽然VOICEVOX fork也在源码锁文件中，`frontend.py` 和 `devoicing.py` 使用当前训练解释器安装的 `pyopenjtalk`；`export.py` 有意不把 `vendor/pyopenjtalk` 加入 worker 的 `sys.path`。因此不能把被锁定的 fork 描述成当前前端必然正在使用的包。实际模块版本、字典路径及字典哈希会随前端记录。若要求逐字节复现读音预测，应单独核对 pip wheel/fork 与目标 Core 使用的字典和版本。

JSUT 标签仓库是另一项固定外部资料，并非模型源码锁中的训练器：`sarulab-speech/jsut-label` revision `1978271ca6212e1ea742da8f149160f5679e8971`，归档 SHA-256 `fbc4c6ef8a86a451915e0007537f25b5fc45f5c8a8078fff5e7a8e0f6110179c`。标签导入器独立校验 revision、归档和所选标签/license文件哈希。

## 3. 本机集成资产：仅最终 Core/服务验证需要

基础语料导入、特征提取和作者训练器运行不需要启动 VOICEVOX Engine 服务。最终 Core 或 service 验证才会访问以下工作区外资产：

| 外部资产 | 何时需要 | 是否训练依赖 |
|---|---|---|
| Open JTalk字典目录，至少含 `sys.dic` | 日语文本前端、无声化候选、最终Core合成。默认配置指向兄弟服务目录的字典。 | 前端/标签处理需要；不是模型权重。 |
| `voicevox_core` Python 0.17.0及其本机二进制 | `validate-core` 打开候选VVM并实际合成。 | 否；导出训练可与Core运行时分开。 |
| Core配套ONNX Runtime native library与assets目录 | Core验证时加载ONNX模型。默认native文件名按原工作区macOS ARM64路径写定，Linux需显式改成对应`.so`。 | 否。Python包 `onnxruntime` 用于工具侧图验证，和Core native Runtime不是同一个依赖。 |
| 现有模型目录/VVM | 检查 public style ID 冲突及候选与现有角色共存。 | 否。已有VVM不是PyTorch训练基座。 |
| `local_engine.py` 服务适配器及服务目录 | 仅 `validate-service` 检查已有服务能否发现并合成候选声音。 | 否；它是特定本地服务的集成目标，不是通用服务依赖。 |

配置字段及迁移要求见 [`configs/project.yaml`](../configs/project.yaml)、[`RUNBOOK.md`](../RUNBOOK.md) 和 [`core_validation.py`](../src/new_vv_train/core_validation.py)。当前源码没有导入 `voicevox_resource`，也不需要完整 `voicevox_engine` Python 包来训练；Engine/Resource源码可作为项目外部协议、行为参考，不能因此列为训练必需依赖。

## 4. 哪些是真正的训练数据

本机使用的是 JSUT BASIC5000 单人日语语料。音频是训练样本；标签、转写和分割是监督信息；profile与训练参数是配置，不是样本。模型由配置中的 `scratch` 从随机权重训练，不把已有角色VVM/ONNX当作初始化权重。

| 数据层 | 内容与用途 | Git/当前状态 |
|---|---|---|
| 原始语料 | JSUT `basic5000/wav` 的5,000条原始单声道录音，以及 `transcript_utf8.txt`、录音信息。当前配置使用仓库外的本机数据根目录；具体机器绝对路径不写入本报告。 | 位于外部数据盘；不进Git。导入器复制使用的原音频到本机 `corpora/.../raw/`。 |
| 语料说明/许可 | JSUT音频许可文件与标签仓库许可证分开保存。`basic5000`文字许可包含Wikimedia CC BY-SA 3.0、TANAKA CC BY 2.0及原始句子CC BY-SA 4.0；标签README也列明修订文字及发音信息的许可来源。 | `corpora/.../licenses/` 本机保留，目录被忽略。二次使用应分别核对原始音频和标签的完整许可、署名及共享条款；“个人研究”不会自动取消原许可证要求。 |
| JSUT人工标签 | pinned `jsut-label` 的音素/重音full-context、Julius估计时间，以及 `text_level2` 实际发音文本。 | `vendor/jsut-label/` 是下载缓存，不是训练器依赖，也不提交。 |
| 导入语料v1 | `manifest.jsonl`、`raw/`、full-context、秒制音素时间对齐、speakers、split、license和receipt。有效4,999句；排除一条时间端点偏差过大的录音。 | `corpora/jsut_basic5000_v1/` 被忽略。本机存在，不随Git推送。 |
| 标注语料v2 | 从v1复制出的独立版本；自动增加1,781个 `I/U` 大写候选，68,960个元音位置待复核，尚无人工审核决定。 | `corpora/jsut_basic5000_v2/` 被忽略。本机存在，是当前全量特征流程输入；不能宣称为人工金标准。 |
| 训练/验证/测试划分 | v1/v2均为训练4,799、验证100、测试100。materialize只输出训练和验证数据；测试集不进入模型训练视图。 | 保存在各语料的 `splits/`；随corpus一起本地留存，不提交音频。 |
| 基础配置/profile | 24kHz、hop256、93.75帧/秒、45类固定顺序音素、F0/Mel参数和模块训练配置。 | `configs/` 是源代码配置，会提交；其中JSUT配置含机器相关绝对源路径/兄弟服务路径，迁移到新机器前必须修改。 |

准确的v2统计、特征及模型验证边界见 [`JSUT_V2_VALIDATION_2026-10-27.md`](JSUT_V2_VALIDATION_2026-10-27.md)。自动标注不是模型依赖，也不改变音频；它改变监督标签并应作为新的语料版本管理。

## 5. 派生数据与训练产物

以下是从训练数据生成的本地文件，不是独立外部依赖或原始语料：

| 路径 | 内容 | 用途 |
|---|---|---|
| `corpora/<id>/derived/audio24k/` | 原始音频的24kHz单声道训练副本，不裁切或按句响度归一化。 | 特征提取与声码器输入。 |
| `derived/frontend/`、`derived/full_context/`、`derived/phoneme_list/` | 字典前端上下文、结构化音素、秒制对齐和音素序列。 | 时长、音高、声学训练器。 |
| `derived/f0/`、`derived/f0_hz/`、`derived/spec/`、`derived/phoneme/`、`derived/volume/`、`derived/silence/` | log-F0、诊断Hz、80维自然对数Mel、45维one-hot及辅助帧特征。 | 原生训练特征；经 `materialize` 转成各上游loader期望的本地NPY视图。 |
| `runs/<run-id>/native/` | `yukarin_s`、`yukarin_sa`、`yukarin_sosoa`、HiFi-GAN四种原生数据目录与配置。 | 上游训练器读取的数据适配视图。 |
| `runs/<run-id>/native-output/`、`models/` | Optimizer snapshot、模型权重、配置和训练日志。 | 中断恢复、评估与ONNX导出。 |
| `runs/<run-id>/onnx/`、`vvm/` | 三张ONNX、验证报告和候选VVM。 | 导出和本地集成测试；只有通过验证的训练结果才应作为发布候选。 |
| `bases/<base-id>/` | 显式注册的完整四模块PyTorch基座、profile、来源和训练凭证。 | 后续兼容配置的微调起点。当前JSUT短程验证没有生成可用基座。 |

`derived`与`runs`通常位于corpus/run配置路径下；即使配置名不在顶层`runs/`，也仍是本机生成物，不因路径命名自动变成源数据。

## 6. Git提交边界与可复现性

[`.gitignore`](../.gitignore) 排除 `.venv/`、`vendor/`、`corpora/`、`bases/`、`runs/`、build产物及本地环境文件。仓库应跟踪的是程序、配置模板、上游提交与patch哈希、文档、测试和schema；训练音频、许可证副本、派生数组、checkpoint、ONNX/VVM等大型/本机数据应另行按数据治理要求保存和备份。

当前复现边界：

1. 上游Git源码有提交号和逐文件SHA检查；JSUT标签归档有固定revision/SHA。
2. Python直接依赖有版本声明，但没有完整传递依赖锁；CUDA驱动、Python wheel及本机Core二进制也要按机器重新确认。
3. 语料导入、训练run、特征及checkpoint都有本机收据/哈希，能检测输入或输出变化；这些收据随被忽略的数据目录保存在本机。
4. 配置中绝对的JSUT源路径、Open JTalk字典、Core Runtime、已有模型和服务目录都需要在另一台机器重配。不能把路径恰好存在或dry-run成功称为跨平台已验证。
5. 报告在 `main` 分支下提交时，只包含Markdown文件；被忽略的 `corpora/`、`vendor/`、`runs/`、`bases/`不会被Git推送。

## 主要本地依据

- [`pyproject.toml`](../pyproject.toml)：Python版本范围、依赖组及CLI入口。
- [`upstream.lock.json`](../upstream.lock.json)：9个上游仓库、固定提交、patch及其SHA-256。
- [`patches/`](../patches/)：本项目维护的兼容补丁，不是上游原始源码。
- [`configs/feature_profile.json`](../configs/feature_profile.json)：音频、音素、F0、Mel与帧率契约。
- [`configs/jsut_basic5000_v1`配置](../configs/jsut_basic5000.yaml)、[`v2配置`](../configs/jsut_basic5000_v2.yaml)：源路径、训练预算和语料版本。
- [`src/new_vv_train/jsut.py`](../src/new_vv_train/jsut.py)：JSUT标签固定版本、导入、许可收据、原音频复制与标签策略。
- [`src/new_vv_train/features.py`](../src/new_vv_train/features.py) 与 [`training.py`](../src/new_vv_train/training.py)：派生特征和上游训练视图。
- [`src/new_vv_train/core_validation.py`](../src/new_vv_train/core_validation.py)：Core与现有service是集成验证目标，而非数据训练器。
- [`docs/JSUT_BASIC5000_TRAINING.md`](JSUT_BASIC5000_TRAINING.md)、[`docs/JSUT_V2_VALIDATION_2026-10-27.md`](JSUT_V2_VALIDATION_2026-10-27.md)：本机语料和训练验证记录。

上游一手入口：[`vv_core_inference`](https://github.com/Hiroshiba/vv_core_inference)、[`yukarin_s`](https://github.com/Hiroshiba/yukarin_s)、[`yukarin_sa`](https://github.com/Hiroshiba/yukarin_sa)、[`yukarin_sosoa`](https://github.com/Hiroshiba/yukarin_sosoa)、[`hifi-gan`](https://github.com/Hiroshiba/hifi-gan)、[`JSUT label repository`](https://github.com/sarulab-speech/jsut-label)。每次实际使用的提交号以本仓库锁文件为准。
