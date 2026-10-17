# new_vv_train

这是一个面向个人研究的日语音色训练框架。它保留作者的 `yukarin_s`（时长）、`yukarin_sa`（音高）、`yukarin_sosoa`（声学）和 `hifi-gan`（NSF 声码器）源码，通过数据契约、特征收据、原生训练器和 ONNX/VVM 验证，把新音色接入现有 VOICEVOX Core 0.17.0 talk 服务。

当前目录包含可执行的数据阶段、原生配置生成、训练调用、导出和 VVM/Core 检查；作者源码由固定版本锁下载到本地。仓库不包含目标录音或训练基座。运行 `scripts/smoke_cpu.py` 后生成的 `runs/diagnostics/cpu-smoke.json` 是合成音调的链路报告，不是音色结果。

源码分析与训练路线总结见 [VOICEVOX_ROLE_TRAINING_ANALYSIS.md](docs/VOICEVOX_ROLE_TRAINING_ANALYSIS.md)。执行步骤见 [RUNBOOK.md](RUNBOOK.md)，素材字段见 [DATA_FORMAT.md](DATA_FORMAT.md)，作者模块和基座规则见 [BASE_MODELS.md](BASE_MODELS.md)。

## 快速开始

```sh
git clone https://github.com/Zephyrsz/vv_voice_train.git
cd vv_voice_train
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev,features,training,onnx]'
.venv/bin/new-vv-train --help
.venv/bin/new-vv-train fetch-sources
.venv/bin/new-vv-train init-corpus
```

已有本地 checkout 时直接进入其根目录。默认配置的字典、Core、Runtime 和服务路径引用原开发工作区的兄弟目录，独立克隆后须按 [RUNBOOK.md](RUNBOOK.md) 配置；这些外部资产不随仓库发布。

把真实日语 WAV、manifest 和 split 写入 `corpora/new_voice_v1/` 后执行：

```sh
.venv/bin/new-vv-train validate-corpus
.venv/bin/new-vv-train prepare-audio
.venv/bin/new-vv-train frontend
# 人工检查前端读音/重音，再准备 alignment_input/*.lab
.venv/bin/new-vv-train import-alignment
.venv/bin/new-vv-train extract-features
.venv/bin/new-vv-train materialize
.venv/bin/new-vv-train train duration
.venv/bin/new-vv-train train intonation
.venv/bin/new-vv-train train acoustic
# vocoder 需要 NVIDIA CUDA 环境
.venv/bin/new-vv-train train vocoder
.venv/bin/new-vv-train evaluate-native
.venv/bin/new-vv-train export
.venv/bin/new-vv-train validate-onnx
.venv/bin/new-vv-train package
.venv/bin/new-vv-train validate-core
.venv/bin/new-vv-train validate-service
```

任何阶段输入或哈希不一致都会失败并要求重新生成受影响阶段。最终 Core/service 检查只证明接口、加载和推理链路；自然度、说话人相似度和语料充分性仍需真实数据上的保留集评估与试听。
