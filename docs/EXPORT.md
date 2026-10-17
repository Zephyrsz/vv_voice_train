# 导出、VVM 与实际 Core 验证

这些命令需要真实四模块 checkpoint。当前仓库没有目标人的录音或可用的已训练权重；单元测试里的 ONNX 是契约夹具，不是训练结果。

```sh
.venv/bin/new-vv-train --config configs/project.yaml evaluate-native
.venv/bin/new-vv-train --config configs/project.yaml export
.venv/bin/new-vv-train --config configs/project.yaml validate-onnx
.venv/bin/new-vv-train --config configs/project.yaml package
.venv/bin/new-vv-train inspect-vvm runs/first_run/vvm/candidate.vvm
.venv/bin/new-vv-train --config configs/project.yaml validate-core
.venv/bin/new-vv-train --config configs/project.yaml validate-service
```

CLI 全局选项的位置以 `new-vv-train --help` 为准。Python API 分别为 `export.evaluate_native(cfg)`、`export.export_models(cfg)`、`export.validate_onnx(cfg)`、`vvm.package(cfg)`、`vvm.inspect_vvm(path)`、`core_validation.validate_core(cfg)`、`core_validation.validate_service(cfg)`。

## 输入与环境

`paths.run/models/{duration,intonation,acoustic}` 各需 `config.yaml` 和 `model.pth`；`vocoder` 需 `config.json` 和 `model.pth`。前三者是作者 predictor 的 state_dict，声码器是含 `generator` 的字典。完整训练 snapshot 需由训练阶段按作者保存格式提取，不能重命名冒充。

导出使用 `export.python`，默认当前 Python。它需要 Torch、ONNX、ONNX Runtime、NumPy、PyYAML 及锁定作者代码的依赖；其模型定义来自经完整性验证的 `vendor` 源码。历史 acoustic wrapper 还导入 `old_yukarin_sosoa` 和它锁定的 `espnet_pytorch_library`，缺依赖会真实失败并记录日志。不会下载产品权重或从 VVM 恢复训练参数。

`feature_profile` 文件控制 24 kHz / hop 256、F0 和有序音素表；导出逐项对比作者 OjtPhoneme 表。三个 speaker table 大小必须相同，声码器上采样率乘积必须为 256。profile 的频谱定义必须与训练一致；这不是正式产品的内部模型规格。

```yaml
export:
  python: .venv/bin/python
  device: cpu
  opset: 17
  nsf_policy: allow-stochastic
  rtol: 0.001
  atol: 0.0001
core:
  version: 0.17.0
  python: ../vv_tts_ws_server/.venv/bin/python
  assets: ../vv_tts_ws_server/assets
  existing_models: ../vv_tts_ws_server/assets/models
validation:
  text: 今日はいい天気ですね。
  timeout_seconds: 300
```

解释器路径保留 `venv/bin/python` 链接本身，避免将它解析成基础 Python 后丢失虚拟环境。

## 导出验收

原生评估调用作者完整 Forwarder 从测试文本合成，保留 WAV、波形统计和中间数组形状。编译型 pyopenjtalk 使用导出解释器安装的模块，报告实际版本/路径；未经证明与锁定提交一致时明确记录 `frontend_pin_verified=false`。它只证明实际 PyTorch 推理运行，不能替代保留集误差评估或听测。

ONNX 使用作者 duration/intonation/acoustic/vocoder wrappers。intonation 采用 TorchScript 保留自回归循环；decode 只串联 acoustic 和 vocoder 并返回一维 `wave`，不返回辅助 `spec`，不声明 streaming_talk。

每个 ONNX 运行 checker、输入输出名称/类型/shape 检查和多个动态长度的实际 CPU Runtime 推理；每个公开风格关联的内部 ID 都测试。duration 必须为正，输出必须 float32/有限，decode 必须为 `T*256` 个样本且非全零。另在三个长度比较 PyTorch/ONNX 数值。

带 NSF 的声码器包含随机激励。默认 `nsf_policy: reject` 会阻止未明确处理的导出；显式 `allow-stochastic` 接受随机模型时仍检查动态长度、shape 与有限输出，duration/intonation 仍比较数值，但 **不比较 decode 波形逐样本一致性**。报告的 `parity_checked=false` 和 `max_abs_error=null` 明示此限制。未修改随机源，也不宣称已实现确定性激励。

产物在 `run/onnx/`，含 `validation.json`、原生配置/权重哈希、源版本记录和 `export_receipt.json`。子进程失败时查 `export.log`；不存在训练文件时在导入 Torch 前失败。

## 打包与静态检查

打包重新执行 Runtime 验证，确认 profile、三 ONNX 的 SHA-256 与成功导出记录一致，要求导出记录含来源检查及原生模型哈希。`corpus/speakers.json` 显式提供 public_style_id 与 inner_voice_id。public ID 不得重复，不得与 `core.existing_models` 的任何风格冲突；inner ID 必须落在导出 speaker table 范围内。

`run/vvm/identity.json` 持久化独立 model UUID 和每 speaker_key 的 speaker UUID；首次可用 `package.model_uuid`、`package.speaker_uuids` 显式指定。一个 speaker 可有多个 talk 风格。只写 format 2 talk manifest、metadata 和三 ONNX，ZIP 先写临时文件，通过静态检查后原子替换候选。

inspect 不解压文件，拒绝路径逃逸、反斜杠/绝对路径、重复成员、符号链接、超大 metadata、缺文件、非法 UUID、重复 public ID、非 talk 域，以及 metadata 与 style mapping 不一致。它检查结构；不等于 Core 已接受模型。

## 实际 Core 和服务

Core 验证使用 `core.python` 的独立进程，加载 `core.assets` 下当前 Runtime 和字典，实际 open/load/query/synthesis；检查每风格 WAV 的 24 kHz、mono、16-bit、非空、非全零并保留 WAV。报告记录真实 Python/Core 版本、扩展库路径/哈希、Runtime 路径/版本/哈希和候选 VVM 哈希。

服务验证从现有服务目录导入 `local_engine.py`，创建仅含已有 VVM 链接与候选链接的临时目录。检查 get_speakers 可见性，合成每个候选风格及一个已有风格。它不会替换服务资产、改服务配置或重启服务。当前 adapter 内部固定资产布局，改用另一种布局需显式适配服务入口。

失败日志及真实 traceback 保存在 `run/validation/{core,service}`；失败状态不会变成兼容通过。若 `core.target: custom-core-only`，报告保留该目标；自建 Core 成功不能代表现有服务已兼容。质量验收仍需真实目标录音和人工试听。

## 本次工具验证

自动测试覆盖 3 个动态长度、错误 ONNX 类型/固定长度、缺权重、非法 ZIP 路径/UUID、重复 ID、mapping 不完整、导出哈希变化、既有模型冲突及空/静音/错误采样率 WAV。另在临时目录用 45 音素 synthetic ONNX 实际运行当前 Core 0.17.0 / Runtime 1.17.3：候选输出 25,344 samples @24 kHz；服务 adapter 同时合成候选及已有音色（22,528 samples）。这只验证格式和加载/推理链路，不是作者网络训练或声音品质结果。

另外使用锁定作者网络的微型随机权重夹具完成真实导出：Torch 2.4.1 的 duration/intonation 在三种长度与 ONNX 数值一致，带 NSF decode 通过动态长度和有限值检查。完整作者 Forwarder 实际输出 47,872 samples @24 kHz；前端是安装的 pyopenjtalk 0.4.1，未声称锁定提交已验证。随机权重夹具已删除，未作为基座或训练成果交付。
