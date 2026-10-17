# 基座与作者源码复用策略

## 1. 现有资源的真实状态

当前未确认存在公开、可直接用于新增正式 VOICEVOX 音色的高质量 PyTorch 基座。作者 `vv_core_inference` 的演示权重明确声明接近噪声，不作为有用的音色微调基座。已有正式 VVM 是推理产物，不当作提供了训练参数/优化器状态的 checkpoint。

本框架设计不捏造基座下载地址。代码可从固定的作者仓库取得；训练初始化分为 scratch、自有/授权 pretrained predictor、同一实验完整 resume。三者须分别登记。

## 2. 锁定作者实现

以 `vv_core_inference` 的 pyproject 依赖版本为起点，不混用各仓库最新 HEAD：

| 组件 | 仓库 | 固定提交 | 用途 |
| --- | --- | --- | --- |
| 推理/导出 | Hiroshiba/vv_core_inference | df60099ccd58a54352bf0239b60cb29794385473 | forwarder、前端、导出包装器 |
| 时长 | Hiroshiba/yukarin_s | 50556896d74acd6252b84dc16e58af8cb15cc7d0 | CNN、Dataset、train.py |
| 音高 | Hiroshiba/yukarin_sa | 2ad1065f94d61e53006e7311086294a8b1407b6f | AR GRU、Dataset、train.py |
| 声学 | Hiroshiba/yukarin_sosoa | 22aa731d56bb5f1c9fe4bfab12b661035cb4a0e2 | Conformer/Postnet、Dataset、train.py |
| 声码器 | Hiroshiba/hifi-gan | 265da0f0f71923336615d6657f195ffdc6df25e4 | F0/NSF Generator、Mel 提取、GAN 训练 |
| 导出历史依赖 | Hiroshiba/old_yukarin_sosoa | 38dcd81805549658d2a8a6e871cbc75513154c7a | 原导出工程历史模型依赖，按入口保留 |
| 日语前端 | VOICEVOX/pyopenjtalk | 74703b034dd90a1f199f49bb70bf3b66b1728a86 | 作者推理工程锁定的前端 |

参考导出工程声明 Python >=3.11、torch 2.4.1、onnx 1.14.0、onnxruntime 1.15.1、numpy <1.26.4。**这不是已经验证的统一训练环境**：旧训练库、STFT 与 librosa API 还需实际检查。首选 Linux Python 3.11 起步，训练模块可分别使用隔离环境；服务 Python 3.13 不作为训练环境。

下载后保留每个上游 LICENSE、提交号和本地补丁。若依赖修复改变数值行为，必须留下原函数对照测试。优先避免修改网络层、损失、参数命名和 Core 语义。

## 3. 基座登记格式

使用 `bases/<base_id>/base.json`，以下是格式说明而不是随项目提供的真实权重：

```json
{
  "schema_version": 1,
  "base_id": "owned-ja-base-v1",
  "origin": "self_trained",
  "feature_profile": "feature_profile.json",
  "voice_registry": "voices.json",
  "components": {
    "duration": {"config": "duration/config.yaml", "weights": "duration/model.pth"},
    "intonation": {"config": "intonation/config.yaml", "weights": "intonation/model.pth"},
    "acoustic": {"config": "acoustic/config.yaml", "weights": "acoustic/model.pth"},
    "vocoder": {"config": "vocoder/config.json", "weights": "vocoder/model.pth"}
  },
  "rights_record": "rights.md"
}
```

实际登记时补充每个权重/config 的 SHA-256、上游提交/补丁哈希、训练数据版本、训练步骤和模型品质记录。加载权重前验证文件存在、校验和、网络配置、音素表、声学 profile、speaker 表形状；缺任何一个模块不能默认为已找到完整基座。

初始化只接受来源明确的本地 PyTorch checkpoint；产品 ONNX/VVM 不转换为可训练基座。自有语料从零训练出的四模块，可以整理并登记成下一轮可复用基座。

## 4. 首轮训练和适配策略

### scratch：没有可用基座时

固定网络和特征配置后随机初始化四模块，用覆盖足够的获授权日语语料训练。先在小子集验证读取、优化、保存和导出，再扩展；短训练只证明链路，不能证明模型品质。从零声码器和声学模型通常比有良好基座的适配更依赖数据，不能直接套用“400 句即可”的保证。

若需要建立通用基座，可以先用允许训练的公开日语录音或自有多说话人素材做预训练，再加入目标人数据。JSUT/JVS 等音频的许可和预期用途须单独记录；ROHAN 文本 CC0 不自动授权任意朗读音频。

### finetune：已有匹配基座时

首版优先支持**配置和 speaker 表形状不变的单说话人适配**：输出新模型和新 VVM，旧 VVM 原样保留。此方法可以重用单人基座的内部 0 号条件，不声称保留同一新模型中的旧人声音。

在一个新多说话人模型中保留旧声音并新增 embedding 行，是后续独立策略：需扩充三个模块的 speaker 表、复制旧行、初始化新行、维护稳定 ID，并用可访问的旧语料或其他已验证方法控制遗忘。仅增加一行随机向量不产生目标音色；这不作为首版微调默认行为。

声码器应先测试基座在目标人的真实 spec/F0 条件下能否重建；匹配时可冻结，不匹配时再适配。真实频谱到波形的预训练与预测 spec 到波形的 fine_tuning 分开记录。

### resume：恢复同一次训练

使用上游 snapshot 保存的模型、优化器、调度器和步骤恢复；先验证数据/config/特征 profile 一致。载入 predictor 或只载入 GAN generator 不等于完整 resume，不复用过期的“阶段成功”标记代替检查。

## 5. 兼容性不能从基座名称推断

只要输入输出契约、音素顺序、F0/帧率与模型执行满足 Core，可以使用不同内部网络；本次选择作者网络是为了减少偏离。但参数规模、embedding 维度、Mel 维数、训练步数不是正式产品已公开的固定值，必须写在所选模型配置里。

首版不把未经质量验证的作者演示权重标为 `pretrained-ready`。doctor 应分别报告 `source-ready`、`data-ready`、`base-ready`、`native-synthesis-validated`、`core-validated` 等具体检查结果，而不是一个模糊的“已就绪”。

## 6. 一手来源

- [作者参考权重限制](https://github.com/Hiroshiba/vv_core_inference/blob/df60099ccd58a54352bf0239b60cb29794385473/README.md)
- [上游精确依赖版本](https://github.com/Hiroshiba/vv_core_inference/blob/df60099ccd58a54352bf0239b60cb29794385473/pyproject.toml)
- [duration 配置](https://github.com/Hiroshiba/yukarin_s/blob/50556896d74acd6252b84dc16e58af8cb15cc7d0/yukarin_s/config.py)
- [intonation 配置](https://github.com/Hiroshiba/yukarin_sa/blob/2ad1065f94d61e53006e7311086294a8b1407b6f/yukarin_sa/config.py)
- [acoustic 配置](https://github.com/Hiroshiba/yukarin_sosoa/blob/22aa731d56bb5f1c9fe4bfab12b661035cb4a0e2/yukarin_sosoa/config.py)
- [HiFiGAN 训练入口](https://github.com/Hiroshiba/hifi-gan/blob/265da0f0f71923336615d6657f195ffdc6df25e4/train.py)
