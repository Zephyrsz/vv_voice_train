# 原生 VOICEVOX 训练框架设计

## 1. 目标和首版范围

采用作者公开网络，建立可追踪的“语料 → 派生特征 → 四模块训练 → PyTorch 合成 → ONNX → VVM → Core/服务验证”流水线。首版面向一个日语说话人、一个中性风格，数据格式预留多人/多风格。

首版固定 Core 0.17.0、VVM format 2、普通 `talk` 域：

- `predict_duration.onnx`：音素 ID、speaker ID → 音素时长（秒）。
- `predict_intonation.onnx`：mora 音素、四种重音/短语标记、speaker ID → mora log-F0。
- `decode.onnx`：逐帧音素 one-hot、log-F0、speaker ID → 浮点波形。
- 波形 24,000 Hz，条件帧率 93.75 Hz，即 hop 256。

兼容性与模型品质分别验收。文件能打包、模型能加载、输出声音自然是三个不同状态。

不在首版引入歌唱、参考音频克隆、LoRA 或 embedding-only 微调。streaming_talk 另作后续接口适配；不能仅把分离模型放进 manifest 就声明支持流式。

## 2. 方案比较与选择

| 方案 | 优点 | 代价 | 决策 |
| --- | --- | --- | --- |
| 锁定作者源码，增加薄适配层 | 最接近作者网络，差异可审计 | 历史依赖与数据准备需要整合 | 首选 |
| 在本项目重写同结构网络 | 统一现代训练栈 | 参数命名、数值行为、导出容易偏离 | 不作为首版 |
| ESPnet/SBV2 独立后端 | 公共训练配方更完善 | 不保留原模块与 VVM 契约 | 与本次目标不符 |

“尽量复用”优先指复用作者源码及已有 Core/服务代码，不复用行为不完整的空实现。

## 3. 现有代码如何复用

| 已有路径 | 复用内容 | 处理方式 |
| --- | --- | --- |
| `../source_backup/local_dependecy/voicevox_core-0.17.0/` | talk 签名、音素表、mora/重音处理、VVM manifest、样例 | 作为契约与回归参照；训练标签须与之比对 |
| `../vv_tts_ws_server/assets/dict/` | 日语字典 | Core 最终测试使用现有字典；训练前端另记录自己的字典版本，不默认两者相同 |
| `../vv_tts_ws_server/assets/onnxruntime/` | macOS ARM64 推理运行时 | 最终服务端合成验证复用 |
| `../vv_tts_ws_server/local_engine.py` | 模型扫描、style ID 映射、Core 推理接口 | 用独立候选模型目录调用；正式接入前与现有模型一起检查 ID 冲突 |
| `../voicevox_engine/voicevox_engine/tts_pipeline/` | 日语音素/mora/重音处理参考 | 优先以作者导出工程为实现源，Engine 作为测试比对；不直接导入整个服务依赖 |
| `../role_train/scripts/data_prepare.py` | 音频转换/质量统计的思路 | 现有相对 dB Mel、近似 10.67 ms F0、纯数组命名不符合本方案契约，不照搬特征实现 |
| `../role_train/scripts/scheduler.py` | 阶段依赖概念 | 不复用返回成功的训练占位逻辑；缺产物或命令失败必须终止 |

不复制大体积产品 VVM、字典、Runtime 或现有服务虚拟环境。

## 4. 模块边界与目录

以下是已实现目录。作者网络位于本地下载的 vendor 快照中，编排代码位于 src：

```text
new_vv_train/
├── README.md / DESIGN.md / DATA_FORMAT.md / BASE_MODELS.md
├── pyproject.toml                 # 编排/数据工具环境，独立于服务 Python 3.13
├── configs/
│   ├── project.yaml               # 数据、运行目录、Core、公开/内部 ID
│   ├── feature_profile.json       # 音素表、F0/频谱参数与配置哈希
│   └── modules/                   # 四个作者训练器的真实原生配置
├── upstream.lock.json             # 作者源码 URL、固定提交与本地补丁
├── src/new_vv_train/
│   ├── cli.py                     # 明确的阶段命令
│   ├── corpus.py                  # manifest、音频质量、集合划分
│   ├── frontend.py                # 作者日语前端；导出标签以便人工修正
│   ├── alignment.py               # 导入强制对齐；检查音素序列和时间
│   ├── features.py                # 作者频谱函数与一致帧时基
│   ├── upstream.py                # 固定源码下载、补丁及完整性校验
│   ├── training.py                # 真实子进程与 checkpoint 整理
│   ├── export.py                  # 原导出器 + 合并 decode 适配
│   ├── vvm.py                     # talk manifest/metas、ZIP 打包
│   ├── core_validation.py         # Core、服务的实际隔离验证
│   ├── doctor.py                  # 环境与输入检查
│   └── common.py                  # 输入/配置/产物哈希与状态记录
├── tests/                         # 有意义的数据与兼容性回归
├── patches/                       # 最小依赖/入口/导出修复及原因
├── vendor/                        # 按锁定版本取得上游源码，保留许可证
├── corpora/<dataset_id>/           # 用户素材与派生数据（忽略提交）
├── bases/<base_id>/                # 自有/授权基座（忽略权重）
└── runs/<run_id>/                  # 配置、日志、产物、报告（忽略大文件）
```

编排层只管理输入输出和边界；网络类、损失及训练循环优先由上游提供。依赖冲突时允许每个训练模块指定独立 Python 可执行文件，通过磁盘契约交换数据，不强行塞进服务虚拟环境。

训练默认环境目标是 Linux + NVIDIA CUDA；macOS 负责素材检查、ONNX/Core CPU 验证。具体 Python/CUDA/显存必须随依赖与模型实测，不继承 SBV2 batch 1≈6GB 的参考数字。

## 5. 阶段及阻断条件

| 阶段 | 输入 | 产物及必须通过的检查 |
| --- | --- | --- |
| doctor | 配置、上游锁、环境、资产路径 | 记录依赖版本、GPU、Core；区分缺原料、缺基座、缺依赖 |
| validate-corpus | manifest、原始 WAV | 逐条质量报告、有效时长、重复 ID/文本/音频及集合泄漏检查 |
| prepare-audio | 合格素材 | 24 kHz mono 训练副本、重采样记录；原件不覆盖 |
| frontend | 准确文本/读音 | full-context labels、音素序列、重音标记、修正记录 |
| import-alignment | 外部强制对齐结果 | 秒制音素边界；序列、连续性、末尾时间与音频一致 |
| extract-features | 音频、前端、对齐、profile | log-F0/volume/silence/phoneme/spec，统一 stem 与 rate |
| materialize | 合格特征、固定集合、speaker 映射 | 四模块可实际加载的数据视图与配置；运行上游 dataset 样本检查 |
| train | 数据视图、scratch/基座、模块配置 | 子进程退出码为 0、有效 checkpoint、日志、loss/验证结果 |
| evaluate-native | 四个模型 | 保留测试句合成；时长/F0/频谱误差、听测文件、稳定性记录 |
| export | 合格四模块、真实配置 | duration/intonation/decode ONNX；动态长度、多输入比较与数值报告 |
| package | ONNX、公开/内部 ID、角色信息 | 新 UUID 的 talk-only VVM、校验和、manifest/metas 静态校验 |
| validate-core | VVM、明确指定的 Core/Runtime/字典 | 打开、加载、创建 query、合成并检查 WAV；记录实际二进制路径 |
| validate-service | 合格 VVM + 现有模型集合 | 唯一 style ID、get_speakers 可见、synthesize 正常；候选与现有音色共存 |

首版的对齐输入是“已由外部工具生成的秒制标签”，不伪装已集成任意日语 aligner。可先使用支持目标日语音素集的 Julius 工具链，再导入验证；MFA 需要另行确认日语声学模型、词典和音素映射。手工修正边界可以作为小规模验证手段，不建议全量手标。

各阶段成功由有效产物判定，不是进程启动或配置解析成功。显式 `--dry-run` 只打印命令，不推进训练完成状态。恢复执行必须重新检查输入/配置/上游版本哈希；训练器自己的 resume 与“载入 predictor 初始化”分开处理。

## 6. 四个训练模块的实际连接

1. `yukarin_s/train.py <config_yaml_path> <output>`：时长 CNN。该锁定版本的 TrainConfig 没有通用 pretrained_predictor_path；微调需要在 trainer 创建模型处加入可审计的初始化适配，不能传不存在的参数。
2. `yukarin_sa/train.py <config_yaml_path> <output>`：mora F0 编码器 + AR GRU。采用其已有 pretrained_predictor_path；保留 teacher-forcing 训练与自回归推理差异。
3. `yukarin_sosoa/train.py <config_yaml_path> <output_dir>`：Conformer + Postnet。支持既有 predictor 初始化；完整 snapshot resume 另按上游行为执行。
4. `hifi-gan/train.py`：使用真实 `--input_wavs_dir`、`--input_f0_dir`、训练/验证列表和 config；明确配置 `with_hn`。区分真实 Mel 训练、预测 spec 微调和完整 G/D/optimizer 恢复；不同 checkpoint 格式不混用。

时长和音高训练不依赖彼此训练产物；声学模型可使用录音真实时长/F0 条件训练；声码器也可使用真实频谱独立训练。为便于首轮诊断，调度器默认顺序执行四模块，不宣称存在必须训练 AR Tacotron 教师的依赖。

各模块保存 checkpoint 后，导出适配层整理成作者 forwarder 期望的 `config.yaml + model.pth`（声码器 `config.json + model.pth`）。checkpoint 选择与 state_dict 解包规则须按实际上游保存代码校验，不能只重命名 snapshot。

## 7. ONNX、VVM 和服务兼容

作者当前 `convert.py` 分别导出 duration、intonation、spectrogram、vocoder；本框架新增一个合并导出适配器，复用其 acoustic/vocoder wrapper，调用后只返回 `wave`，从而符合普通 talk 的 `decode` 输出。不能直接导出返回 `(spec, wave)` 的辅助 wrapper 后冒充普通 talk。

Core 0.17.0 的 `streaming_talk.render_audio_segment` 只接收 `spec`，但作者分离 vocoder 接收 `spec + F0`。首版不把它直接注册为 streaming；未来可以设计携带 F0 的中间表示适配，但必须重新验证契约。

ONNX 必须满足以下实际签名：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| duration | phoneme_list int64 `[N]`；speaker_id int64 `[1]` | phoneme_length float32 `[N]` |
| intonation | length int64 标量；母音/辅音与四个标记 int64 `[M]`；speaker_id int64 `[1]` | f0_list float32 `[M]` |
| decode | f0 float32 `[T,1]`；phoneme float32 `[T,P]`；speaker_id int64 `[1]` | wave float32 `[S]` |

名称必须使用 Core 字段对应名称；`length=M`，不是秒。`P` 由锁定音素表生成，不能按训练语料出现次数重建或排序。ONNX 检查要覆盖导出样例之外的 N/M/T，避免把 AR 循环或音频长度固化。

VVM 使用 format 2、独立 model UUID/speaker UUID、单一 talk 域，显式设置 `style_id_to_inner_voice_id`。公开 style ID 与内部 embedding 行号分开记录。新建模型先保存在 `runs/<run_id>/vvm/`，Core 通过后再进入本地服务候选目录，不自动覆盖产品 VVM。

Core 样例说明指出其样例不能用于 Releases 预编译 Core，故首版设置两个不同验证目标：

- 自建 Core 0.17.0：自制 ONNX VVM 的开发验证基线。
- 当前 `vv_tts_ws_server` wheel/Core：现有服务真正需要通过的集成目标。

若只通过前者，报告状态为 `custom-core-only`，不得写“现有服务兼容”。不得自动切换现有服务 Core；如需自建 Core 接入，再提供单独可回退的候选环境。

## 8. 验收与实施顺序

1. **先完成数据/基座检查器**：缺录音、对齐、基座或不匹配配置时给出明确失败；完成秒制标签、log-F0、音素表、集合泄漏回归测试。
2. **再接通真实上游训练**：用少量有真实录音的测试集完成数据加载和短训练，验证权重发生有限且非零变化、checkpoint 可重新载入；短训练不作为音质达标。
3. **再验证原生合成**：测试真实条件声学重建、预测 F0/时长完整合成，定位问题所在模块。
4. **最后导出和兼容**：ONNX 与 PyTorch 在同一条件下比较；确定 NSF 随机激励策略后设置对应数值容差；验证 VVM 和两种 Core 目标；在候选环境验证新旧音色共存。

单元测试使用合成张量/短 WAV 检查格式和失败路径；上游训练、ONNX/Core 合成属于需指定依赖/模型的集成测试，不能因本地缺 GPU 而伪报通过。没有真实目标录音时，交付状态只能达到工具/契约级，不能声称训练完成。

## 9. 设计依据

- 本地 `../source_backup/local_dependecy/voicevox_core-0.17.0/crates/voicevox_core/src/core/infer/domains/talk.rs` 与 `streaming_talk.rs`。
- 同版本 `model/sample.vvm/manifest.json`、`model/README.md`、`src/engine/talk/interpret_query.rs` 和 VVM 文档。
- 作者固定版本与相关源码见 [BASE_MODELS.md](BASE_MODELS.md)。
- [架构分析、训练路线与固定源码链接](docs/VOICEVOX_ROLE_TRAINING_ANALYSIS.md)。
