# 训练素材与派生数据契约

此处定义已实现的中间格式和上游转换规则。特征提取已通过合成音调夹具验证；示例中的目标音频仍需要实际录制，标签示例不代表已对齐的真实录音。

## 1. 需要人工准备什么

建议原始录音为无损 WAV、48 kHz / 24 bit / mono，安静且无削波，固定设备、距离和风格。保存原始音频，不做逐句 loudnorm、相对最大值频谱归一化、变调或强降噪。训练副本统一为 24 kHz mono，具体 PCM/float 存储由处理配置记录。

一条 manifest 对应一句完整录音，通常数秒至十余秒；不机械切断辅音或吞掉尾音。文本须与实际发声一致，自动转写后逐句人工核对。原文、规范化文本和实际读音分开保留，数字与专名读法尤其需要确认。

不要求手工逐帧绘制 F0，但作者训练器需要音素边界、重音和 F0 等派生数据。自动前端、强制对齐、音高提取后仍须抽查，不能沿用“只有 SBV2 四列 list 即已完成”的假设。

## 2. 原始目录与 manifest

```text
corpora/new_voice_v1/
├── raw/newvoice_000001.wav
├── manifest.jsonl
├── speakers.json
├── splits/train.txt
├── splits/valid.txt
├── splits/test.txt
├── corrections/                   # 读音、重音与对齐修正记录
└── dataset_info.json              # 来源、设备、授权、语料版本
```

manifest 每行 UTF-8 JSON，路径相对于该语料根目录：

```json
{"utterance_id":"newvoice_000001","audio":"raw/newvoice_000001.wav","text":"今日はいい天気ですね。","normalized_text":"今日はいい天気ですね。","reading":"キョーワイーテンキデスネ","speaker_key":"new_voice","style_key":"neutral","session_id":"20261014_a","language":"ja"}
```

- `reading` 为人工读音记录；此示例使用实际长音读法，不直接当作 Open JTalk 标准标签。需要前端适配和核对。
- utterance_id 全语料唯一，文件名安全；audio 必须在数据根目录内且确实存在。
- 第一轮只接受 `ja`；不把 SBV2 的 `JP` 字段直接传到作者模型。
- speaker_key/style_key 是素材标识；不等于对外 style ID 或模型内部 speaker ID。
- 每个 ID 只能出现于一个集合。相同文本、音频哈希重复、同源切片要成组分配，防止泄漏。单说话人可以出现在 train/valid/test 中，这正是评估该声音未见文本的需要。
- 初始划分可按约 90/5/5 设计，但固定划分结果和随机种子；小语料必须显式保证 valid/test 非空。最终 test 不用于调参。

`speakers.json` 的设计示例：

```json
{"schema_version":1,"voices":[{"speaker_key":"new_voice","style_key":"neutral","inner_voice_id":0,"public_style_id":1000000,"name":"新しい声","style_name":"ノーマル"}]}
```

1000000 仅为示例候选 ID，不是已分配且保证无冲突的号段。打包前扫描当前服务 VVM 的 metas 验证唯一性；生成独立 UUID 并保存在项目配置中。

## 3. 对齐和重音标签

前端产物保存 full-context labels 与导出的标准音素序列；强制对齐后产生与其对应的真实边界。

标签文件采用作者 `OjtPhoneme.load_julius_list` 接收的秒制三列：

```text
0.0000 0.1000 pau
0.1000 0.1800 k
0.1800 0.3000 o
0.3000 0.4000 pau
```

以上是格式示意，不对应 manifest 的示例句子。首端从 0 开始、边界连续、结束大于开始、末端与音频有效长度一致。它不是直接采用 100 ns 整数时间的 HTK 标签；导入 HTK 时必须显式转换时间单位。

`sil` 等工具专用符号先按版本化映射转为目标表；不允许把未知音素静默替换成 `pau`。ROHAN 的音素与 OjtPhoneme 并非全部一一对应，任何近似映射需记录并人工审查。

四种 accent 文件分别为：

```text
start_accent/<id>.txt
end_accent/<id>.txt
start_accent_phrase/<id>.txt
end_accent_phrase/<id>.txt
```

每个文件为可空白分隔的 0/1 整数序列。按 `yukarin_sa` dataset 的期待，先与**完整对齐音素列表**对应，再由 loader 用 vowel_indexes 提取 mora 位置；不能只保存 mora 长度的向量又让 loader 再索引一次。音素匹配不能只验证数量，还要逐项验证符号及重复音素的位置。

重音修正需要同时更新 full-context/结构化前端数据与这些派生数组，不能只编辑一个不再参与训练的展示文件。

## 4. 特征 profile：一份配置控制全部模块

固定值：sample_rate=24000、hop_length=256、frame_rate=93.75、Core 输入有声 F0 为 ln(Hz)、无声 0。音素字典从锁定源码提取并逐项对比 Core，记录列表及哈希，不自己按语料重新建词表。

频谱参数由所选基座或 scratch 配置明确给出：n_fft、win_length、num_mels/spec_dim、fmin/fmax、center、padding、幅度定义、log 压缩和归一化。不能把“通常 80 维/2048 FFT”当作正式模型规格。

声码器与声学目标采用**同一频谱函数**，优先复用作者 HiFi-GAN fork 的 `meldataset.mel_spectrogram`：其实现为幅度谱 → Mel → 自然对数压缩，并有自己的反射填充规则。禁止用 `librosa.power_to_db(..., ref=np.max)` 替代。旧 PyTorch STFT/librosa API 的修复作为显式补丁，须用固定输入比较数值，不顺便更换频谱定义。

profile 必须记录边界帧、padding 和时长转帧规则。F0 提取时严格使用 `1000*256/24000` ms，避免四舍五入为 10.67 ms；若提取工具产生额外末帧，按约定的时间坐标统一到目标长度，并记录处理方法，不通过截掉任意多帧掩盖对齐错误。

## 5. 中间特征与作者 loader 适配

```text
derived/
├── audio24k/<id>.wav
├── frontend/<id>.json
├── full_context/<id>.lab
├── phoneme_list/<id>.lab
├── start_accent/<id>.txt
├── end_accent/<id>.txt
├── start_accent_phrase/<id>.txt
├── end_accent_phrase/<id>.txt
├── f0/<id>.npz
├── spec/<id>.npz
├── phoneme/<id>.npz
├── silence/<id>.npz
├── volume/<id>.npz
└── feature_profile.json
```

框架自己的中间文件使用 `npz` 的 `array + rate`，读取不需要 pickle；只在适配器产生供上游读取的本地受控 `.npy` dict。作者两个 SamplingData loader 需要 `numpy.load(..., allow_pickle=True).item()` 得到 `{"array": ..., "rate": ...}`，不是裸数组。因此旧 `role_train` 的 `f0_<id>.npy` 不能直接作为它们的输入。

| 特征 | 框架中间数组 | rate | 说明 |
| --- | --- | --- | --- |
| f0 | float32 `[T,1]` | 93.75 | 有声 ln(Hz)，无声 0；保留原始 Hz/提取配置用于诊断 |
| phoneme | float32 `[T,P]` | 93.75 | 对齐音素 one-hot；静音也有目标音素 |
| spec | float32 `[T,D]` | 93.75 | 与声码器条件一致的 log 频谱 |
| silence | bool `[T,1]` | 93.75 | 静音标记，与 pau/边界对应 |
| volume | float32 `[T,1]` | 93.75 | 明确窗口定义的非负幅度/RMS；用于上游加权，不是新增 energy predictor |

导出作者训练数据视图时：

- `yukarin_s`：原始秒制标签、speaker 字典；监督是 `end-start`。
- `yukarin_sa`：标签、四个 accent 数组、SamplingData F0/volume；loader 负责按 mora 等模式聚合，保存选定 f0_process_mode。
- `yukarin_sosoa`：SamplingData 的 F0/phoneme/spec/silence/volume，以及需要时的 phoneme_list；文件同 stem。
- `hifi-gan`：独立列表每行 `wav_stem|f0_stem`，按其 `get_dataset_filelist` 规则解析；不要放入 SBV2 的文本四列。该 fork 载入的 F0 应为 `[T]` 的 dict-array，适配器从 `[T,1]` 显式 squeeze；源码会将正 log-F0 指数化。
- 声码器 `fine_tuning` 载入的 Mel 为裸数组 `[D,T]`，不是 acoustic 的 `[T,D]` SamplingData dict；需要单独的转换数据视图。

各模块的 speaker_dict 文件格式遵循其实际 dataset；用同一稳定的 voice 注册顺序生成并检查，不能假定“发了一个 JSON 映射”就保证上游 ID 不重新排序。

## 6. 数据验收

必须拒绝：重复 ID、丢失音频、空/非有限数组、未知音素、音频与标签顺序不符、时间单位错误、负时长、特征 stem 集不一致、profile 不同、非法 speaker ID、集合泄漏。F0 八度跳变、过长静音、削波与可疑重音生成单独报告，修正后重算受影响的派生文件。

自动质量检查不声称能可靠测得真实背景噪声或验证听感。首批选择少量真实录音，逐条试听并画对齐/F0；再做完整子集。有效录音时长和覆盖缺口比总文件大小更有意义。

基于这些契约编写测试：秒与 100 ns 的转换、重复音素对齐、无声 F0、大小写无声母音、长音/促音/拨音、frame-rate 漂移、SamplingData round-trip、HiFiGAN shape 适配、固定 speaker ID、训练/测试重复检查。

## 7. 源码依据

- [作者 OjtPhoneme 秒制标签](https://github.com/Hiroshiba/yukarin_sosoa/blob/22aa731d56bb5f1c9fe4bfab12b661035cb4a0e2/yukarin_sosoa/data/phoneme.py)
- [yukarin_sa dataset：accent 索引与 mora F0](https://github.com/Hiroshiba/yukarin_sa/blob/2ad1065f94d61e53006e7311086294a8b1407b6f/yukarin_sa/dataset.py)
- [作者 SamplingData 文件格式](https://github.com/Hiroshiba/yukarin_sosoa/blob/22aa731d56bb5f1c9fe4bfab12b661035cb4a0e2/yukarin_sosoa/data/sampling_data.py)
- [作者 HiFiGAN Mel/F0 loader](https://github.com/Hiroshiba/hifi-gan/blob/265da0f0f71923336615d6657f195ffdc6df25e4/hifi_gan/meldataset.py)
- [ROHAN 原始录音建议](https://github.com/mmorise/rohan4600)
