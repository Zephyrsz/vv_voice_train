# 环境与 API 适配

素材/编排工具在独立 Python 3.11 环境运行，不修改现有服务 Python 3.13。项目依赖可按 pyproject 分为轻量 data、features、onnx 和上游 training 依赖。

频谱提取直接读取锁定 HiFi-GAN 源码中的三个纯函数，做两个 API 适配：librosa Mel 的参数改为 keyword；torch.stft 使用 return_complex=True 后 view_as_real，维持作者 sqrt(real²+imag²+1e-9) 的幅度计算。它不把频谱改成相对 dB，不改变网络或损失。

该适配生成源码哈希和 api_patches 记录。帧频严格为 93.75，F0 的 frame_period=1000/93.75 ms；F0 提取器的末帧按时间坐标对应到声码器频谱，不用近似 10.67 ms。

研究起步 profile 为 FFT/win 1024、80 Mel、fmin 0/fmax 12000，这是本项目自行选择的初始配置，不是对正式 VVM 参数的推断。有基座时需要严格匹配它的 profile。

开发机 CPU 验证不能代替 CUDA/GPU 训练验证。pyopenjtalk 的 pip 版本可用于前端工具测试，但正式复现应使用 upstream.lock.json 中的 VOICEVOX fork；每次前端运行记录实际版本。训练与 Core 的字典仍需对照检查。
