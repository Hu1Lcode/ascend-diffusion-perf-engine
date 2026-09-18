# Ascend Diffusion Perf Engine（实验性）

本项目基于 [NVlabs/Sana 的 `sol-engine` 分支](https://github.com/NVlabs/Sana/tree/sol-engine)（起点提交 `bb60499af0e675095ff67424196d8c18e265f32a`），复用其配置清单、`scripts/run.py` 启动器、独立 run bundle、`ModelTransform`/`compose` 冲突检查和优化实验记录方式。新增的 `ascend_engine/` 将运行时接到 **vLLM-Omni v0.28.0**，不会把原项目的 CUDA/Triton kernel 当作 Ascend kernel 使用。`vLLM-Omni` 源码及模型权重由使用者在本机另行准备，本仓库不分发它们。

当前交付的是可运行的**基线和逐项试验框架**，不是已经完成 NPU 实测的优化配方。本开发机没有 Ascend NPU、模型权重及 ffmpeg，因此已验证源码校验、配置展开、T2V/I2V 三样本 dry-run、单元测试及脚本语法；真实推理、速度与画质结果需要在 Ascend 机器上生成。任何 `report.md` 中的 `provisional_keep` 都只表示数值烟测通过且测量更快，还要人工并排看视频才可交付。

## 运行前准备

1. Ascend 驱动、CANN、兼容的 PyTorch/`torch_npu` 环境；在该环境中按 vLLM-Omni v0.28.0 的 NPU 安装说明安装依赖和源码。启动器会检查 `torch.npu.is_available()`、设备名称与版本。`attention_flash` 还会检查 `mindiesd` 是否可发现。
2. 本地 `vllm-omni` 源码 checkout 到 tag `v0.28.0`，提交必须是 `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c`。桥接器同时校验 T2V/I2V 示例文件 SHA-256，防止静默沿用修改后的启动脚本；实际运行时在 run bundle 中生成带计时和种子固定代码的副本，不改用户的源码。
3. 本地 **Wan2.2-TI2V-5B-Diffusers** 权重目录，供 T2V 和 I2V 共用。当前配置专为这一模型冻结，不要把其它模型直接代入并据此比较。
4. `ffmpeg` 和 `ffprobe`，用于视频帧数、帧率、PSNR 和 SSIM 快速质量门禁；缺少它们时质量状态为 `blocked`，候选不能进入最终配置。

从项目根目录运行（路径替换为你的本地绝对路径）：

```bash
python3 scripts/run.py config/ascend_wan22/baseline_t2v.toml \
  --set VLLM_OMNI_ROOT=/path/to/vllm-omni \
  --set MODEL_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers \
  --set PYTHON_BIN=/path/to/ascend-env/bin/python

python3 scripts/run.py config/ascend_wan22/baseline_i2v.toml \
  --set VLLM_OMNI_ROOT=/path/to/vllm-omni \
  --set MODEL_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers \
  --set PYTHON_BIN=/path/to/ascend-env/bin/python
```

先检查计划而不推理，可增加 `--set ASCEND_DRY_RUN=1`。基线必须先成功生成视频及 `outputs/benchmark.json`，再试优化。`runs/` 被 git 忽略；生成的 run bundle 含 `launch.sh`、`outputs/run_spec.json`、固化启动脚本 `outputs/entrypoint.py`、每条命令、日志和视频。`run_spec.json` 保存模型目录变更指纹（文件名、大小、修改时间；**不是权重内容哈希**）、环境版本、输入图哈希、样本配置、源码脚本哈希及启用的特性。

## 固化的工作负载与评测

`evals/ascend_samples/t2v.json` 和 `i2v.json` 各有三个固定 prompt、负向 prompt、seed；I2V 图片保存在 `evals/ascend_samples/images/`。这是随仓库固化的快速**回归样本**，不需用户另准备验证集。想扩充到五个样本时，先在 JSON 中增加两条，再传 `--sample-limit 5`；同一次比较必须使用完全相同的样本和设置。

默认 T2V：1280×720、81 帧、40 步、24 fps、guidance 4；I2V：832×480、81 帧、50 步、16 fps、guidance 5。每个样本使用同一 seed 先预热一次，再重置 generator 执行正式计时；同时固定 Python、NumPy、Torch seed 和 `PYTHONHASHSEED`。某些 NPU 算子或编码器仍可能非逐位确定，因此保留输入、参数、源码和输出用于比较。正式性能是三个样本的**预热生成 + I2V 输入图片预处理 + 输出视频后处理/编码**时间之和的中位数；不含模型加载。每个样本另记含加载的进程耗时。

首个样本另跑一次诊断 profiler，按可用的 `text_encode`、DiT/diffuse、VAE encode/decode 汇总阶段时间，并把 I2V 图片解码/预处理及输出媒体后处理/编码单列。profiler 加同步点，会改变性能，所以诊断值**不能直接相加当正式端到端时间**；未被 vLLM-Omni profiler 暴露的阶段显示缺失，不视作零。视频编解码中的输入路径目前只涵盖 I2V 静态图片；没有独立的视频输入解码阶段。

数值质量比较逐帧检查宽高、帧数、帧率，再通过 ffmpeg 计算 PSNR/SSIM。默认阈值 20 dB/0.8 只是快速烟测，不代表人眼画质充分；对 I2V 还应人工查看主体身份、首帧一致性，对 T2V 查看动作、时序闪烁和细节。候选相对**固定基线**比质量，测量加速则同时与前一个保留方案和原始基线相比。可直接比较两个 run：

```bash
python3 scripts/ascend_compare.py /path/to/baseline-run /path/to/candidate-run
```

## 一项一项地试验

```bash
python3 scripts/ascend_sweep.py \
  --task t2v \
  --vllm-root /path/to/vllm-omni \
  --model-path /path/to/Wan2.2-TI2V-5B-Diffusers \
  --python-bin /path/to/ascend-env/bin/python \
  --feature attention_sdpa \
  --feature cache_dit \
  --feature vae_tiling

# I2V 使用同一入口，把 --task 改为 i2v；亦可 --baseline-run /path/to/run
```

每步只**新增一项**特性，在上一个通过速度与数值门禁的组合上试验；失败/变慢则记录并回到上一个组合。`runs/<时间>-ascend-<任务>-sweep/report.md` 逐项列出开启前后秒数、增量加速比、相对开箱基线加速比、质量结果、选择原因和阶段拆分；`experiments.json`/`result.json` 提供机器可读记录。每次 run 的完整脚本、参数与日志另存。无速度实测时不输出虚构的加速比。

| 当前可执行特性 | 来源 | 对应 Sol 类别/位置 | 限制 |
|---|---|---|---|
| `cache_dit` | vLLM-Omni 现有 `--cache-backend cache_dit` | 跨步缓存 | 仅已配置 Wan2.2；可能影响画质 |
| `vae_tiling`、`vae_slicing` | vLLM-Omni 现有 VAE 开关 | VAE 执行/显存策略 | 互斥，通常先解决显存，不保证提速 |
| `attention_sdpa` | vLLM-Omni NPU SDPA 后端 | 注意力基准 | 与 Flash 互斥 |
| `attention_flash` | vLLM-Omni NPU + MindIE-SD 后端 | 稠密注意力实现 | 需要 `mindiesd`；不是 Sol 稀疏注意力 |

Sol-Engine 原来的五类是**缓存、量化、kernel 融合、稀疏注意力、token pruning**。当前只有缓存类有直接候选；VAE 执行策略和稠密注意力后端是前置的运行时能力，不应误标为其它四类已经移植。后续按以下顺序扩展：先核查 vLLM-Omni 自带的量化/compile/图捕获等功能是否适配当前 Wan 模型与 NPU；再为 `torch_npu` 的融合算子或图执行做独立 `ModelTransform`，用实测确认调用命中；最后根据 Ascend profiler 中的瓶颈实现新的 kernel 融合、量化、稀疏注意力或 token pruning。每项需沿用同一质量与实验账本。未经 NPU 验证，不会仅靠一个环境变量宣称接入了融合算子。

当前 `attention_flash` 只做安装可发现性校验，实际 backend 命中仍需检查 run 日志/设备 profiler；`cache_dit` 的 summary 日志也应审核。自动搜索胜者保持 `provisional_keep`。如果机器、CANN/`torch_npu` 版本、模型权重、工作负载或启动参数变了，应重建基线，不能沿用旧速度比。

## 验证与出处

本仓库包含 CPU 可运行的合同测试：

```bash
python3 -m unittest discover -s tests -p 'test_ascend_engine.py' -v
python3 -m compileall -q ascend_engine scripts/ascend_compare.py scripts/ascend_sweep.py
```

继承的 Sol-Engine 代码和文档保留其上游署名，见 [原项目](https://github.com/NVlabs/Sana/tree/sol-engine)；vLLM-Omni 示例脚本只在运行时从用户本地的 v0.28.0 checkout 复制到忽略的 run bundle 中，见 [其上游](https://github.com/vllm-project/vllm-omni/tree/v0.28.0)。随仓库附带的 I2V 图片是专为三个烟测场景生成的测试素材，不代表任何模型输出质量。代码许可沿用 Apache-2.0；模型权重遵循各自的许可。
