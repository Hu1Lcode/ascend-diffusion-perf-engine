# Ascend Diffusion Perf Engine

基于 [Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine) 的实验框架，使用 [vLLM-Omni v0.28.0](https://github.com/vllm-project/vllm-omni/tree/v0.28.0) 在 Ascend 平台上建立可复现的视频扩散推理基线，并逐项测量优化特性。

**[完整中文使用说明与设计边界](README_ASCEND.md)** · [Sol-Engine 原始 README](README_SOL.md)

目前支持 Wan2.2 TI2V 5B 的 T2V/I2V 三样本基线，固化输入、随机种子、启动脚本和运行环境；单独记录文本编码、DiT、VAE、媒体处理阶段能采到的耗时。逐项试验报告包含开启前后耗时、增量及相对基线加速比、PSNR/SSIM 烟测与人工复核状态。已接入的候选来自 vLLM-Omni 现有缓存、VAE 和 Ascend 注意力后端；`torch_npu` 融合、量化、稀疏注意力和 token pruning 尚待 NPU 实测后接入。

```bash
python3 scripts/run.py config/ascend_wan22/baseline_t2v.toml \
  --set VLLM_OMNI_ROOT=/path/to/vllm-omni \
  --set MODEL_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers \
  --set PYTHON_BIN=/path/to/ascend-env/bin/python
```

在当前无 Ascend NPU 的开发环境中，T2V/I2V dry-run 与 11 项合同测试已通过；**尚无真实 NPU 加速或画质结论**。运行前请阅读 [环境和评估说明](README_ASCEND.md)。代码沿用 Apache-2.0，Sol-Engine 原项目署名见 [出处说明](NOTICE_ASCEND.md)。
