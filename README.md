# CPSP

## 1. 环境

已验证：Linux、Python 3.8.20、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、
torch-pruning 1.6.0。已有兼容环境可直接用；需要新环境时：

```bash
conda env create -f environment.yml
conda activate cpsp
python scripts/check.py --smoke
```

已有环境安装依赖可用 `python -m pip install -r requirements.txt`，但需自行确保
PyTorch/torchvision 的 CUDA 构建匹配。运行检查不需要数据或权重。

所有命令在仓库根目录执行。服务器长任务放在 tmux 中；以下示例使用卡 1：

```bash
tmux new -s cpsp
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

默认 `num_workers=0`，即单进程数据读取；正数 worker 未保证在参考环境稳定。
UIUNet 训练 batch 至少为 2，评测 batch 为 1。

## 2. 数据和权重

从 BasicIRSTD 提供的途径获取数据/原始模型权重，沿用其 **train/test** 划分。
本包不提供数据、权重或权重下载地址，不额外划分 validation。

```text
datasets/NUAA-SIRST/
  images/<id>.png
  masks/<id>.png
  img_idx/train_NUAA-SIRST.txt
  img_idx/test_NUAA-SIRST.txt
checkpoint/NUAA-SIRST/DNANet_400.pth.tar
```

其他数据集同样布局，支持 NUAA-SIRST、IRSTD-1K、NUDT-SIRST。
训练、CP 统计和剪枝规划只读取 train；评测读取 test，固定阈值 0.5。

## 3. CP 训练 → 剪枝微调 → 测试

以 DNANet / NUAA-SIRST 为例：

```bash
# CP 训练；不使用预训练权重时去掉 --pretrained。
python train.py --mode ocp_train --model DNANet --dataset NUAA-SIRST \
  --pretrained checkpoint/NUAA-SIRST/DNANet_400.pth.tar \
  --epochs 100 --experiment_name dnanet_cp

# 全局预算规划、DependencyGraph 物理剪枝及恢复训练。
python sweep_pruning.py --mode sweep_pruning --model DNANet --dataset NUAA-SIRST \
  --pretrained experiments/NUAA-SIRST/DNANet/dnanet_cp/best.pth \
  --pruning_method ours --global_prune_ratio 0.70 --finetune_epochs 40 \
  --experiment_name dnanet_cpsp

# 根据同一次剪枝的 plan/bundle 重建结构，再严格加载紧凑权重。
python evaluate_checkpoint.py --mode eval --model DNANet --dataset NUAA-SIRST \
  --pretrained experiments/NUAA-SIRST/DNANet/dnanet_cpsp/pruning_ours/global_0.70/finetune/best.pth \
  --pruning_plan_json experiments/NUAA-SIRST/DNANet/dnanet_cpsp/pruning_ours/global_0.70/pruning_plan.json \
  --pruning_bundle_json experiments/NUAA-SIRST/DNANet/dnanet_cpsp/pruning_ours/auto_prunable_layers.json \
  --experiment_name compact_eval
```

结果在 `experiments/<dataset>/<model>/<experiment_name>/`；评测结果为
`evaluation/evaluation.json`。稠密模型测试时不传两个剪枝 JSON 参数。
从头训练普通模型可用 `train.py --mode baseline_train`。

默认配置为 [configs/cpsp.yaml](configs/cpsp.yaml)，可用 `--config` 或 CLI 参数
覆盖。全局规划器是论文思想的工程实现，不声称逐项复现公式；示例用于跑通流程，
不保证仅凭默认配置就重现论文所有指标。替换模型/数据集时需要调整剪枝预算和
恢复训练设置。UIUNet 宜先使用 `--min_remaining_channels_per_layer 32`。

## 文件与许可

`core/` 负责训练/测试，`ocp/` 实现 CP，`pruning/` 实现规划与物理剪枝；
`model/` 为 BasicIRSTD 必需源码，`compat/` 为 ISNet 外部兼容，`analysis/` 为
运行时报告。没有内部日志、结果统计工程、发布矩阵、历史产物或缓存。

CPSP 自有新增代码采用 [MIT](LICENSE)。BasicIRSTD、模型依赖、数据和权重
保留其各自条款，MIT 不覆盖上游；见 [第三方说明](THIRD_PARTY_NOTICES.md)。
论文引用信息见 [CITATION.cff](CITATION.cff)。
