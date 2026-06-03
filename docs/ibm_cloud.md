# IBM Cloud Deployment Notes

IBM Cloud should run the same CLI as SLURM. The recommended flow is:

1. Create a VPC GPU virtual server instance.
2. Attach a floating IP.
3. SSH in.
4. Copy this project and any checkpoint archives.
5. Run `scripts/install_cloud_env.sh`.
6. Run `gct` commands or `scripts/run_cloud_ablation.sh`.

Example on the IBM instance:

```bash
cd ~/generic_curriculum_transfer
chmod +x scripts/install_cloud_env.sh scripts/run_cloud_ablation.sh
./scripts/install_cloud_env.sh
conda activate arc-sae-sweep

gct --config configs/competitive_math.yaml prepare-data
gct --config configs/competitive_math.yaml resource-report
gct --config configs/competitive_math.yaml plan-curriculum --token-capacity 32000
gct --config configs/competitive_math.yaml shard-plan --num-shards 4
./scripts/run_cloud_ablation.sh
```

For V100, use fp16:

```bash
gct --config configs/competitive_math.yaml --dtype fp16 resource-report
```

For a different model:

```bash
gct --config configs/competitive_math.yaml \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --dtype fp16 \
  resource-report
```

If `resource-report` says tensor or mixed tensor/pipeline parallelism is needed,
launch through a backend that supports it, such as Accelerate, DeepSpeed, vLLM,
or a provider-specific launcher. The scaffold keeps the estimate and config
explicit so that backend can be added without changing dataset or scheduling
logic.

