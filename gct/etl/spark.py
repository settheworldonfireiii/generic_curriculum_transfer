from __future__ import annotations

from pathlib import Path

from gct.config.schema import ExperimentConfig


def prepare_dataset_spark(config: ExperimentConfig) -> Path:
    """Run large-scale preprocessing through Spark.

    This path expects data already exported to files Spark can read. The HF path
    is still the canonical source selector; for very large datasets use
    `gct prepare-data --engine hf --streaming --saved-format jsonl` first, then
    point Spark at those files for joins/aggregations/sharding.
    """

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, length
    except ImportError as exc:
        raise RuntimeError("Install the Spark extra first: `pip install -e .[spark]`.") from exc

    source = config.runtime.output_dir / "datasets" / "tasks.jsonl"
    if not source.exists():
        raise FileNotFoundError(
            f"Spark input {source} does not exist. Run the HF preparation step first."
        )

    spark = SparkSession.builder.appName("generic-curriculum-transfer-etl").getOrCreate()
    try:
        df = spark.read.json(str(source))
        df = df.filter((length(col("prompt")) > 0) & (length(col("answer")) > 0))
        output_dir = config.runtime.output_dir / "datasets" / "spark_prepared"
        df.repartition(max(1, config.runtime.num_workers)).write.mode("overwrite").json(str(output_dir))
    finally:
        spark.stop()
    return output_dir

