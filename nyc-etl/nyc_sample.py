from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("nyc-local").master("local[*]").getOrCreate()


def run(spark, raw_path, curated_path, quarantine_path):
    raw = spark.read.parquet(raw_path)
    raw = raw.withColumn("source_file", F.input_file_name())
    raw.printSchema()

    df = (
        raw.withColumn("pickup_ts", F.col("tpep_pickup_datetime").cast("timestamp"))
        .withColumn("dropoff_ts", F.col("tpep_dropoff_datetime").cast("timestamp"))
        .withColumn("passenger_count", F.col("passenger_count").cast("int"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("year", F.year("pickup_ts"))
        .withColumn("month", F.month("pickup_ts"))
    )

    df.select(
        F.min("pickup_ts").alias("min_ts"), F.max("pickup_ts").alias("max_ts")
    ).show(truncate=False)

    df.groupBy("year", "month").count().orderBy("year", "month").show(50)

    required_ok = (
        F.col("pickup_ts").isNotNull()
        & F.col("dropoff_ts").isNotNull()
        & F.col("total_amount").isNotNull()
    )

    business_rules_ok = (
        (F.col("total_amount") >= 0)
        & (F.col("trip_distance").isNull() | (F.col("trip_distance") >= 0))
        & (F.col("passenger_count").isNull() | (F.col("passenger_count") >= 0))
    )

    is_good = required_ok & business_rules_ok

    good = df.filter(is_good)
    bad = df.filter(~is_good)

    good.repartition(8, "year", "month").write.mode("overwrite").partitionBy(
        "year", "month"
    ).parquet(curated_path)

    bad = df.filter(~is_good).withColumn(
        "dq_reason",
        F.when(F.col("pickup_ts").isNull(), F.lit("missing_pickup_ts"))
        .when(F.col("dropoff_ts").isNull(), F.lit("missing_dropoff_ts"))
        .when(F.col("total_amount").isNull(), F.lit("missing_total_amount"))
        .when(F.col("total_amount") < 0, F.lit("negative_total_amount"))
        .otherwise(F.lit("failed_other_rules")),
    )

    (
        bad.select(
            "source_file",
            "dq_reason",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "total_amount",
        )
        .write.mode("overwrite")
        .parquet(quarantine_path)
    )

    metrics = df.agg(
        F.count("*").alias("rows_total"),
        F.sum(F.when(is_good, 1).otherwise(0)).alias("rows_good"),
        F.sum(F.when(~is_good, 1).otherwise(0)).alias("rows_bad"),
    )

    (
        metrics.withColumn("run_ts", F.current_timestamp())
        .write.mode("append")
        .parquet("data/metrics")
    )


run(
    spark,
    raw_path="raw",
    curated_path="data/curated",
    quarantine_path="data/quarantine",
)
