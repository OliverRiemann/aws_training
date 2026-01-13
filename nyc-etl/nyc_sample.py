from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

spark = SparkSession.builder.appName("nyc-local").master("local[*]").getOrCreate()


def run(spark, raw_path, curated_path, quarantine_path):
    # paste ETL logic here (no GlueContext / Job)
    schema = T.StructType(
        [
            T.StructField("tpep_pickup_datetime", T.StringType(), True),
            T.StructField("tpep_dropoff_datetime", T.StringType(), True),
            T.StructField("passenger_count", T.StringType(), True),
            T.StructField("trip_distance", T.StringType(), True),
            T.StructField("fare_amount", T.StringType(), True),
            T.StructField("total_amount", T.StringType(), True),
            # Add more columns here if your file has them
        ]
    )

    raw = spark.read.parquet(raw_path)

    df = (
        raw
        # Parse timestamps (string -> timestamp)
        .withColumn("pickup_ts", F.to_timestamp("tpep_pickup_datetime"))
        .withColumn("dropoff_ts", F.to_timestamp("tpep_dropoff_datetime"))
        # Cast numeric fields safely
        .withColumn("passenger_count", F.col("passenger_count").cast("int"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        # Partition columns from pickup timestamp
        .withColumn("year", F.year("pickup_ts"))
        .withColumn("month", F.month("pickup_ts"))
    )

    # -----------------------------
    # 4) Data quality rules
    # -----------------------------
    # Define what "good" means for your analytics tables.
    # Keep it strict enough for data quality, but not so strict that you drop everything.
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

    bad = df.filter(~is_good).withColumn(
        "dq_reason",
        F.when(F.col("pickup_ts").isNull(), F.lit("bad_or_missing_pickup_ts"))
        .when(F.col("dropoff_ts").isNull(), F.lit("bad_or_missing_dropoff_ts"))
        .when(F.col("total_amount").isNull(), F.lit("missing_total_amount"))
        .when(F.col("total_amount") < 0, F.lit("negative_total_amount"))
        .otherwise(F.lit("failed_other_rules")),
    )

    # -----------------------------
    # 5) Write curated dataset (Parquet + partitions)
    # ----------------------------
    # Use overwrite for a single-run prototype.
    # In production, you'd usually use append + partition overwrite strategies.
    (
        good.drop("tpep_pickup_datetime", "tpep_dropoff_datetime")
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .parquet(curated_path)
    )

    # -----------------------------
    # 6) Write quarantine dataset
    # -----------------------------
    # Quarantine is useful to debug and to keep pipeline resilient.
    (
        bad.select(
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "dq_reason",
        )
        .write.mode("overwrite")
        .parquet(quarantine_path)
    )


run(
    spark,
    raw_path="raw",
    curated_path="data/curated",
    quarantine_path="data/quarantine",
)
