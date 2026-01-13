from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("nyc-local").master("local[*]").getOrCreate()


def run(spark, raw_path, curated_path, quarantine_path):
    raw = spark.read.parquet(raw_path)

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

    good.write.mode("overwrite").partitionBy("year", "month").parquet(curated_path)

    bad.write.mode("overwrite").parquet(quarantine_path)


run(
    spark,
    raw_path="raw",
    curated_path="data/curated",
    quarantine_path="data/quarantine",
)
