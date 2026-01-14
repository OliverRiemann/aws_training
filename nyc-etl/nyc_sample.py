from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("nyc-local").master("local[*]").getOrCreate()


def read_raw(spark, raw_path):
    raw_df = spark.read.parquet(raw_path).withColumn("source_file", F.input_file_name())
    return raw_df


def transform(raw_df):
    df = (
        raw_df.withColumn("pickup_ts", F.col("tpep_pickup_datetime").cast("timestamp"))
        .withColumn("dropoff_ts", F.col("tpep_dropoff_datetime").cast("timestamp"))
        .withColumn("passenger_count", F.col("passenger_count").cast("int"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("year", F.year("pickup_ts"))
        .withColumn("month", F.month("pickup_ts"))
    )
    return df


def data_quality():
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

    return required_ok & business_rules_ok


def write_df(df, curated_path, quarantine_path):
    is_good = data_quality()

    flagged = df.withColumn("is_good", is_good)

    good = (
        flagged.filter(F.col("is_good"))
        .filter(F.col("year").isNotNull() & F.col("month").isNotNull())
        .drop("is_good")
    )

    bad = (
        flagged.filter(~F.col("is_good"))
        .withColumn(
            "dq_reason",
            F.when(F.col("pickup_ts").isNull(), F.lit("missing_pickup_ts"))
            .when(F.col("dropoff_ts").isNull(), F.lit("missing_dropoff_ts"))
            .when(F.col("total_amount").isNull(), F.lit("missing_total_amount"))
            .when(F.col("total_amount") < 0, F.lit("negative_total_amount"))
            .otherwise(F.lit("failed_other_rules")),
        )
        .drop("is_good")
    )

    (
        good.repartition(8, "year", "month")
        .write.mode("append")
        .partitionBy("year", "month")
        .parquet(curated_path)
    )

    (
        bad.select(
            "source_file",
            "dq_reason",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "total_amount",
        )
        .write.mode("append")
        .parquet(quarantine_path)
    )

    metrics = flagged.agg(
        F.count("*").alias("rows_total"),
        F.sum(F.col("is_good").cast("int")).alias("rows_good"),
        F.sum((~F.col("is_good")).cast("int")).alias("rows_bad"),
    )

    (
        metrics.withColumn("run_ts", F.current_timestamp())
        .write.mode("append")
        .parquet("data/metrics")
    )


def get_processed_partitions(spark, curated_path):
    try:
        return spark.read.parquet(curated_path).select("year", "month").distinct()
    except Exception:
        # First run: curated path does not exist
        return spark.createDataFrame([], "year INT, month INT")


def filter_incremental(df, curated_path):
    processed = get_processed_partitions(df.sparkSession, curated_path)

    incoming = df.select("year", "month").distinct()

    new_partitions = incoming.join(processed, ["year", "month"], "left_anti")

    return df.join(new_partitions, ["year", "month"], "inner")


def run(
    spark,
    raw_path,
    curated_path,
    quarantine_path,
):
    raw = read_raw(spark, raw_path)
    df = transform(raw)

    # 🔹 INCREMENTAL FILTER
    incremental_df = filter_incremental(df, curated_path)

    if incremental_df.take(1) == []:
        print("No new data to process. Exiting.")
        return
    incremental_df.explain("formatted")

    write_df(
        df=incremental_df,
        curated_path=curated_path,
        quarantine_path=quarantine_path,
    )


run(
    spark,
    raw_path="raw",
    curated_path="data/curated",
    quarantine_path="data/quarantine",
)


spark.read.parquet("data/curated").groupBy("year", "month").count().orderBy(
    "year", "month"
).show()
