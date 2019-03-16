import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{SparkSession, Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

object Main extends App {

  val startTime = System.currentTimeMillis()

  // Remove system's INFO messages from console output
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val spark = SparkSession
    .builder
    .master("local")
    .appName("BD_SPW")
    .getOrCreate

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Print numbers formatted with commas
  val formatter = java.text.NumberFormat.getIntegerInstance

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // LOAD DATA

  val inputFilePath = "src/main/resources/2008.csv"

  print("[info] Reading file...")
  var data = spark
    .read
    .format("csv")
    .option("header", "true")
    .load(inputFilePath)
  println("\r[info] Reading file: complete")

  print("[info] Evaluating the dataset's shape...")
  var nRows = data.count
  var nCols = data.columns.length
  println(s"\r[info] Evaluating the dataset's shape: ${formatter.format(nRows)} rows and $nCols columns")

  // Columns (29)

  // Year         ArrTime             CRSElapsedTime    Distance            CarrierDelay
  // Month        CRSArrTime          AirTime           TaxiIn              WeatherDelay
  // DayofMonth   UniqueCarrier       ArrDelay          TaxiOut             NASDelay
  // DayofWeek    FlightNum           DepDelay          Cancelled           SecurityDelay
  // DepTime      TailNum             Origin            CancellationCode    LateAircraftDelay
  // CRSDepTime   ActualElapsedTime   Dest              Diverted

  // Subset data and adjust data types
  data = data.select(
    data.col("Month").cast("int"),
    data.col("DayofMonth").cast("int"),
    data.col("DayofWeek").cast("int"),
    data.col("DepTime").cast("int"),
    data.col("CRSDepTime").cast("int"),
    data.col("CRSArrTime").cast("int"),
    data.col("UniqueCarrier"),
    data.col("CRSElapsedTime").cast("int"),
    data.col("ArrDelay").cast("int"),
    data.col("DepDelay").cast("int"),
    data.col("Origin"),
    data.col("Dest"),
    data.col("Distance").cast("int")
  )

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // CLEAN DATA

  // Remove rows containing NA's
  print("[info] Removing NA's...")
  data = data.na.drop
  var aux = nRows
  nRows = data.count
  println(s"\r[info] Removing NA's: ${formatter.format(aux - nRows)} rows removed")

  // Verify data is not out of field range
  print("[info] Verifying data is not out of field range...")
  data = data
    .filter(data.col("Month").isin(1 to 12 : _*))
    .filter(data.col("DayofMonth").isin(1 to 31 : _*))
    .filter(data.col("DayofWeek").isin(1 to 7 : _*))
    .filter(data.col("DepTime") >= 1 && data.col("DepTime") <= 2400)
  aux = nRows
  nRows = data.count
  println(s"\r[info] Verifying data is not out of field range: ${formatter.format(aux - nRows)} rows removed")

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // TRANSFORM DATA

  print("[info] Transforming data...")

  // Index categorical variables
  def indexer (inputCol: String, outputCol: String, df: DataFrame): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
    indexer.fit(df).transform(df)
  }
  data = indexer("UniqueCarrier", "UniqueCarrierIndex", data)
  data = indexer("Origin", "OriginIndex", data)
  data = indexer("Dest", "DestIndex", data)

  // Change time format from hhmm to minutes
  def hhmmToMinutes (hhmm: Column): Column = {
    val hh = (hhmm / 100) - ((hhmm / 100) % 1)
    val min = hh * 60 + (hhmm % 100)
    min
  }
  data = data
    .withColumn("DepTime", hhmmToMinutes(col("DepTime")))
    .withColumn("CRSDepTime", hhmmToMinutes(col("CRSDepTime")))
    .withColumn("CRSArrTime", hhmmToMinutes(col("CRSArrTime")))

  println("\r[info] Transforming data: complete")

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // MODEL CREATION

  data = data.select(
    data.col("Month"),
    data.col("DayofWeek"),
    data.col("CRSDepTime"),
    data.col("CRSArrTime"),
    data.col("UniqueCarrierIndex"),
    data.col("CRSElapsedTime"),
    data.col("DepDelay"),
    data.col("OriginIndex"),
    data.col("DestIndex"),
    data.col("Distance"),
    data.col("ArrDelay")
  )

  // Split data into training a test sets
  val split = data.randomSplit(Array(0.7, 0.3))
  val training = split(0)
  val test = split(1)

  val assembler = new VectorAssembler()
    .setInputCols(Array(
      "Month",
      "DayofWeek",
      "CRSDepTime",
      "CRSArrTime",
      "UniqueCarrierIndex",
      "CRSElapsedTime",
      "DepDelay",
      "OriginIndex",
      "DestIndex",
      "Distance"
    ))
    .setOutputCol("features")

  val lr = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("ArrDelay")
    .setMaxIter(10)
    .setElasticNetParam(0.8)

  val pipeline = new Pipeline()
    .setStages(Array(assembler, lr))

  print("[info] Training model...")
  val model = pipeline.fit(training)
  println("\r[info] Training model: complete")

  print("[info] Evaluating test...")
  val df = model.transform(test)

  var df2 = df.withColumn("error", abs(col("ArrDelay") - col("prediction")))

  val evaluator = new RegressionEvaluator()
    .setLabelCol("ArrDelay")
    .setPredictionCol("prediction")

  // Error
  val avgError = df2.select(avg(col("error"))).head.getDouble(0)
  val maxError = df2.select(max(col("error"))).head.getDouble(0)
  val minError = df2.select(min(col("error"))).head.getDouble(0)

  // Root Mean Square Error
  val rmse = evaluator.setMetricName("rmse").evaluate(df2)

  // Coefficient of determination
  val r2 = evaluator.setMetricName("r2").evaluate(df2)

  println("\r[info] Evaluating test: complete")
  println(f"[result] Avg. error: $avgError%.2f minutes")
  println(f"[result] Max. error: $maxError%.2f minutes")
  println(f"[result] Min. error: $minError%.2f minutes")
  println(f"[result] RMSE = $rmse%.2f")
  println(f"[result] r2 = $r2%.2f")

  println(s"Total execution time: ${(System.currentTimeMillis() - startTime) / 1000 / 60} minutes ${(System.currentTimeMillis() - startTime) / 1000 % 60} seconds")

}
