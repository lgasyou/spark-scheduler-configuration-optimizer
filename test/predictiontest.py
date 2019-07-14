from optimizer.environment.spark.sparkapplicationtimedelaypredictor import SparkApplicationTimeDelayPredictor
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.spark.finishedsparkapplicationanalyzer import FinishedSparkApplicationAnalyzer
from optimizer.environment.spark.finishedsparkapplicationpredictor import FinishedSparkApplicationPredictor
from optimizer.util import timeutil

if __name__ == '__main__':
    builder = SparkApplicationBuilder('http://omnisky:18080/api/v1/')
    analyzer = FinishedSparkApplicationAnalyzer()
    predictor = FinishedSparkApplicationPredictor()

    app = builder.build('application_1562834622700_0018')
    kmeans_model = analyzer.analyze_and_save(app)
    predictor.add_algorithm('KMeans', kmeans_model)

    app = builder.build('application_1562834622700_0039')
    als_model = analyzer.analyze_and_save(app)
    predictor.add_algorithm('ALS', als_model)

    app = builder.build('application_1562834622700_0016')
    time = predictor.predict('KMeans', 126463, app)
    print(timeutil.convert_timestamp_to_str(time))

    app = builder.build('application_1562834622700_0049')
    time = predictor.predict('ALS', 65536, app)
    print(timeutil.convert_timestamp_to_str(time))

    # predictor = SparkApplicationTimeDelayPredictor('http://omnisky:18080/api/v1/')
    # time_delay, finish_time = predictor.predict('application_1562834622700_0018')
    # print(time_delay, timeutil.convert_timestamp_to_str(finish_time))
