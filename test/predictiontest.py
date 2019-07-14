from optimizer.environment.spark.sparkapplicationtimedelaypredictor import SparkApplicationTimeDelayPredictor
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.spark.finishedsparkapplicationanalyzer import FinishedSparkApplicationAnalyzer
from optimizer.environment.spark.finishedsparkapplicationpredictor import FinishedSparkApplicationPredictor
from optimizer.util import timeutil

if __name__ == '__main__':
    builder = SparkApplicationBuilder('http://omnisky:18080/api/v1/')
    analyzer = FinishedSparkApplicationAnalyzer()
    predictor = FinishedSparkApplicationPredictor()

    app = builder.build('application_1562834622700_0014')
    svm_model = analyzer.analyze_and_save(app)
    predictor.add_algorithm('SVM', svm_model)
    time = predictor.predict('SVM', 5041781335, app)
    print('09:42:26', 'SVM\t', timeutil.convert_timestamp_to_str(time), '12:39:31')

    app = builder.build('application_1562834622700_0018')
    svm_model = analyzer.analyze_and_save(app)
    predictor.add_algorithm('KMeans', svm_model)
    time = predictor.predict('KMeans', 3007440320, app)
    print('10:28:44', 'KMeans\t', timeutil.convert_timestamp_to_str(time), '10:51:42')

    app = builder.build('application_1562834622700_0039')
    als_model = analyzer.analyze_and_save(app)
    predictor.add_algorithm('ALS', als_model)
    time = predictor.predict('ALS', 65536, app)
    print('15:57:21', 'ALS\t', timeutil.convert_timestamp_to_str(time), '15:59:09')

    app = builder.build('application_1562834622700_0016')
    time = predictor.predict('KMeans', 126463, app)
    print('10:23:14', 'KMeans\t', timeutil.convert_timestamp_to_str(time), '10:23:30')

    app = builder.build('application_1562834622700_0049')
    time = predictor.predict('ALS', 65536, app)
    print('21:10:25', 'ALS\t', timeutil.convert_timestamp_to_str(time), '21:10:52')

    # predictor = SparkApplicationTimeDelayPredictor('http://omnisky:18080/api/v1/')
    # time_delay, finish_time = predictor.predict('application_1562834622700_0018')
    # print(time_delay, timeutil.convert_timestamp_to_str(finish_time))
