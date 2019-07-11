from optimizer.environment.spark.sparkapplicationtimedelaypredictor import SparkApplicationTimeDelayPredictor
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.util import timeutil

if __name__ == '__main__':
    builder = SparkApplicationBuilder('http://omnisky:18080/api/v1/')
    app = builder.build('application_1562679470261_0014')
    # predictor = ApplicationTimeDelayPredictor()
    # time_delay, finish_time = predictor.predict('application_1562679470261_0014')
    # print(time_delay, timeutil.convert_timestamp_to_str(finish_time))
