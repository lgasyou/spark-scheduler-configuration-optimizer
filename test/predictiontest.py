from optimizer.environment.delayprediction.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.delayprediction.sparkmodelanalyzer import SparkModelAnalyzer
from optimizer.environment.delayprediction.timedelaypredictor import SingleDelayPredictor
from optimizer.util import timeutil
from test.completedapplicationbuilder import CompletedApplicationBuilder


def print_helper(application_id, t, *args):
    global predictor, app_builder
    time = predictor.predict_running_app(application_id, t) + 5000
    c = app_builder.build_application(application_id)
    predicted = time - c.start_time
    print(c,
          timeutil.convert_timestamp_to_str(time, '%H:%M:%S'),
          '{:6.0f}s{:6.0f}s{:6.1f}%'.format(c.elapsed_time / 1000,
                                            predicted / 1000,
                                            (predicted - c.elapsed_time) / c.elapsed_time * 100),
          ' ',
          *args
          )


if __name__ == '__main__':
    app_builder = CompletedApplicationBuilder()
    app_builder.build('application_1562834622700_0051')

    builder = SparkApplicationBuilder('http://omnisky:18080/api/v1/')
    analyzer = SparkModelAnalyzer()
    predictor = SingleDelayPredictor('http://omnisky:18080/api/v1/')

    print('{:<8} {:<5} {:<5} {:<8} {:<4} {:<4} {:<4} {}'.format(
        '负载', '开始时间', '完成时间', '预测完成时间', '时长', '预测时长', '误差率', '原型?'
    ))

    # Linear
    app = builder.build_application('application_1562834622700_0051')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('linear', svm_model)
    print_helper('application_1562834622700_0051', 'linear', '原型')
    print_helper('application_1562834622700_0038', 'linear')
    print_helper('application_1562834622700_0037', 'linear')

    # ALS
    app = builder.build_application('application_1562834622700_0039')
    als_model = analyzer.analyze(app)
    predictor.add_algorithm('ALS', als_model)
    print_helper('application_1562834622700_0039', 'ALS', '原型')
    print_helper('application_1562834622700_0049', 'ALS')

    # KMeans
    app = builder.build_application('application_1562834622700_0018')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('KMeans', svm_model)
    print_helper('application_1562834622700_0018', 'KMeans', '原型')
    print_helper('application_1562834622700_0016', 'KMeans')

    # SVM
    app = builder.build_application('application_1562834622700_0014')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('SVM', svm_model)
    print_helper('application_1562834622700_0014', 'SVM', '原型')
    print_helper('application_1562834622700_0050', 'SVM')

    # Bayes
    app = builder.build_application('application_1562834622700_0043')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('Bayes', svm_model)
    print_helper('application_1562834622700_0043', 'Bayes', '原型')
    print_helper('application_1562834622700_0044', 'Bayes')

    # FPGrowth
    app = builder.build_application('application_1562834622700_0054')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('FPGrowth', svm_model)
    print_helper('application_1562834622700_0054', 'FPGrowth', '原型')
    print_helper('application_1562834622700_0053', 'FPGrowth')
    print_helper('application_1562834622700_0055', 'FPGrowth')

    # LDA
    # TODO
    app = builder.build_application('application_1562834622700_0058')
    svm_model = analyzer.analyze(app)
    predictor.add_algorithm('LDA', svm_model)
    print_helper('application_1562834622700_0058', 'LDA', '原型')
    print_helper('application_1562834622700_0057', 'LDA')
    print_helper('application_1562834622700_0056', 'LDA')
