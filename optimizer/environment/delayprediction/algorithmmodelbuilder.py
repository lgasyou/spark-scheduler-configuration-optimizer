import logging
import os
import pickle
from typing import Dict

from optimizer.environment.delayprediction import predictionsparkmodel
from optimizer.environment.delayprediction.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.delayprediction.sparkmodelanalyzer import SparkModelAnalyzer


class AlgorithmModelBuilder(object):

    SAVE_FILENAME = './results/algorithm-models.pk'

    def __init__(self, application_builder: SparkApplicationBuilder):
        self.logger = logging.getLogger(__name__)
        self.app_builder = application_builder
        self.models = {}

    def load_or_build_model(self):
        filename = self.SAVE_FILENAME
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            self.models = self.build()
            self.save(self.models)
        else:
            self.models = self.load()
        return self.models

    def build(self) -> Dict[str, predictionsparkmodel.Application]:
        analyzer = SparkModelAnalyzer()
        models = {}

        # Linear
        app = self.app_builder.build_application('application_1563794174354_0025')
        models['linear'] = analyzer.analyze(app)

        # KMeans
        app = self.app_builder.build_application('application_1563794174354_0020')
        models['kmeans'] = analyzer.analyze(app)

        # SVM
        app = self.app_builder.build_application('application_1563794174354_0038')
        models['svm'] = analyzer.analyze(app)

        # Bayes
        app = self.app_builder.build_application('application_1563794174354_0023')
        models['bayes'] = analyzer.analyze(app)

        # FPGrowth
        app = self.app_builder.build_application('application_1563794174354_0029')
        models['FPGrowth'] = analyzer.analyze(app)

        # LDA
        app = self.app_builder.build_application('application_1563794174354_0036')
        models['lda'] = analyzer.analyze(app)

        return models

    def save(self, models: dict):
        with open(self.SAVE_FILENAME, 'wb') as f:
            pickle.dump(models, f)
            self.logger.info('Algorithm models %s saved.' % self.SAVE_FILENAME)

    def load(self) -> dict:
        with open(self.SAVE_FILENAME, 'rb') as f:
            models = pickle.load(f)
            self.logger.info('Algorithm models %s loaded.' % self.SAVE_FILENAME)
            return models
