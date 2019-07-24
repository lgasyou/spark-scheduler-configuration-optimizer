import logging
import os
import pickle
from typing import Dict

from optimizer.environment.timedelayprediction import predictionsparkmodel
from optimizer.environment.timedelayprediction.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.timedelayprediction.sparkmodelanalyzer import SparkModelAnalyzer


class AlgorithmModelBuilder(object):

    SAVE_FILENAME = './results/algorithm-models.pk'

    def __init__(self, application_builder: SparkApplicationBuilder):
        self.logger = logging.getLogger(__name__)
        self.app_builder = application_builder
        self.models = {}

    def get_model(self):
        filename = self.SAVE_FILENAME
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            self.build()
            self.save()
        else:
            self.load()
        return self.models

    def build(self) -> Dict[str, predictionsparkmodel.Application]:
        analyzer = SparkModelAnalyzer()

        # Linear
        app = self.app_builder.build_application('application_1563794174354_0025')
        self.models['linear'] = analyzer.analyze(app)

        # KMeans
        app = self.app_builder.build_application('application_1563794174354_0020')
        self.models['kmeans'] = analyzer.analyze(app)

        # SVM
        app = self.app_builder.build_application('application_1563794174354_0038')
        self.models['svm'] = analyzer.analyze(app)

        # Bayes
        app = self.app_builder.build_application('application_1563794174354_0023')
        self.models['bayes'] = analyzer.analyze(app)

        # FPGrowth
        app = self.app_builder.build_application('application_1563794174354_0029')
        self.models['FPGrowth'] = analyzer.analyze(app)

        # LDA
        app = self.app_builder.build_application('application_1563794174354_0036')
        self.models['lda'] = analyzer.analyze(app)

        return self.models

    def save(self):
        with open(self.SAVE_FILENAME, 'wb') as f:
            pickle.dump(self.models, f)
            self.logger.info('Algorithm models %s saved.' % self.SAVE_FILENAME)

    def load(self) -> dict:
        with open(self.SAVE_FILENAME, 'rb') as f:
            self.models = pickle.load(f)
            self.logger.info('Algorithm models %s loaded.' % self.SAVE_FILENAME)
            return self.models
