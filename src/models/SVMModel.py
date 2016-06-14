__author__ = 'Rays'

from sklearn import svm

from models import ModelBase


class SVMModel(ModelBase.ModelBase):

    def __init__(self, settings):
        self.model = svm.SVC(C=settings["C"],
                             max_iter=settings["max_iter"],
                             cache_size=settings["cache_size"],
                             class_weight=settings["class_weight"])
        pass

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)


