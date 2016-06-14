__author__ = 'Rays'


class ModelBase:

    def __init__(self, settings):
        pass

    def train(self, x_train, y_train, x_cv, y_cv):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % (self.__class__.__name__))

    def predict(self, x):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % (self.__class__.__name__))