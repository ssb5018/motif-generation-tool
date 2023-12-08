class ConstraintHyperparameters:
    def __init__(self, shape=2, weight=1):
        self.shape = shape
        self.weight = weight

    def set_shape_hyperparameter(self, hyperparameter):
        self.shape = hyperparameter

    def set_weight_hyperparameter(self, hyperparameter):
        assert(hyperparameter > 0)
        self.weight = hyperparameter


class Hyperparameters:
    def __init__(self, shape_hyperparameters={}, weight_hyperparameters={}):
        self.hom = ConstraintHyperparameters()
        self.hairpin = ConstraintHyperparameters()
        self.gc_content = ConstraintHyperparameters()
        self.similarity = ConstraintHyperparameters()

        for constraint in shape_hyperparameters:
            self.set_shape_hyperparameter(constraint, shape_hyperparameters[constraint])

        for constraint in weight_hyperparameters:
            self.set_weight_hyperparameter(constraint, weight_hyperparameters[constraint])

    def set_shape_hyperparameter(self, constraint, hyperparameter):
        if constraint == 'hom':
            self.hom.set_shape_hyperparameter(hyperparameter)
        elif constraint == 'hairpin':
            self.hairpin.set_shape_hyperparameter(hyperparameter)
        elif constraint == 'gcContent':
            self.gc_content.set_shape_hyperparameter(hyperparameter)
        elif constraint == 'similarity':
            self.similarity.set_shape_hyperparameter(hyperparameter)

    def set_weight_hyperparameter(self, constraint, hyperparameter):
        if constraint == 'hom':
            self.hom.set_weight_hyperparameter(hyperparameter)
        elif constraint == 'hairpin':
            self.hairpin.set_weight_hyperparameter(hyperparameter)
        elif constraint == 'gcContent':
            self.gc_content.set_weight_hyperparameter(hyperparameter)
        elif constraint == 'similarity':
            self.similarity.set_weight_hyperparameter(hyperparameter)
