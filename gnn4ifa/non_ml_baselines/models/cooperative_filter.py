from simpful import FuzzySet, FuzzySystem, LinguisticVariable
from .base import BaseDetector
from non_ml_baselines.data import Sample


class CooperativeFilter(BaseDetector):
    def __init__(self):
        super().__init__()
        self.fuzzy_system = FuzzySystem()

        self.por_sets = [FuzzySet(points=[[0, 1.], [0.6, 1.], [1., 0]], term="normal"),
                         FuzzySet(points=[[0, 0.], [0.6, 0.], [1., 1.]], term="high")]
        self.fuzzy_system.add_linguistic_variable("Por", LinguisticVariable(self.por_sets,
                                                                            concept="Por",
                                                                            universe_of_discourse=[0, 1]))

        self.per_sets = [FuzzySet(points=[[0, 1.], [0.2, 1.], [0.6, 0.], [1., 0]], term="normal"),
                         FuzzySet(points=[[0, 0.], [0.2, 0.], [0.6, 1.], [1., 1.]], term="high")]
        self.fuzzy_system.add_linguistic_variable("Per", LinguisticVariable(self.per_sets,
                                                                            concept="Per",
                                                                            universe_of_discourse=[0, 1]))

        self.outputs_sets = [FuzzySet(points=[[0, 1.], [0.5, 1.], [0.9, 0.], [1., 0]], term="normal"),
                             FuzzySet(points=[[0, 0.], [0.4, 0.], [0.8, 1.], [1., 1.]], term="ifa")]
        self.fuzzy_system.add_linguistic_variable("Output", LinguisticVariable(self.outputs_sets,
                                                                               universe_of_discourse=[0, 1]))
        self.rules = ["IF (Por IS normal) AND (Per IS normal) THEN (Output IS normal)",
                      "IF (Por IS high) AND (Per IS normal) THEN (Output IS ifa)",
                      "IF (Por IS normal) AND (Per IS high) THEN (Output IS ifa)",
                      "IF (Por IS high) AND (Per IS high) THEN (Output IS ifa)"]
        self.fuzzy_system.add_rules(self.rules)
        self.output_threshold = 0.6

    @staticmethod
    def global_prediction(routers_predictions):
        for router, router_dict in routers_predictions.items():
            if routers_predictions[router] == 1:
                return 1
        return 0

    @staticmethod
    def aggregate_interfaces_features(features_per_interface):
        features = {'por': 0.,
                    'per': 0.}
        n_interfaces = 0
        for interface, feat_dict in features_per_interface.items():
            features['por'] += feat_dict['por']
            features['per'] += feat_dict['per']
            n_interfaces += 1
        # features['por'] /= float(n_interfaces)
        # features['per'] /= float(n_interfaces)
        return features

    def predict(self, sample):
        features_per_router = sample.get_routers_feat()
        predictions = {}
        for router, router_dict in features_per_router.items():
            predictions[router] = None
            features = CooperativeFilter.aggregate_interfaces_features(features_per_router[router])
            self.fuzzy_system.set_variable("Por", features['por'])
            self.fuzzy_system.set_variable("Per", features['per'])
            fs_output = self.fuzzy_system.inference()['Output']
            predictions[router] = 1 if fs_output > 0.6 else 0
        return predictions
