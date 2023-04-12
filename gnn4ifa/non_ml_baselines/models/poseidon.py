from .base import BaseDetector


class Poseidon(BaseDetector):
    def __init__(self):
        super().__init__()
        self.starting_thresholds = {'omega': 3.,
                                    'rho': 1. / 8. * 1200.}
        self.thresholds_history = {}
        self.scale_factor = 2.

    def get_prediction_of_previous_sample(self, sample):
        setup_key = sample.get_sim_setup_key()
        try:
            predictions_dict = self.predictions_history[setup_key]
            return predictions_dict[sample.get_time() - 1]
        except KeyError:
            predictions = {}
            for router, router_dict in sample.get_routers_feat().items():
                predictions[router] = {}
                for interface, interface_dict in router_dict.items():
                    predictions[router][interface] = 0.
            return predictions

    def get_thresholds_for_previous_sample(self, sample):
        setup_key = sample.get_sim_setup_key()
        try:
            thresholds_dict = self.thresholds_history[setup_key]
            return thresholds_dict[sample.get_time() - 1]
        except KeyError:
            thresholds = {}
            for router, router_dict in sample.get_routers_feat().items():
                thresholds[router] = {}
                for interface, interface_dict in router_dict.items():
                    thresholds[router][interface] = self.starting_thresholds
            return thresholds

    def get_thresholds(self, sample):
        previous_predictions = self.get_prediction_of_previous_sample(sample)
        previous_thresholds = self.get_thresholds_for_previous_sample(sample)
        thresholds = {}
        for router, router_dict in previous_thresholds.items():
            thresholds[router] = {}
            for interface, interface_dict in router_dict.items():
                if previous_predictions[router][interface] == 1:
                    thresholds[router][interface] = self.scale_thresholds(previous_thresholds[router][interface])
                else:
                    thresholds[router][interface] = previous_thresholds[router][interface]
        return thresholds

    def scale_thresholds(self, previous_thresholds):
        return {key: value / self.scale_factor for key, value in previous_thresholds.items()}

    def insert_thresholds_in_history(self, sample, thresholds):
        setup_key = sample.get_sim_setup_key()
        sample_time = sample.get_time()
        try:
            self.thresholds_history[setup_key]
        except KeyError:
            self.thresholds_history[setup_key] = {}
        self.thresholds_history[setup_key][sample_time] = thresholds

    def predict(self, sample):
        thresholds = self.get_thresholds(sample)
        self.insert_thresholds_in_history(sample, thresholds)
        features_per_router = sample.get_routers_feat()
        predictions = {}
        for router, router_dict in features_per_router.items():
            predictions[router] = {}
            for interface, interface_dict in router_dict.items():
                current_omega = features_per_router[router][interface]['omega']
                omega_threshold = thresholds[router][interface]['omega']
                current_rho = features_per_router[router][interface]['pit_size']
                rho_threshold = thresholds[router][interface]['rho']
                if current_omega <= omega_threshold and current_rho <= rho_threshold:
                    predictions[router][interface] = 0
                else:
                    predictions[router][interface] = 1
        return predictions
