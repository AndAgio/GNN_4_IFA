from .base import BaseDetector


class Chokifa(BaseDetector):
    def __init__(self):
        super().__init__()
        self.thresholds = {'delta': 3.,
                           'rho_min': 1. / 8. * 1200.,
                           'rho_max': 3. / 4. * 1200.}
        self.w_p = 0.001
        self.rho_avg_history = {}

    def compute_new_average_rho(self, old_rho_avg, current_rho):
        return (1 - self.w_p) * old_rho_avg + self.w_p * current_rho

    def update_rho_avg(self, sample):
        previous_rho_avgs = self.get_avg_rho_for_previous_sample(sample)
        new_rho_avgs = {}
        for router, router_dict in sample.get_routers_feat().items():
            new_rho_avgs[router] = {}
            for interface, interface_dict in router_dict.items():
                current_rho = sample.get_routers_feat()[router][interface]['pit_size']
                previous_rho = previous_rho_avgs[router][interface]
                new_rho_avgs[router][interface] = self.compute_new_average_rho(previous_rho, current_rho)
        self.insert_rho_avg_in_history(sample, new_rho_avgs)

    def compute_dropping_probability(self, sample, router, interface):
        rho_avg = self.get_avg_rho_for_this_sample(sample)[router][interface]
        return (rho_avg - self.thresholds['rho_max'])/(self.thresholds['rho_max'] - self.thresholds['rho_min'])

    def get_avg_rho_for_previous_sample(self, sample):
        setup_key = sample.get_sim_setup_key()
        try:
            avg_rho_dict = self.rho_avg_history[setup_key]
            return avg_rho_dict[sample.get_time() - 1]
        except KeyError:
            avg_rho = {}
            for router, router_dict in sample.get_routers_feat().items():
                avg_rho[router] = {}
                for interface, interface_dict in router_dict.items():
                    avg_rho[router][interface] = sample.get_routers_feat()[router][interface]['pit_size']
            return avg_rho

    def get_avg_rho_for_this_sample(self, sample):
        setup_key = sample.get_sim_setup_key()
        return self.rho_avg_history[setup_key][sample.get_time()]

    def insert_rho_avg_in_history(self, sample, rho_avg):
        setup_key = sample.get_sim_setup_key()
        sample_time = sample.get_time()
        try:
            self.rho_avg_history[setup_key]
        except KeyError:
            self.rho_avg_history[setup_key] = {}
        self.rho_avg_history[setup_key][sample_time] = rho_avg

    def predict(self, sample):
        self.update_rho_avg(sample)
        features_per_router = sample.get_routers_feat()
        predictions = {}
        for router, router_dict in features_per_router.items():
            predictions[router] = {}
            for interface, interface_dict in router_dict.items():
                current_rho = features_per_router[router][interface]['pit_size']
                current_delta = features_per_router[router][interface]['omega']
                if current_rho <= self.thresholds['rho_min']:
                    predictions[router][interface] = 0
                else:
                    if current_delta > self.thresholds['delta']:
                        predictions[router][interface] = 1
                    else:
                        if current_rho > self.thresholds['rho_max']:
                            predictions[router][interface] = 1
                        else:
                            probability = self.compute_dropping_probability(sample, router, interface)
                            if probability > 0.6:
                                predictions[router][interface] = 1
                            else:
                                predictions[router][interface] = 0
        return predictions
