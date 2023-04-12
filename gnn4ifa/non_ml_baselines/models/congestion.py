from .base import BaseDetector


class CongestionAware(BaseDetector):
    def __init__(self):
        super().__init__()
        self.averages_history = {}

    def get_average_for_previous_sample(self, sample):
        setup_key = sample.get_sim_setup_key()
        try:
            averages_dict = self.averages_history[setup_key]
            return averages_dict[sample.get_time() - 1]
        except KeyError:
            averages = {}
            for router, router_dict in sample.get_routers_feat().items():
                averages[router] = {}
                for interface, interface_dict in router_dict.items():
                    averages[router][interface] = sample.get_routers_feat()[router][interface]['in_interests']
            return averages

    def get_averages(self, sample):
        previous_averages = self.get_average_for_previous_sample(sample)
        # print('previous thresholds: {}'.format(previous_thresholds))
        new_averages = {}
        for router, router_dict in previous_averages.items():
            new_averages[router] = {}
            for interface, interface_dict in router_dict.items():
                prev_avg = previous_averages[router][interface]
                inr = sample.get_routers_feat()[router][interface]['in_interests']
                if prev_avg - prev_avg * 0.5 <= inr <= prev_avg:
                    new_averages[router][interface] = 0.9 * prev_avg + 0.1 * inr
                elif prev_avg <= inr <= prev_avg + 0.1 * prev_avg:
                    new_averages[router][interface] = 0.99999 * prev_avg + 0.00001 * inr
                else:
                    new_averages[router][interface] = previous_averages[router][interface]
        # print('new thresholds: {}'.format(thresholds))
        return new_averages

    def insert_averages_in_history(self, sample, thresholds):
        setup_key = sample.get_sim_setup_key()
        sample_time = sample.get_time()
        try:
            self.averages_history[setup_key]
        except KeyError:
            self.averages_history[setup_key] = {}
        self.averages_history[setup_key][sample_time] = thresholds

    @staticmethod
    def is_congested(sample):
        timed_out = 0
        nacks = 0
        for router, router_dict in sample.get_routers_feat().items():
            for interface, interface_dict in router_dict.items():
                timed_out += sample.get_routers_feat()[router][interface]['in_timedout_interests']
                nacks += sample.get_routers_feat()[router][interface]['in_nacks']
        if timed_out > 100 * nacks:
            return True
        else:
            return False

    def predict(self, sample):
        averages = self.get_averages(sample)
        self.insert_averages_in_history(sample, averages)
        # print('\n\n\nself.thresholds_history: {}'.format(self.thresholds_history))
        congestion = CongestionAware.is_congested(sample)
        features_per_router = sample.get_routers_feat()
        predictions = {}
        for router, router_dict in features_per_router.items():
            predictions[router] = {}
            for interface, interface_dict in router_dict.items():
                inr = features_per_router[router][interface]['in_interests']
                avg = averages[router][interface]
                ratio = features_per_router[router][interface]['sr']
                # print('ratio: {}'.format(ratio))
                if inr > avg and ratio < 0.5 and not congestion:
                    predictions[router][interface] = 0
                else:
                    predictions[router][interface] = 1
        return predictions