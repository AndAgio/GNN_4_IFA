from .base import BaseDetector


class Coordination(BaseDetector):
    def __init__(self):
        super().__init__()
        self.tau = 0.3

    @staticmethod
    def get_pit_size(sample):
        features_per_router = sample.get_routers_feat()
        pit_sizes = {}
        for router, router_dict in features_per_router.items():
            pit_sizes[router] = 0.
            for interface, interface_dict in router_dict.items():
                pit_sizes[router] += features_per_router[router][interface]['pit_size']
        return pit_sizes

    def predict(self, sample):
        pit_sizes = Coordination.get_pit_size(sample)
        features_per_router = sample.get_routers_feat()
        predictions = {}
        for router, router_dict in features_per_router.items():
            predictions[router] = {}
            for interface, interface_dict in router_dict.items():
                per = features_per_router[router][interface]['per']
                pit = pit_sizes[router]
                # print('ratio: {}'.format(ratio))
                if per > 0. and pit > self.tau:
                    predictions[router][interface] = 1
                else:
                    predictions[router][interface] = 0
        return predictions
