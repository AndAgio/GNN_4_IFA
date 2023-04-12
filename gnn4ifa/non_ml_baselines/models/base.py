from non_ml_baselines.data import Sample


class BaseDetector:
    def __init__(self):
        self.predictions_history = {}

    def insert_predictions_in_history(self, sample, predictions):
        setup_key = sample.get_sim_setup_key()
        sample_time = sample.get_time()
        try:
            self.predictions_history[setup_key]
        except KeyError:
            self.predictions_history[setup_key] = {}
        self.predictions_history[setup_key][sample_time] = predictions

    @staticmethod
    def global_prediction(interfaces_predictions):
        for router, router_dict in interfaces_predictions.items():
            for interface, interface_dict in router_dict.items():
                if interfaces_predictions[router][interface] == 1:
                    return 1
        return 0

    def predict(self, sample):
        raise NotImplementedError('Detector class should implement the predict method!')

    def test(self, dataset, metrics, verbose=True):
        all_predictions = []
        all_labels = []
        previous_sample = Sample()
        for sample_index, sample in enumerate(dataset):
            features_per_router = sample.get_routers_feat()
            label = 1 if sample.get_label() else 0
            all_labels.append(label)
            interfaces_predictions = self.predict(sample)
            self.insert_predictions_in_history(sample, interfaces_predictions)
            global_prediction = self.global_prediction(interfaces_predictions)
            all_predictions.append(global_prediction)
            previous_sample = sample
            print('\r| Sample {} out of {}... | Prediction: {} | Label: {} |'.format(sample_index + 1,
                                                                                     len(dataset),
                                                                                     global_prediction,
                                                                                     label),
                  end="\r")
        print()
        for met_key, met in metrics.items():
            print('{} = {:.3f} %'.format(met_key.upper(), 100. * met.compute(y_true=all_labels,
                                                                             y_pred=all_predictions)))

