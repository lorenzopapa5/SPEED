import numpy as np
from batches import get_batch_evaluation


def errors(gt, pred):
    valid_mask = gt > 0.0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))

    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()

    abs_diff = np.abs(pred_eval - gt_eval)

    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))

    abs_rel = np.mean(abs_diff / gt_eval)

    log_abs_diff = np.abs(np.log10(pred_eval) - np.log10(gt_eval))

    log_mae = np.mean(log_abs_diff)
    log_rmse = np.sqrt(np.mean(np.power(log_abs_diff, 2)))

    return mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3


class Metric(object):

    def __init__(self, model, bs, num_test, dts_test, img_shape, d_shape, dts=0):
        self.results = {}
        self.best_predictions = {}
        self.model = model
        self.batch = bs
        self.num_test_elem = num_test
        self.test_dataset = dts_test
        self.img_shape = img_shape
        self.d_shape = d_shape
        self.dts = dts
        self.eval_keys = ['mae', 'rmse', 'abs_rel', 'log_mae', 'log_rmse', 'delta1', 'delta2', 'delta3']

        for item in self.eval_keys:
            self.results[item] = []

    def update(self):
        for i in range(self.num_test_elem // self.batch):
            index_img = i * self.batch
            in_img, gt = get_batch_evaluation(bs=self.batch, num_test=self.num_test_elem, dts_test=self.test_dataset,
                                              img_shape=self.img_shape, d_shape=self.d_shape, dts=self.dts,
                                              index=index_img)
            pred = self.model.predict(in_img)
            assert (gt.shape == pred.shape)
            mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3 = errors(gt, pred)

            for item in self.eval_keys:
                self.results[item].append(eval(item))

        return self.best_predictions

    def display_avg(self):
        print("Evaluation Complete:")
        for item in self.eval_keys:
            print("{}: {:.4f}".format(item, np.mean(self.results[item])))
