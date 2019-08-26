import hyperopt
import sys


class MyObjective(object):
    """
    Code modifeid from here:
        https://github.com/catboost/tutorials/blob/master/classification/
        classification_with_parameter_tuning_tutorial.ipynb
    """

    def __init__(self, model, X, y, const_params, loss_fun):
        """
        loss_fun is a function containing the loss function expression.
        It must take an argument for the cv_results and must return the value
        that must be minimized by the Optimizer
        Example -np.mean(cv_results['test_recall_promoted_only']
                + cv_results['test_f1_weighted'])
        names must match what is there in scorer object
        """
        self.model = model
        self.X = X
        self.y = y
        self._const_params = const_params.copy()
        self.loss_fun = loss_fun

        self._evaluated_count = 0

    # hyperopt optimizes an objective using `__call__` method (e.g. by doing
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters
        # provided by the user

        params = hyper_params
        params.update(self._const_params)

        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()

        self.model.set_params(**params)  # Set model parameters

        loss = self.loss_fun(self.model, self.X, self.y)
        print('Loss={}'.format(loss), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count),
              file=sys.stdout)

        return {'loss': loss, 'status': hyperopt.STATUS_OK}
