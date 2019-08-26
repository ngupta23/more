import hyperopt
import numpy as np


class HyperoptHelper():
    """
    Code modifeid from here:
        https://github.com/catboost/tutorials/blob/master/classification/
        classification_with_parameter_tuning_tutorial.ipynb
    """

    def __init__(self, space, objective, max_evals, random_state):
        """
        X: Features to be used for training
        y: output variables for supervised learning problem.
        space: Search Space for Hyperparameters. Example
            space = {'n_clusters': hyperopt.hp.choice('n_clusters',
                                                      np.arange(3,
                                                                11,
                                                                dtype=int)),
                     'n_init': hyperopt.hp.choice('n_init',
                                                  np.arange(10,
                                                            100,
                                                            dtype=int)),
                     'max_iter': hyperopt.hp.choice('max_iter',
                                                    np.arange(300,
                                                              500,
                                                              dtype=int)) }
        objective: MyObjective Class Object
        max_evals: Number of times to try hyperparameter optimization
        random_state: Seed value to use
        """

        self.space = space
        self.objective = objective
        self.max_evals = max_evals
        self.random_state = random_state

    def find_best_hyper_params(self):
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
                fn=self.objective,
                space=self.space,
                algo=hyperopt.rand.suggest,
                max_evals=self.max_evals,
                rstate=np.random.RandomState(seed=self.random_state)
                )
        return best

    def train_best_model(self):
        best = self.find_best_hyper_params()

        # merge subset of hyper-parameters provided by hyperopt
        # with hyper-parameters provided by the user

        # Need to evaluate space for hp.choice values in space (Categorical)
        # https://github.com/hyperopt/hyperopt/issues/284
        hyper_params = hyperopt.space_eval(self.space, best)
        hyper_params.update(self.objective._const_params)

        # drop `use_best_model` because we are going to use entire dataset for
        # training of the final model
        hyper_params.pop('use_best_model', None)

        print('Best Hyperparameters={}'.format(hyper_params))
        model = self.objective.model.set_params(**hyper_params)

        if (self.objective.y is None):
            model.fit(self.objective.X)
        else:
            model.fit(self.objective.X,
                      self.objective.y)

        return model, hyper_params
