
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from algo_evaluation import LOGGER


def grid_search_best_parameter(split_dataset, clf, tuned_parameters):
    scores = ['precision', 'recall']

    for score in scores:
        LOGGER.info("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(clf(), tuned_parameters, cv=5, scoring=score)
        clf.fit(split_dataset['training']['features'], split_dataset['training']['labels'])

        LOGGER.info("Best parameters set found on development set:")
        LOGGER.info(clf.best_estimator_)
        LOGGER.info("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            LOGGER.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

        LOGGER.info("Detailed classification report:")
        LOGGER.info("The model is trained on the full development set.")
        LOGGER.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = split_dataset['test']['labels'], clf.predict(split_dataset['test']['features'])
        LOGGER.info(classification_report(y_true, y_pred))
