from sklearn.metrics import accuracy_score

def evaluation_score(observations, predictions):
    return accuracy_score(observations, predictions)