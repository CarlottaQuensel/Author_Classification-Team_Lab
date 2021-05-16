import numpy, math, sys, scipy

'''
after learning x numbers of features, initialize
list of x numbers of weights between +2 and -2

1. compute probabilities for all instances
2. for every weight:
    - count all training instances switched on by property
    - δB -> multiply by respective probability
    - δA -> count those also switched on by label

    -> A - B
3. update λ

-> repeat until convergence
'''
# some code examples from the internet

def compute_conditional_probability(self, lbl, instance):
        top = sum([self.lambda_weight(lbl, feature) for feature in instance.feature_vector])
        bottom = [sum([self.lambda_weight(label, feature) for feature in instance.feature_vector]) for label in self.labels]
        posterior = math.exp(top - scipy.misc.logsumexp(bottom))
        return posterior

def compute_negative_log_likelihood(self, batch):
    log_likelihood = sum([math.log(self.compute_conditional_probability(instance.label, instance)) for instance in batch])
    return log_likelihood * -1

def train(self, instances, dev_instances=None):


    """Construct a statistical model from labeled instances."""

    #create dict mapping features => ids
    features = 1
    feat_id_dict = {"___BIAS___": 0}
    labels = 0
    labels_dict = {}

    # get features
    for instance in instances:
        try:
            x = labels_dict[instance.label]
        except KeyError:
            labels_dict[instance.label] = labels
            labels += 1
        for feature in instance.features():
            #if feature not in stop:
                try:
                    x = feat_id_dict[feature]
                except KeyError:
                    feat_id_dict[feature] = features
                    features += 1

def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
    """Train MaxEnt model with Mini-batch Stochastic Gradient 
    """
    negative_log_likelihood = sys.maxint
    no_change_count = 0
    while True:
        minibatches = self.chop_up(train_instances, batch_size)
        for minibatch in minibatches:
            gradient = self.compute_gradient(minibatch)
            self.model += gradient * learning_rate
        new_negative_log_likelihood = self.compute_negative_log_likelihood(dev_instances)
        if new_negative_log_likelihood >= negative_log_likelihood:
            no_change_count += 1
            print("No change ", no_change_count)
            print("log likelihood was: ", new_negative_log_likelihood)
            if no_change_count == 5:
                break
        else:
            no_change_count = 0
            negative_log_likelihood = new_negative_log_likelihood
            self.save("model.db")
            print("log likelihood is: " + str(negative_log_likelihood))
