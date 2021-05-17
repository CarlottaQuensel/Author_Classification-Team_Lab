# -*- coding: utf-8 -*-

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

class MaxEnt():
    features = [] # or maybe {}
    weight = []

    def __init__(self, features)