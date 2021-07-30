# Author Classification
Katrin Schmidt &mdash; Carlotta Quensel


## About
As part of the CL Team Laboratory NLP, we classify the author of poetry through NLP methods.
Detailed progress reports can be found [here](https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2425930_Group_4%3A_Carlotta_Nele_Farina_Quensel%2C_Katrin_Schmidt%2C_Author_Classification "Ilias wiki")

This project deals with author prediction as opposed to author allocation, which means that we predict the author of a given poem from a learned set of authors. 

## Data
The data we use is the collection of the [Poetry Foundation](https://www.poetryfoundation.org/), which is pulled from [kaggle](https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset?select=kaggle_poem_dataset.csv) as a finished csv-database. Depicted below is the distribution of poems and authors over the data set.
The data includes many authors who only wrote 1-5 poems, which is why we used only the 30 most prolific authors to get enough data points per class for the method.
> This number of authors is a hyperparameter that can be changed in ```main.py```:

> ```build_dataset(token_data, max_author=30)```

### Preprocessing
Katrin sorted the poems by author, cleaned up remaining HTML from the poems and tokenized them with the NLTK WordPunctTokenizer (see ```poetryfoundation_tokenizer.py```). Then Carlotta split the data into train and test set and converted the poems into bag-of-word ("bow") vectors using the vocabulary in the train set (```tok_to_vec()``` in ```main.py```). 
With that, the baseline implementation begins.

## The Program
### Main
To run the implementation, the **path to the current folder** has to be set at the beginning of ```main.py```, which then runs all following methods consecutively. The program begins by splitting the data into training and test set. Following the features needed for the classifier are learned. For this purpose the user can choose on the data (training set or test set) and on the number of features that are learned at most for each class (or author). Furthermore the three features can be switched on or off:
> ```classifier.learnFeatures(train_set, bow_features=30, verse_features=True,```
                      >```rhyme_features=5, vocabulary=vocabulary, trace=True)```

### Features
The classifier uses ```advanced_features.py``` which was written by Carlotta and includes a class for maximum entropy features. Apart from that it includes two methods that calculate pointwise mutual information ("pmi") respectively for the bow-features and the rhyme-fetures. The verse-features are computed by a counting algorithm. The information of the several methods is then passed to the method which bundles them together and constructs them to a feature object.

#### PMI
In order to use primarily relevant features, the bow and the rhyme features are computed with pointwise mutual information. 

#### Bow
The bow features consist of vectors and are constructed within ```document.py```.

#### Rhyme/Verses
The rhymes are computed with the help of an additional pronouncing module, also in ```document.py```
The verses are computed by a counting algorithm while iteration and then being put into several bins. The verse features are built within ```advanced_features.py```


### Training
After learning the features, the classifier's weights are randomized between -10 and 10 for each feature. The training was written by Katrin and is done by either running the whole ```main.py``` at once (with preset hyperparameters) or by typing

> ```classifier.train(data, min_improvement=0.001, trace=True)```

The first parameter sets the threshold over which the improvement of accuracy between training epochs should lie and with the second parameter the user can track the improvement of accuracy and loss at each training step. The second method used as a part of the training is the partial derivative, which was also written by Katrin and follows the formula for maximum entropy training.

### Classification
The classification method was written by Carlotta. Apart from the straightforward implementation of the maximum entropy formula, it also includes an option to set custom weights for checking the accuracy of a training update before commiting to it and an option to get the probability of a label, which is used during training.
The poem is passed to the method as a document vector and the method returns the most probable author.

## Evaluation
The classifier is evaluated by instanciating an element of the Evaluation class (see ```evaluation.py```). The class requires the true authors and author predictions as two separate lists and computes all scores based on a confusion matrix. The main framework of the evaluation was written by Carlotta while Katrin implemented the precision, recall and f1-score. To conduct a full evaluation, the method ```fullEval()``` can be used, all other methods (including formated outputs) can be looked up in the class description.
