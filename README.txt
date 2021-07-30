-AUTHOR CLASSIFICATION-
(Katrin Schmidt and Carlotta Quensel)

As part of the CL Team Laboratory NLP, we classify the author of poetry through NLP methods. 
Detailed progress reports can be found here:
https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2425930_Group_4%3A_Carlotta_Nele_Farina_Quensel%2C_Katrin_Schmidt%2C_Author_Classification 


-INSTALLATION AND DOWNLOADS-

1. Download the program on github.
2. Make sure that you installed the following modules: 

- pronouncing
- nltk 
- numpy

-USAGE-

1.  Navigate to main.py

2.  Set the path to the current folder (in which the poems.json file is located)

3.  Dataset
    Option 1: Change the number of authors for the data with the parameter "max_author":
              build_dataset(token_data, max_author=30)
              
4.  learnFeatures
    Obligatory: presence of the parameter "data" and at least 1 "feature"
    Option 1: Within the first parameter you can choose on the data 
              from that the features are learned (train_set or test_set)
    Option 2: Change the number of bow features that are learned
              (e.g. bow_features=30)
    Option 3: Switch the verse features on/off 
              (verse_features=True or verse_features=False)
    Option 4: Change the number of rhyme features that are learned
              (e.g. rhyme_features=5)
    Option 5: Add the parameter "vocabulary=vocabulary" in order to obtain an 
              overview of the feature assignment to indexed words
    Option 6: Change the presence or absence of the trace that keeps track of the program 
              (trace=True or trace =False)

    Your settings could look like the following:          
    classifier.learnFeatures(train_set, bow_features=30, verse_features=True,
                         rhyme_features=5, vocabulary=vocabulary, trace=True)
5.  Train
    Obligatory: presence of the parameter "data"
    Option 1:   Within the first parameter you can choose on the data 
                from that the features are learned (train_set or test_set)
    Option 2:   Change the threshold for the improvement within on training iteration
                (e.g. min_improvement=0.001)
    Option 3:   Change the presence or absence of the trace that keeps track of the program 
                (trace=True or trace =False)

    Your settings could look like the following:
    classifier.train(train_set, min_improvement=0.001, trace=True)

6.  Run the program
