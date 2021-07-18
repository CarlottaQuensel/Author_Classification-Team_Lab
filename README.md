# Author Classification
Katrin Schmidt &mdash; Carlotta Quensel

## About
This project investigates which linguistic or literary features inherent to poetry can be used to predict a poem's author. We use four different feature types in several combinations. The project is part of the _Team Laboratory NLP_ module for the M.Sc. Computational Liguistic at Uni Stuttgart.

### Method
The data was sourced from the [Poetry Foundation](https://www.poetryfoundation.org/) as a finished csv dataset from [kaggle](https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset?select=kaggle_poem_dataset.csv).  This set as well as visualizations of the data statistics can be found in the data folder. The preprocessing included general html-cleanup and tokenization using NLTK's WordPunctTokenizer.

To predict the authors, we built a maximum entropy classifier from scratch that can be found under the baseline folder. The classifier uses bag-of-word features as well as the length of the poem in terms of verse number and stanza number and the poem's rhyme scheme for classification. 

## Progress
> Detailed progress reports can be found [here](https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2425930_Group_4%3A_Carlotta_Nele_Farina_Quensel%2C_Katrin_Schmidt%2C_Author_Classification "Ilias wiki")

- [x] Getting the data from the [poetry foundation](https://www.poetryfoundation.org/)
- [x] Implementing the evaluation
- [x] Implementing the baseline classifier from scratch
- [x] Laying out the advanced method
- [ ] Implementing the advanced features (verse/stanza number and rhyme scheme)
      * Extracting the features from the poems
      * Adapting pointwise mutual information for features
- [ ] Fitting the advanced features into the baseline
      * adapt feature class to apply features with different data types
- [ ] Testing different feature combinations
- [ ] Evaluate and report
