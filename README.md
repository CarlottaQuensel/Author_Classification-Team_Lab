# Author Classification
Katrin Schmidt &mdash; Carlotta Quensel

## About
This project investigates which linguistic or literary features inherent to poetry can be used to predict a poems author. It is part of the Team Laboratory NLP module for the M.Sc. Computational Liguistic at Uni Stuttgart. The data can be found in the respective folder and was sourced from the [Poetry Foundation](https://www.poetryfoundation.org/) as a finished csv dataset from [kaggle](https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset?select=kaggle_poem_dataset.csv).

### Method
To predict the authors, we use a maximum entropy classifier that was built from scratch. The classifier uses bag-of-word features as well as the length of the poem in terms of verse number and stanza number and the poem's rhyme scheme for classification. 

The data is preprocessed using NLTK's WordPunctTokenizer. Visualizations of the data statistics can be found in the data folder.

## Progress
> Detailed progress reports can be found [here](https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2425930_Group_4%3A_Carlotta_Nele_Farina_Quensel%2C_Katrin_Schmidt%2C_Author_Classification "Ilias wiki")

- [x] Getting the data from the [poetry foundation](https://www.poetryfoundation.org/)
- [x] Implementing the evaluation
- [x] Implementing the baseline classifier from scratch
- [x] Laying out the advanced method
- [ ] Implementing the advanced features (verse/stanza number and rhyme scheme)
- [ ] Fitting the advanced features into the baseline
