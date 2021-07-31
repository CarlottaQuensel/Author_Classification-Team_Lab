-AUTHOR CLASSIFICATION-
(Katrin Schmidt and Carlotta Quensel)
Author: Katrin Schmidt

Goal: As part of the CL Team Laboratory NLP, we classifiy the author of poetry through NLP methods. 


-DATASET-

poetryfoundation-dataset.csv

1.  Decided for the poetry collection of Poetry Foundation
2.  Downloaded finished dataset as csv from www.kaggle.com
3.  Normalized remaining unicode strings


-DATA PLOTS-

data_plotting.ipynb 
    provides an overall plot of poems per author, shown in a graphic

data_plotting_numbers.ipynb 
    allows queries with respect to specific numbers of authors/poems

verse_plotting.ipynb:
    provides a plot of verses per author, shown in a graphic
    very huge and very small verse counts are excluded


-DATA ANALYSIS-

data_analysis.ipynb
    provides an opportunity of analysing the data in a proper table
    data contains still some information about Poetry Foundation ID/Numeration

poetryfoundation_tokenizer.py 
    provides an overview of tokenizing data in our program