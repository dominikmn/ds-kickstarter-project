# Kickstarter Project (alpha version)
The final version version of this repo is found [here](https://github.com/mue94/ds-kickstarter-project).

---------------------------------------

This project considers that we are a consulting company that advise people to find out **which probability they have to get a successful fundraising** by kickstarters with their projects. We analyse the features of thier project and in specaly we will tell you what is better not to do but how our clients **can improve their chances** to reach their aim.  

## What is the project about and what is Kickstarter
This project uses a dataset from Kickstarter collected by https://webrobots.io/ to explore prediction models.
Kickstarter PBC is a funding platform for creative projects. Everything from films, games, and music to art, design, and technology. Kickstarter is full of ambitious, innovative, and imaginative ideas that are brought to life through the direct support of others.

Everything on Kickstarter must be a project with a clear goal, like making an album, a book, or a work of art. A project will eventually be completed, and something will be produced by it.

Kickstarter is not a store, backers pledge to projects to help them come to life and support a creative process. To thank their backers for their support, project creators offer unique rewards that speak to the spirit of what they're hoping to create.


## The Repository consists of:
* Project: Kickstarter - XXX.pdf summarizing findings in a presentation
* XXX.ipynb Jupyter Notebook containing the code and steps of analyzing data
* a figure showing the most important influencers for the Model
* XXX.ipynb : contains our EDA.    
* XXX.ipynb : contains our exploratory modelling.  
* train.py : script for training a model  
* predict.py : script for predicting values using a test data set   
* feature_engineering.py : library of functions used in train and predicitions    
* environment.yml: includes environment requirements

## Summary
* model performs best with an accuracy of 
* SVM
* KNN
* with the AdaBoost model 71% of failure and 81% of successful cases can be predicted correctly
* most important influencers in model (name the model) are
    * 
    * 
    * 
    * 
    * 
* KNN has at least ... as most important features
* projects in Hongkong have the highest chance to be successful, projects from Italy have only less than 50% chance

## Outlook
* further data cleaning, e.g. remove outliers from the data
* get out most important features from SVM to identify the most common feature that influence success
* check if further new variables can help to improve the models
* check other models
* Take a look on these projects which have collected more money as they initial wanted 
* analyze how backers count might be affected to find out what attracts many backers

## To train the model run the following (in our case data_folder is 'kickstarter/data'):
    XXX.py <data_folder>

## To make predictions using a dataset (one is provided, replace filename as necessary) use the following command:
    XXX.py models/XXX_model.sav test_data/X_test.csv test_data/y_test.csv
    
## Environment & packages via makefile


### Enviroment:
- python=3.8.5
- python -m venv .venv
- source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

### Packages: 

- jupyterlab==3.0.7
- matplotlib==3.3.4
- numpy==1.20.1
- pandas==1.2.1
- scipy==1.6.0
- seaborn==0.11.1
- statsmodels==0.12.2
- sklearn==0.0
