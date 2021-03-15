This project considers that we are a consulting company that advise people to find out **which probability they have to get a successful fundraising** by kickstarters with their projects. We analyse the features of thier project and in specaly we will tell you what is better not to do but how our clients **can improve their chances** to reach their aim.  

## What is the project about and what is Kickstarter
This project uses a dataset from Kickstarter collected by https://webrobots.io/ to explore prediction models.
Kickstarter PBC is a funding platform for creative projects. Everything from films, games, and music to art, design, and technology. Kickstarter is full of ambitious, innovative, and imaginative ideas that are brought to life through the direct support of others.

Everything on Kickstarter must be a project with a clear goal, like making an album, a book, or a work of art. A project will eventually be completed, and something will be produced by it.

Kickstarter is not a store, backers pledge to projects to help them come to life and support a creative process. To thank their backers for their support, project creators offer unique rewards that speak to the spirit of what they're hoping to create.


## The Repository consists of:
* Project: Kickstarter - kickstarter_presentation.pdf summarizing findings in a presentation
* kickstarter_01_cleaning.ipynb & kickstarter_02_preparation.ipynb & kickstarter_03_feature_engineering.ipynb Jupyter Notebooks containing the code and steps of analyzing data
* kickstarter_EDA.ipynb : contains our EDA  
* kickstarter_04_model.ipynb : contains our exploratory modelling 
* kickstarter_success_predictor.py : script for training a model & predicting values using a test data set       
## Summary
* Random Forest model performs best with an accuracy of 0.80
* Baselinemodel with Dummy Classifier (Accuracy = 0.61)
* SVM Classifier (Accuracy = 0.74)
* KNN Classifier (Accuracy = 0.65)
* SGD Classifier with GridSearchCV (Accuracy = 0.61)
* Extra Trees (Accuracy = 0.79)
* Quadratic Discriminant Analysis (Accuracy = 0.67)
* AdaBoost GridSearchCV (Accuracy = 0.77)
* LightGBM GridSearchCV (Accuracy  = 0.79)
* XGBoost (Accuracy = 0.78)
* Naive Bayes (Accuracy = 0.73)
* most important influencers in model Random Forest are:
    * goal_usd
    * duration
    * blurb_len
    * launched_at_month & launched_at_year
    * country_US & country_GB
* projects in Hongkong have the highest chance to be successful, projects from Italy have only less than 50% chance
* projects in Hongkong have the highest chance to be successful, projects from Italy have only less than 50% chance

## Outlook
* further data cleaning, e.g. remove outliers from the data
* check if further new variables can help to improve the models
* Take a look on these projects which have collected more money as they initial wanted 
* further analysis on subcategories
* which specific words in the blurb correlate with successful projects
    
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
