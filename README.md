# Disaster Response Pipeline Project
This is a webapp created using plotly and flask. The main goal of this project
 was to crate a machine learning model to classify disaster messages.The Disaster Response project has educational purpose.

## Table of content
* [General Info](#general-info)
* [Technologies](#technologies)
* [Usage](#usage)
* [File Descriptions](#file-descriptions)
* [Acknowledgements](#acknowledgements)

## General Info
This project objective is to create a NLP model that helps disaster response
organizations to classify messages of disasters and direct the the messages for the correct disaster organizations, reducing the time response to the problems.

## Technologies

I used python 3.6.3 to code this project. You can find the package versions utilized [here](https://github.com/vitorscheifler/figure-eight-pipeline/blob/master/requirements.txt).

## Usage
To run this project, you will need to clone the repository in your local machine:
```bash
git clone https://github.com/vitorscheifler/figure-eight-pipeline.git
```
After cloning the repository, run the ETL pipeline to create a sqlite database:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/InsertDatabaseName.db
```

Then, run the train_classifier.py to create the machine learning model:

```bash
python models/train_classifier.py data/InsertDatabaseName.db models/classifier.pkl
```
 After creating the model, cd to the app folder and run run.py file.

```bash
python run.py
```
You can copy the port and open the corresponding ip with port on the browser.
(example http://127.0.0.1:5000/)

## File Descriptions
* app
- | - template
- | |- master.html # main page of web app
- | |- go.html # classification result page of web app
- |- run.py # Flask file that runs app
* data
- |- disaster_categories.csv # data to process
- |- disaster_messages.csv # data to process
- |- process_data.py # ETL process
- |- InsertDatabaseName.db # database to save clean data to
* models
- |- train_classifier.py # Train the model and save the model into a pickle file
- |- classifier.pkl # saved model
README.md

## Acknowledgements
Feel free to use this code as you would like.
