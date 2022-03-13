# Arabic Dialect Identification 

- This project includes my work on Arabic Dialect Identification Project.

- Dialect Identification is the process of classifyingg Arabic text to Corresponding Dialect eg: Egyptian, Syrian, ...

- Used  [QADI](https://www.kooora.com/) Dataset with the help of [AIM Technologies API ](https://www.kooora.com/) to retrieve Data.

- Data Collected from Various twitter accounts from around 18 arabian countries

- Data consists of around 500k tweets with their corresponding Label

- I used the following tools:
    - python 3.8
    - Flask : to build the wep application
    - tensorflow & keras : for training Deep Learning model
    - sklearn : to train machine learning model

<br><br>

# How to Run

- clone this repo to your local machine
- due to github maximum file size i couldn't upload models to the repo but you can download them from [here](https://drive.google.com/drive/folders/1hi4qeTFQbjLxOgwyz4U1zz9WARLrCGKD?usp=sharing) ,and put them into assets folder
- run the following command "pip install requirements.txt" , better use a virtual environment

- run the following command "python app.py"
- wait for the application to boot up then open the following [link](http://127.0.0.1:8888/) on your browser 



<br>

# Repo Structure

- assets : this thould hold the models and all relevant object for the application
- templates : html files used for rendering the application
- utils : contains python script with helper functions to be used for preprocessing
- app.py : main entry point for the flask application
- concat4.csv : modified dataset with concatenaed rows
- data_fetching.ipynb : notebook used to consume aim api to retreive data
- dialect_dataset.csv : original dataset provided by aim with tweets ids for the data
- DL_Model.ipynb : Notebook used for training the deep learning model
- EDA.ipynb : notebook used to get insights from data
- full_data.csv : original dataset retreievd from aim API
- ML_Model.ipynb : notebook used to train the machine learning model

<br>

# ٍScreenshots 
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/1.PNG)
<br>
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/2.PNG)
<br>
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/3.PNG)
<br>
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/4.PNG)
<br>
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/5.PNG)
<br>
![alt text](https://github.com/msaoudallah/Arabic-Dialect-Identification/blob/master/images/6.PNG)
<br>


# ToDo:
- Fine-Tune a Transformers based model
- Augment With Madar Dataset
- Add Various models to select from in the Application