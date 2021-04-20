# Predicting Solar Power Generation
---



### Objective

The project objective is to predict solar power generation for more efficient grid management.  


### Contents

01. [Data cleaning and EDA](code/01-data-cleaning-and-eda.ipynb)
02. [Preprocessing Data and Modelling](code/02-preprocessing-and-modelling.ipynb)
03. [Model Tuning](code/02-preprocessing-and-modelling.ipynb)


### Folder Organisation

    |__ code
    |   |__ 01-data-cleaning.ipynb   
    |   |__ 02-modelling-clustering.ipynb
    |   |__ 03-data-exploration.ipynb
    |   |__ 04-modelling-forecast.ipynb
    |   |__ 05-modelling-neural-nets.ipynb  
    |__ data
    |   |__ EMHIRES_PVGIS_TSh_CF_n2_19862015.csv
    |   |__ EMHIRESPV_TSh_CF_Country_19862015.csv
    |   |__ spain-energy-potential.csv
    |   |__ solar-ctry-clean.csv
    |   |__ solar-nuts-clean.csv
    |__ images
    |   |__ framework.png
    |   |__ seasonality.jpg
    |   |__ eda.jpg
    |__ presentation_slides
    |   |__ pricing-products-capstone.pdf
    |__ README.md



### Analysis and Findings

#### Data Cleaning and EDA

#### Preprocessing and modelling

<ins> L

<ins> FB Prophet </ins>

<ins> Recurrent Neural Networks </ins>

- Simple RNN
- LSTM
- GRU

### Data Dictionary

Below description of the dataset, sourced from [kaggle](https://www.kaggle.com/sohier/30-years-of-european-solar-generation). The data was made available by the [European Commission's STETIS Program](https://setis.ec.europa.eu/about-setis).


| Axis | Type | Description |
| :-----: | :--: | :---------- |
| Columns | str | European Country Codes |
| Rows | float | Hourly estimates of an area's energy potential for 1986-2015 as a percentage of a power plant's maximum output |



### References
