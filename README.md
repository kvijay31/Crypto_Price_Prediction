# Crypto Price Prediction: 

## Project Description: 

The aim of the project is to utilize machine learning and deep learning algorithms in predicting the price of crypto currencies. Here, 2 approached were followed: 

  - Standardized all crypto currencies and treated them as one(pooled data)
  - Created seperate models for each crypto currency(un-pooled data)
  
The notebooks can be viewed from the repository. However, the data has been confidential and as a result been hidde. If you are more interested in the project and the dataset, please contact me. 

## Insights: 

- Generally, the simpler models performed the best for pooled and non-pooled data
- Advanced model like Stacked LSTMs could be utilized after exploring tuning with substantially more computational resources.
- Pooling data restricts the  use of certain cutting-edge deep learning models .
- Normalization and non-normalized data produce similar results.
- Average performance of the non-pooled models is below that of the pooled models.
- Different amount of lags (e.g., an expanding window) might help model performance.
- More hyperparameter tuning or other models (e.g., LSTMs with attention) could help performance.
- All of the individual models on the non-pooled data have forecasts that follows the actual test return values quite closely. We suspect that the reason for our model's good performance is because our margin of error is low when predicting hourly returns. There is less volatility when predicting an hour into the future than predicting a day, week, or even month into the future.

## Models attempted: 

### Pooled Data: 
  - Baseline Models: Dummy Regressor, Linear Regression
  - Basic Models: XGBoostMLP, LSTM
  - Advanced Models: Stacked LSTM
  - Best Model: XGBoost
### Unpooled Data: 
  - Baseline Models: Linear Regression, Random Forest, LightGBM
  - Advanced Models: Temporal Convolutional Network
  - Best Model: Temporal Convolutional Network


## Model Overview: 

### Linear Regression & XGBOOST: 
   #### Motivation: 
    - Good starting point to get a baseline on model performance
    - Computationally inexpensive & fast to train
    - Explainability of results and allows for insights into the data
   #### Results: 
    - Predictions mainly around the mean, but does not model volatility well
    - Using default parameters already yields good results, hyperparameter tuning on XGBoost improves it even further
    
### MLP & LSTM:
   #### Motivation: 
    - Get a baseline for neural networks
    - RNN suffer from exploding or vanishing gradients and limited long term memory, LSTM relieves this issue
    - LSTM are neural networks specifically for Time Series Data use cases
    
   #### Results: 
    - XGBoost yields lower MSE on the test data, LSTM is even outperformed by linear regression
    - More complex model configurations (e.g., adding more than one hidden layer) lead to the models not learning properly, therefore hyperparameter            tuning effort was needed to make the model learn and not give a flatline prediction (i.e., to not predict the same value for all samples)

### TEMPORAL CONVOLUTIONAL NEURAL NETWORK: 
  #### Motivation: 
  - Realm of sequence modelling within deep learning has been largely associated with recurrent neural network architectures such as LSTMs.
  - When RNN’s are unrolled, they are extremely deep and have many hidden layers.
  - Thus there are many weights that will have to be updated through backpropagation. The gradient will then be passed through many hidden layers which       causes it to become extremely large or vanish altogether. 
  - Due to this issue, we investigated the possibility of using a convolutional neural network to predict our time series data
  

 





