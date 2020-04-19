# Auction-bidding-price-prediction

> This project was undertaken as a part of a private Data Science Challenge at UTD. 

## Data 

The data has 10 features 

ID : unique identifier of the data point 
Other feature : attributes of the data point in space. 
Feature 10 : Target variable 
Feature 5, 6, 7 and 8 : categorical in nature meaning although they are represented by a number, that number has no hierarchical significance(nominal data)

The first 3400 rows can be used for training/internal testing. The target variable is provided for them. 

The last 382 rows are evaluation data, for which the target is to be predicted and submitted.

## Problem Statement

Key challenges is to correctly predict the price of the vehicle that can be achieved in an auction with the limited amount of information available.

The data set provided is a sample of the high dimensional data we use for predictions. Feature 10 is the price that is to be predicted using other features provided. The categorical features (Make, Model etc.) are encoded for privacy.

The candidate is not expected to use any external data for this problem set but is encouraged to make use of any information available on the internet if it benefits the model. However, the final model should make predictions using only the information provided.

The model can be as complex as needed if it leads to better accuracy. Accuracy measures being defined by:
1.	Mean Absolute Error
2.	Mean Error
3.	Root Mean Squared Error

All of the above 3 metrics will be taken into consideration for final evaluation with MAE as the preferred one. For the final submission, the candidate should submit the target price for each of the data point in the evaluation data and the code in whichever language they used. To qualify for the presentation round, one must achieve a Mean error of 75 or less on the evaluation data.


```
bst <- xgboost(data = sparse_matrix, label = (Data_train$Feature_10), max_depth = 40,
               eta = 0.05, nthread = 2, nrounds = 300)

importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
head(importance)

#Validating predictions
pred_Y <- predict(bst, newdata=sparse_matrix)
pred_Z <- round(exp(1)^pred_Y, digits=2)
mean(abs(pred_Z - Data_train$Feature_10))
```

Prediction using XGBoost

```
pred <- predict(bst, newdata = sparse_test)
Data_test$Feature_10 <- round(exp(1)^pred, digits=2)
```
