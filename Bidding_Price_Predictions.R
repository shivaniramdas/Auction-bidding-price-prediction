library(tidyverse)

Data <- read_csv("UTD_DataSet_Final.csv")
glimpse(Data)
summary(Data)
names(Data)
head(Data)

Data$ID <- NULL

Data$Feature_10[is.na(Data$Feature_10)] <- round(mean(Data$Feature_10, na.rm = TRUE))

summary(Data)

library(corrplot)
M<-cor(Data)
corrplot(M, method="circle")

#Individual correlations
cor(Data_train$Feature_10, Data_train$Feature_1, method = "pearson")

#Converting the variables to factors
Data$Feature_5 <- as.factor(Data$Feature_5)
Data$Feature_6 <- as.factor(Data$Feature_6)
Data$Feature_7 <- as.factor(Data$Feature_7)
Data$Feature_8 <- as.factor(Data$Feature_8)

#Splitting the training Data
Data_train <- Data[1:3400,]
summary(Data_train)
glimpse(Data_train)
head(Data_train)
Data_test <- Data[3401:3782,]

#Feels Outlier
Data_train[(Data_train$Feature_10==51500),]

#plot independent var
hist(Data_train$Feature_10)

boxplot(Data_train$Feature_10)

#Since data is left skewed, we can take log to have a normal distribution
hist(log(Data_train$Feature_10))

boxplot(log(Data_train$Feature_10))


#XGBoost ----
library(xgboost)
library(Matrix)

#creating sparse matrix for applying XGBoost
sparse <- sparse.model.matrix(Feature_10 ~ ., data = Data)[,-1]
nrow(sparse)
head(sparse)
sparse_matrix <- sparse[1:3400,]
sparse_test <- sparse[3401:3782,]
nrow(sparse_test)


bst <- xgboost(data = sparse_matrix, label = (Data_train$Feature_10), max_depth = 40,
               eta = 0.05, nthread = 2, nrounds = 300)

importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
head(importance)

#Validating predictions
pred_Y <- predict(bst, newdata=sparse_matrix)
pred_Z <- round(exp(1)^pred_Y, digits=2)
mean(abs(pred_Z - Data_train$Feature_10))

#Prediction using XGBoost
pred <- predict(bst, newdata = sparse_test)
Data_test$Feature_10 <- round(exp(1)^pred, digits=2)


#creating the submission file ----
Data_ID <- read_csv("UTD_DataSet_Final.csv")
Data_ID <- Data_ID[3401:3782,]
Data_test$ID <- Data_ID[,"ID"]

Submission <- data.frame(ID = Data_test$ID, Feature_10 = Data_test$Feature_10)
glimpse(Submission)
write_csv(Submission, "submission.csv")

