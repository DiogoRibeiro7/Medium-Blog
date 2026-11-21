
set.seed(1); rm( list = ls() )

# Load packages ----
library(randomForest)
library(caret)
library(glmnet)
library(tidyverse)
library(e1071)
library(ipred)
library(nnet)
library(SuperLearner)
library(party)
library(xgboost)


# Load data ----

# Swiss cartels 
my_data <- read.csv("/PATH/TO/MY/DATA/Code and Data Journal/Swissdata.csv")

# Okinawa cartel (Remove the # if you want to execute the exercise on the Okinawa cartel)
# my_data <- read.csv("/PATH/TO/MY/DATA/Japanesedata.csv")

# Italian cartels (Remove the # if you want to execute the exercise on the Italian cartels)
# my_data <- read.csv("/PATH/TO/MY/DATA/Italiandata.csv")

# Swiss cartels using additional coalition-based screens (Remove the # if you want to execute the exercise on the Swiss cartels with additional screens)
# my_data <- read.csv("/PATH/TO/MY/DATA/Swissdata_addpred.csv")

# Swiss cartels with coalitions of four firms (Remove the # if you want to execute the exercise with coalitions of four firms)
# my_data <- read.csv("/PATH/TO/MY/DATA/Swissdata_four.csv")

# The investigation of predictors measuring asymmetry (remove the # if you want to exectue this extercise)
# vars <- c("min3skew1","max3skew1","mean3skew1","median3skew1","min3D","max3D","mean3D","median3D","min3RD1","max3RD1","mean3RD1","median3RD1","min3RDnorm","max3RDnorm","mean3RDnorm","median3RDnorm","min3RDalter","max3RDalter","mean3RDalter","median3RDalter","min3Diffperc","max3Diffperc","mean3Diffperc","median3Diffperc","cartelperiod","idfirm_")
# vars <- c("min3skew1","max3skew1","mean3skew1","median3skew1","min3RD1","max3RD1","mean3RD1","median3RD1","min3RDnorm","max3RDnorm", "mean3RDnorm","median3RDnorm","min3RDalter","max3RDalter","mean3RDalter","median3RDalter","cartelperiod","idfirm_")
# my_data <- my_data[vars]


# Prepare Machine Learning ----
my_data$number_project <- NULL 
my_data$Country <- NULL 
my_data$cartelperiod <- as.character(my_data$cartelperiod); my_data$cartelperiod <- as.factor(my_data$cartelperiod)
# Random Forest (with variable importance)
correctpred_rf=c(); correctpredcollusion_rf=c(); correctpredcocompeti_rf=c()
importvar_rf=100
# Support vector machines
correctpred_svm=c(); correctpredcollusion_svm=c(); correctpredcocompeti_svm=c()
# Lasso
correctpred_lasso = c(); correctpredcocompeti_lasso=c(); correctpredcollusion_lasso=c()
# Super Learner 
correctpred_SL = c(); correctpredcocompeti_SL=c(); correctpredcollusion_SL=c()
# predictive performance of the hundred repetitions
nsim=100

# Train/test data ----
for (j in 1:nsim){
  set.seed(j)

  # Balance data set and create test/training data ----
  DS2<-my_data[(my_data$cartelperiod==0),]
  DS3<-my_data[(my_data$cartelperiod==1),]
  balTI <- min(nrow(DS2),nrow(DS3))
  DS4<-DS2[sample(nrow(DS2),balTI),]
  DS5<-DS3[sample(nrow(DS3),balTI),]
  DS_final <- rbind(DS4,DS5)
  train=sample(nrow(DS_final),nrow(DS_final)*0.75)
  DS_final.test=DS_final[-train,]
  DS_final.train=DS_final[train,]

  # Random Forest ----
  rf <- randomForest(cartelperiod~.-idfirm_,ntree=1000, data=DS_final.train)
  # Variable importance
  importvar_rf <- cbind(varImp(rf),importvar_rf)
  # Predict outcome in test data
  DS_final.test$predicted.response <- predict(rf,DS_final.test)
  DS_final.test$Cor=ifelse(DS_final.test$predicted.response==DS_final.test$cartelperiod,1,0)
  correctpred_rf <- c(correctpred_rf,mean(DS_final.test$Cor))
  # Correct classification
  x <- confusionMatrix(data=DS_final.test$predicted.response,
                       reference=DS_final.test$cartelperiod)
  u <- x$table[1,1]
  o <- x$table[2,2]
  q <- x$table[1,1]+x$table[2,1]
  p <- x$table[1,2]+x$table[2,2]
  r <- u/q
  t <- o/p
  correctpredcocompeti_rf=c(correctpredcocompeti_rf,r); correctpredcollusion_rf=c(correctpredcollusion_rf,t)
  
  # Support vector machines ----
  tune.out=tune(svm,cartelperiod~.-idfirm_,data=DS_final.train,kernel="linear",
                ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100)))
  bestmod =tune.out$best.model
  ypred=predict (bestmod ,DS_final.test)
  t <- table(predict =ypred , truth=DS_final.test$cartelperiod )
  ## Correct classification
  r <- (t[1,1]+t[2,2])/(t[1,1]+t[1,2]+t[2,2]+t[2,1]); o <- (t[1,1])/(t[1,1]+t[2,1]); p <- (t[2,2])/(t[2,2]+t[1,2])
  correctpred_svm = c(correctpred_svm,r); correctpredcocompeti_svm=c(correctpredcocompeti_svm,o); correctpredcollusion_svm=c(correctpredcollusion_svm,p)
  
  # Lasso ----
  x.test=data.matrix(DS_final.test[,! colnames(DS_final.test) %in% c("cartelperiod", "idfirm_", "predicted.response", "Cor", "predict") ])
  x.train=data.matrix(DS_final.train[,! colnames(DS_final.train) %in% c("cartelperiod", "idfirm_") ])
  y.train=factor(DS_final.train$cartelperiod)
  cv.lasso=cv.glmnet(x=x.train,y=y.train,alpha=1,family = "binomial")
  fit.lasso=glmnet(x=x.train,y=y.train,family = "binomial")
  bestlam=cv.lasso$lambda.min
  lasso.pred=predict(fit.lasso,s=bestlam,newx=x.test, type="response")
  DS_final.test$predict <- ifelse(lasso.pred>0.5,1,0)
  t <- table(predict=DS_final.test$predict, truth=DS_final.test$cartelperiod)
  # Correct classification
  r <- (t[1,1]+t[2,2])/(t[1,1]+t[1,2]+t[2,2]+t[2,1]); o <- (t[1,1])/(t[1,1]+t[2,1]); p <- (t[2,2])/(t[2,2]+t[1,2])
  correctpred_lasso = c(correctpred_lasso,r); correctpredcocompeti_lasso=c(correctpredcocompeti_lasso,o); correctpredcollusion_lasso=c(correctpredcollusion_lasso,p)
  
  # Super Learner ----
  # in our case: random forests with bagging (cforest), elastic nets (glmnet), Extreme Gradient Boosting (xgboost) and neural networks (nnet)
  x.test=data.matrix(DS_final.test[,! colnames(DS_final.test) %in% c("cartelperiod", "idfirm_", "predicted.response", "Cor", "predict") ])
  x.train=data.matrix(DS_final.train[,! colnames(DS_final.train) %in% c("cartelperiod", "idfirm_") ])
  y.train=as.numeric(factor(DS_final.train$cartelperiod))-1
  k= 15
  cutoff=0.5
  slpred=SuperLearner(y.train,data.frame(x.train), newX=data.frame(x.test), family="binomial", SL.library=c("SL.cforest", "SL.nnet", "SL.xgboost", "SL.glmnet"), cvControl = list(V = k) )
  # verify the quality of prediction
  SLtestpredict=slpred$SL.predict
  DS_final.test$SLtestpredictbin=1*(SLtestpredict>cutoff)
  t <- table(predict=DS_final.test$SLtestpredictbin, truth=DS_final.test$cartelperiod)
  # Correct classification
  r <- (t[1,1]+t[2,2])/(t[1,1]+t[1,2]+t[2,2]+t[2,1]); o <- (t[1,1])/(t[1,1]+t[2,1]); p <- (t[2,2])/(t[2,2]+t[1,2])
  correctpred_SL = c(correctpred_SL,r); correctpredcocompeti_SL=c(correctpredcocompeti_SL,o); correctpredcollusion_SL=c(correctpredcollusion_SL,p)
 
}

# Support vector machines: Results ----
mean(correctpred_svm); mean(correctpredcollusion_svm); mean(correctpredcocompeti_svm)

# Lasso: Results ----
mean(correctpred_lasso); mean(correctpredcollusion_lasso); mean(correctpredcocompeti_lasso)

# Super Learner: Results ----
mean(correctpred_SL); mean(correctpredcollusion_SL); mean(correctpredcocompeti_SL)

# Random Forest: Results ----
mean(correctpred_rf); mean(correctpredcollusion_rf); mean(correctpredcocompeti_rf)

# Variable importance ----
importvar_rf$importvar_rf <- NULL
importvar_rf$mean <- rowMeans(importvar_rf)
importvar_rf <- importvar_rf[order(-importvar_rf$mean),]
rownames(importvar_rf)
head(importvar_rf$mean,n=10)

