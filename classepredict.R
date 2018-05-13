##Load Packages
library(data.table)
library(caret)
library(e1071)
library(randomForest)
library(gbm)

##Read Data
setwd("C:/Users/owner/Dropbox/Coursera/Data Science/MachineLearning/project")
training <- read.csv("pml-training.csv",
                  na.strings = c("NA", "#DIV/0!",""))
test <- read.csv("pml-testing.csv",
              na.strings = c("NA", "#DIV/0!",""))

##Subset training data as training and cross-validation sets
inTrain <- createDataPartition(y = training$classe, p = 0.6, list = FALSE)
modTrain <- training[inTrain, ]
modTest <- training[-inTrain, ]

##Remove variables mostly containing NAs (threshold of 60%)
modTrainTemp <- modTrain
for (i in 1:length(modTrain)) {
  if(sum(is.na(modTrain[ , i])) / nrow(modTrain) >= .6) {
    for (j in 1:length(modTrainTemp)) {
      if (length(grep(names(modTrain[i]), names(modTrainTemp)[j])) == 1) {
      modTrainTemp <- modTrainTemp[ , -j]
      }
    }
  }
}

##Remove fields irrelevant to prediction
modTrainTemp <- modTrainTemp[, 8:length(modTrainTemp)]
modTrainFinal <- modTrainTemp
rm(modTrainTemp)

##Model Selection
set.seed(42)

#Train Random Forest, Gradient Boosted and Linear Discriminant models
rf_mod <- randomForest(classe ~ ., data = modTrainFinal)
gbm_mod <- train(classe ~ ., data = modTrainFinal, method = "gbm")
lda_mod <- train(classe ~ ., data = modTrainFinal, method = "lda")

#Use three models to predict classe in modTest set
rf_pred <- predict(rf_mod, modTest)
gbm_pred <- predict(gbm_mod, modTest)
lda_pred <- predict(lda_mod, modTest)

#Test accuracy of each of the four models
rf_acc <- confusionMatrix(rf_pred, modTest$classe)$overall[1]
gbm_acc <- confusionMatrix(gbm_pred, modTest$classe)$overall[1]
lda_acc <- confusionMatrix(lda_pred, modTest$classe)$overall[1]
acc <- c(rf_acc, gbm_acc, lda_acc)
names(acc) <- c("RF", "GBM", "LDA")
print(acc)

##Predict classe for final test set
predFinal <- predict(rf_mod, newdata = test, type = "class")
print(predFinal)