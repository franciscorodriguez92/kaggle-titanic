# Load data cleaned 
train <- read_csv("./produced_data/train_cleaned.csv")
test <- read_csv("./produced_data/test_cleaned.csv")

#XGboost needs numeric variables
combine  <- bind_rows(train, test) # bind training & test data
combine <- combine %>% mutate(Sex = if_else(Sex == "male", 0, 1), age_known = as.numeric(age_known),
                              cabin_known = as.numeric(cabin_known), young = as.numeric(young),
                              child = as.numeric(child), alone = as.numeric(alone),
                              large_family = as.numeric(large_family),
                              Embarked = recode(Embarked, "S" = "0", "C" = "1","Q" = "2"),
                              title = recode(title, "Mr." = "0", "Mrs." = "1","Miss." = "2", "Master." = "3", 
                                             "Other" = "4"))
train <- combine %>% filter(!is.na(Survived)) %>% select(Survived, Pclass, Sex, Age, SibSp, Parch,
                                                         Fare, fclass, cabin_known, young, child,
                                                         family, alone, large_family, Embarked, title) %>%
  mutate_all(funs(as.numeric))
test <- combine %>% filter(is.na(Survived)) %>% select(Survived, Pclass, Sex, Age, SibSp, Parch,
                                                       Fare, fclass, cabin_known, young, child,
                                                       family, alone, large_family, Embarked, title) %>%
  mutate_all(funs(as.numeric))



library("xgboost")
dtrain <- xgb.DMatrix(data = as.matrix(select(train, -Survived)),label = train$Survived)
dtest <- xgb.DMatrix(data = as.matrix(select(train[681:891,], -Survived)),label=train[681:891,]$Survived)
dtest_real <- xgb.DMatrix(data = as.matrix(select(test, -Survived)), label=test$Survived)
#default parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=0.5,
  colsample_bytree=1
)


xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 100
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print.every.n = 10
                ,early.stop.round = 20
                ,maximize = F
)
##best iteration = 79

xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = 6
  ,watchlist = list(val=dtest,train=dtrain)
  ,print.every.n = 10
  ,early.stop.round = 10
  ,maximize = F
  ,eval_metric = "error"
)
#model prediction
xgbpred <- predict(xgb1,dtest)
xgbpred <- ifelse(xgbpred > 0.5,1,0)


#confusion matrix
library(caret)
confusionMatrix(xgbpred, train[681:891,]$Survived)
#Accuracy - 0.8531%

predictions = xgbpred <- predict(xgb1,dtest_real)
predictions <- ifelse(predictions > 0.5,1,0)

submission <- read_csv("./raw_data/test.csv") %>% select(PassengerId)
submission$Survived <- predictions
write.csv(submission, file = "./produced_data/submit5_xgboost.csv", row.names = FALSE)


# Hyperparameter Tuning

searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75, 1),
                                colsample_bytree = c(0.6, 0.8, 1))
ntrees <- 100

#Build a xgb.DMatrix object
DMMatrixTrain <- dtrain

rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){

  #Extract Parameters to test
   #(https://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters)
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]

  xgboostModelCV <- xgb.cv(data =  DMMatrixTrain, nrounds = 21 , nfold = 5, showsd = TRUE,
                           metrics = "error", verbose = TRUE, "eval_metric" = "error",
                           "objective" = "binary:logistic", "max.depth" = 6, "eta" = 0.7,
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate)

  xvalidationScores <- xgboostModelCV
  #Save rmse of the last iteration
  rmse <- tail(xvalidationScores$test.rmse.mean, 1)

  print (c(c(rmse, currentSubsampleRate, currentColsampleRate)))
  return(c(rmse, currentSubsampleRate, currentColsampleRate))

})



