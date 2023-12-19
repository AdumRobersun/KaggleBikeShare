#####Code for Kaggle Bike Share Competition#####
################################################



#call relevant libraries
library(tidyverse)
library(patchwork)
library(vroom)
library(tidymodels)
library(skimr)
library(DataExplorer)
#Read in data
BikeShareTrain <- vroom("Desktop/STAT348/KaggleBikeShare/train.csv")
BikeShareTest <- vroom("Desktop/STAT348/KaggleBikeShare/test.csv")


#remove registered, casual
BikeShareTrain <- BikeShareTrain %>%
  select(-casual, -registered)




#Feature Engineering
bike_recipe <- recipe(count~., data=BikeShareTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_time(datetime, features="hour") %>% 
  step_rm(datetime)

#Bake above recipe:

bike_recipe <- prep(bike_recipe)
train_baked <- bake(bike_recipe, new_data = BikeShareTrain)
test_baked <- bake(bike_recipe, new_data = BikeShareTest)

#-----LINEAR REGRESSION-----#

my_mod <- linear_reg() %>% #type of model
  set_engine("lm") # engine = what r function to use

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod) %>%
  fit(data = BikeShareTrain) # fit the workflow


#View the fitted linear regression model
extract_fit_engine(bike_workflow) %>%
  summary()

extract_fit_engine(bike_workflow) %>%
  tidy()

#Get Predictions for test set, format for Kaggle
test_preds <- predict(bike_workflow, new_data = BikeShareTest) %>%
  bind_cols(., BikeShareTest) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle

## Write prediction file to a CSV for submission
vroom_write(x=test_preds, file="LinearTestPreds.csv", delim=",")

#-----POISSON REGRESSION-----#

library(poissonreg)

mypoismodel <- poisson_reg() %>%
  set_engine("glm")

Pois_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(mypoismodel) %>%
  fit(data = BikeShareTrain)

Pois_Preds <-
  predict(Pois_wf, new_data = BikeShareTest)




extract_fit_engine(Pois_wf) %>%
  summary()
extract_fit_engine(Pois_wf) %>%
  tidy()

#get test predictions from poisson regression
Poisson_preds <- predict(Pois_wf, new_data = BikeShareTest) %>%
  bind_cols(., BikeShareTest) %>% # combine predicted values with test data
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


#put test predictions into kaggle csv
vroom_write(x=Poisson_preds, file="PoisTestPreds.csv", delim=",")


#------PENALIZED REGRESSION------#
library(poissonreg)
library(tidymodels)

log_bike_training <- BikeShareTrain %>% mutate(count = log(count))
## create a new recipe for penalized regression
pen_bike_receta <-
  recipe(count~., data=log_bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #change weather level 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% #cut hour from datetime
  step_rm(datetime) %>%
  step_rm(holiday) %>%
  step_rm(workingday) %>%
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


#create model
my_penalized_model <- linear_reg(penalty = 0, mixture = 0) %>%
  set_engine("glmnet")

penalized_wf <- workflow() %>%
  add_recipe(pen_bike_receta) %>%
  add_model(my_penalized_model) %>%
  fit(data=log_bike_training)

extract_fit_engine(penalized_wf) %>%
  tidy()

penalized_predictions <- exp(predict(penalized_wf, new_data=BikeShareTest)) %>%
  bind_cols(., BikeShareTest) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



## Write prediction file to a CSV for submission
vroom_write(x=penalized_predictions, file="PRegTestPreds.csv", delim=",")


#Tuning models for penalized regression

library(tidymodels)
library(poissonreg)

#make model for tuning:
tuned_pen_mod <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

#tuning workflow

pen_tune_wf <- workflow() %>%
  add_recipe(pen_bike_receta) %>%
  add_model(tuned_pen_mod)


#set up tuning grid 
tuning_grid <-
  grid_regular(penalty(),
               mixture(),
               levels = 10)

#split the data into K folds
folds <- vfold_cv(log_bike_training, v = 10, repeats = 1)

#Do cross validation on the K folds
CV_results <-
  pen_tune_wf %>% tune_grid(resamples = folds,
                             grid = tuning_grid,
                             metrics = metric_set(rmse, mae, rsq))

#plot cv results
collect_metrics(CV_results) %>%
  filter(.metric == 'rmse') %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

#find best tuning parameters

best_tune <- CV_results %>%
  select_best('rmse')
#finalize workflow

final_tuned_workflow <-
  pen_tune_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_training)

final_tuned_workflow %>% predict(new_data = BikeShareTrain)


#export to kaggle csv
tune_penalized_predictions <- exp(predict(final_tuned_workflow, new_data=BikeShareTest)) %>%
  bind_cols(., BikeShareTest) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



# get csv for tuned predictions for kaggle
vroom_write(x=tune_penalized_predictions, file="tunedtestpredictions", delim=",")






#---------REGRESSION TREES---------#
library(rpart)

rfrecipe <-
  recipe(count~., data=log_bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% 
  step_rm(datetime)

rfmod <- decision_tree(tree_depth = tune(),
                             cost_complexity = tune(),
                             min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

#create a workflow with model & recipe

dtree_wf <- workflow() %>%
  add_recipe(rfrecipe) %>%
  add_model(rfmod)


#set up a grid of tuning values

tuning_grid <-
  grid_regular(tree_depth(),
               cost_complexity(),
               min_n(),
               levels = 10)

#set up the k-fold cv
folds <- vfold_cv(log_bike_training, v = 5, repeats = 1)

#run the cross validation
CV_results <-
  dtree_wf %>% tune_grid(resamples = folds,
                         grid = tuning_grid,
                         metrics = metric_set(rmse, mae, rsq))



#find best tuning parameters
best_tune <- CV_results %>%
  select_best('rmse')

#finalize wf

final_decision_workflow <-
  dtree_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_bike_training)

### predict

decision_preds <- exp(predict(final_decision_workflow, new_data=BikeShareTest)) %>%
  bind_cols(., BikeShareTest) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle



#export predictions as csv file
vroom_write(x=decision_preds, file="DecisionTreePredictions", delim=",")





#-----RANDOM FORESTS-----#
library(rpart)
library(ranger)


rfrecipe <-
  recipe(count~., data=log_bike_training) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

### create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(rfrecipe) %>%
  add_model(rf_mod)


### set up a grid of tuning values

tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 10)

### set up the k-fold cv
folds <- vfold_cv(log_bike_training, v = 5, repeats = 1)

## run the cross validation
rf_CV_results <-
  rf_wf %>% tune_grid(resamples = folds,
                      grid = tuning_grid,
                      metrics = metric_set(rmse, mae, rsq))



### find best tuning parameters
rf_best_tune <- rf_CV_results %>%
  select_best('rmse')

## finalize workflow

rf_final_tuned_workflow <-
  rf_wf %>%
  finalize_workflow(rf_best_tune) %>%
  fit(data = log_bike_training)

### predict

rf_preds <- exp(predict(rf_final_tuned_workflow, new_data=BikeShareTest)) %>%
  bind_cols(., BikeShareTest) %>%
  select(datetime, .pred) %>% # only datetime and predicted count
  rename(count=.pred) %>% #rename to count for kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


## Write prediction file to a CSV for submission
vroom_write(x=rf_preds, file="RandomForestPreds", delim=",")

rf_preds %>% View()



library(stacks)


#-----STACKING MODELS-----#





# Remove casual and registered from the data set
BikeShareTrain <- BikeShareTrain %>%
  select(-casual, -registered)

#FEATURE ENGINEERING FOR STACKING
bikeshare_recipe <- recipe(count~., data=BikeShareTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_poly(temp, degree = 2) %>%
  step_poly(atemp, degree = 2) %>%
  step_poly(humidity, degree = 2) %>%
  step_poly(windspeed, degree = 2) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime) %>% # remove the original datetime column
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

#cv folds


folds <- vfold_cv(BikeShareTrain, v = 5)

#control settings for stacking models
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()






#penalized regression model stacking component
my_penalized_model_stacked <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

penalized_wf <-
  workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_penalized_model_stacked)

pen_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities


#Run the CV
my_penalized_models <- penalized_wf %>%
  tune_grid(resamples=folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untuned_model) 


#set up stacked linear model

lin_model_stacked <- linear_reg() %>%
  set_engine("lm")


#set up workflow
linreg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model_stacked)


#fit to folds
lin_reg_model <-
  fit_resamples(linreg_wf,
                resamples = folds,
                metrics = metric_set(rmse),
                control = tuned_model)


#random forest for stack
library(rpart)

rfrecipe <-
  recipe(count~., data=BikeShareTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather = factor(weather, levels = 1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season = factor(season, levels = 1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features="hour") %>% # split off the hour of day from datetime
  step_rm(datetime)




rfmod <- decision_tree(tree_depth = tune(),
                             cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")


#random forest stacking workflow

dtree_wf <- workflow() %>%
  add_recipe(rfrecipe) %>%
  add_model(rfmod)


#set up a grid of tuning values

dt_tuning_grid <-
  grid_regular(tree_depth(),
               cost_complexity(),
               
               levels = 5)




#Run the CV
dtree_models <- dtree_wf %>%
  tune_grid(resamples=folds,
            grid=dt_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untuned_model) 



#set up stacked model
my_bike_stack <-
  stacks() %>%
  add_candidates(my_penalized_model_stacked) %>%
  add_candidates(dtree_models) %>%
  add_candidates(lin_reg_model)

stack_mod <-
  my_bike_stack %>%
  blend_predictions() %>%
  fit_members()

#predict(stack_mod, new_data = BikeShareTest)


stacked_preds <- predict(stack_mod, new_data=BikeShareTest) %>%
  bind_cols(., BikeShareTest) %>%
  select(datetime, .pred) %>% # select just datetime and predicted count
  rename(count=.pred) %>% #rename pred to count to fit Kaggle format
  mutate(count=pmax(0, count)) %>% # pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to upload to Kaggle


#Write prediction file to a CSV for submission to kaggle
vroom_write(x=stacked_preds, file="StackingBikesPredictions.csv", delim=",")
############################################################################################################