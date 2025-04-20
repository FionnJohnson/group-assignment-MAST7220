# group-assignment-MAST7220
Machine Learning with R - group assignment - Classification problem
# read in data 
data <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
data
# install rpart and rpart plot for trees 
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
# ensure y value is binary for classification
data$Diabetes_binary<- as.factor(data$Diabetes_binary)
# tree model for seeing if patient has diabetes
tree_model <- rpart(Diabetes_binary ~ ., data = data, method = "class")
# plot tree
rpart.plot(tree_model, type = 3, extra = 104, fallen.leaves = TRUE)
# install caret to split data
install.packages("caret")
library(caret)
set.seed(2)
# set data into train and test, 70% for train
train_index <- createDataPartition(data$Diabetes_binary, p =0.7, list = FALSE)
train_data <- data[train_index,]
test_data <- data[-train_index,]
tree_model <- rpart(Diabetes_binary ~., data = train_data, method = "class")
rpart.plot(tree_model, type = 3, extra = 104, fallen.leaves = TRUE)
# creat predictions on train data
train_predictions <- predict(tree_model, train_data, type ="class")
# create confusion matrix 
confusionMatrix(data = train_predictions, reference = factor(train_data$Diabetes_binary))
# create predicitons on test data 
tree_model <- rpart(Diabetes_binary ~., data = test_data, method = "class")
rpart.plot(tree_model, type = 3, extra = 104, fallen.leaves = TRUE)
test_predictions <- predict(tree_model, test_data, type ="class")
# create confusion matrix 
confusionMatrix(data = test_predictions, reference = factor(test_data$Diabetes_binary))
confusionMatrix
# state True Positive, False Positive and False Negative from confusion matrix
TP <- confusionMatrix["1","1"]
FP <- confusionMatrix["1","0"]
FN <- confusionMatrix["0","1"]
# calculate precision and recall
precision <- TP / (TP +FP)
recall <- TP / (TP+FN)
precision 
recall
# Define maxdepth grid
depth_grid <- expand.grid(maxdepth = 1:10)

# Set minsplit via control (manually)
custom_control <- rpart.control(minsplit = 10)
tuned_depth_model <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "rpart2",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = depth_grid,
  parms = list(split = "gini"),  # optional: can also use "information"
  control = custom_control
)
tuned_depth_model
plot(tuned_depth_model)
depth_preds <- predict(tuned_depth_model, newdata = test_data)

# Confusion matrix
conf_matrix_depth <- table(Predicted = depth_preds, Actual = test_data$Diabetes_binary)
print(conf_matrix_depth)

# Accuracy
accuracy_depth <- sum(diag(conf_matrix_depth)) / sum(conf_matrix_depth)
accuracy_depth

# Define tuning grid
tune_grid <- expand.grid(
  cp = seq(0.001, 0.05, by = 0.005)  # Try a range of complexity parameters
)

# Define cross-validation
train_control <- trainControl(
  method = "cv",      # Cross-validation
  number = 5,         # 5-fold
  verboseIter = TRUE  # Show progress
)
tuned_tree <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid
)
tuned_tree$bestTune
plot(tuned_tree)
install.packages("randomForest")
library(randomForest)
set.seed(2)
rf_model <- randomForest(
  Diabetes_binary ~ ., 
  data = train_data,
  ntree = 200,  # Number of trees
  importance = TRUE   # To get variable importance
)
# Predict using test data 
rf_preds <- predict(rf_model, newdata = test_data)
# Create confusion matrix 
conf_matrix_rf <- table(Predicted = rf_preds, Actual = test_data$Diabetes_binary)
conf_matrix_rf
# Statistics from confusion matrix
rf_cm <- confusionMatrix(rf_preds, test_data$Diabetes_binary)
rf_cm
TP <- conf_matrix_rf["1","1"]
FP <- conf_matrix_rf["1","0"]
FN <- conf_matrix_rf["0","1"]
recall
library(ggplot2)

# Set different numbers of trees to test
tree_counts <- seq(100, 1000, by = 50)
accuracies <- numeric(length(tree_counts))

set.seed(2)

for (i in seq_along(tree_counts)) {
  rf_model <- randomForest(Diabetes_binary ~ ., 
                           data = train_data, 
                           ntree = tree_counts[i])
  
  preds <- predict(rf_model, newdata = test_data)
  cm <- table(Predicted = preds, Actual = test_data$Diabetes_binary)
  acc <- sum(diag(cm)) / sum(cm)
  
  accuracies[i] <- acc
}

# Create a dataframe for plotting
results_df <- data.frame(
  Trees = tree_counts,
  Accuracy = accuracies
)

# Plot
ggplot(results_df, aes(x = Trees, y = Accuracy)) +
  geom_line(color = "forestgreen", size = 1.2) +
  geom_point(color = "darkgreen", size = 2) +
  theme_minimal() +
  labs(title = "Random Forest: Number of Trees vs Accuracy",
       x = "Number of Trees (ntree)",
       y = "Accuracy")
