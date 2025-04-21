# group-assignment-MAST7220
Machine Learning with R - group assignment - Classification problem
# read in data 
data <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
data

# Installing necessary libraries for classification and regression models  
install.packages("rpart") #rpart and rpart plot for trees
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(tidyverse)  # Data manipulation and visualization
library(caret)      # Classification and Regression Training
library(xgboost)    # XGBoost algorithm implementation
library(Matrix)     # Sparse matrix handling
library(hnp)        # Half-Normal Plot for diagnostics
library(broom)      # Tidy model output
library(gridExtra)  # Plot arrangement
library(ROCR)       # Visualization tools for model performance
library(ggplot2)    # Advanced data visualization
library(pROC)       # ROC curve analysis

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

# === LOGISTIC REGRESSION MODEL ===
# Convert relevant variables to factors
cat_vars <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
              "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
              "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
              "Sex", "Education", "Income", "Diabetes_binary")

data <- data %>%
  mutate(across(all_of(cat_vars), as.factor))

# Data Exploration
str(data)
summary(data)

# Logistic Regression Model (Full)

full_model <- glm(Diabetes_binary ~ ., data = train_data, family = "binomial")

summary(full_model)

# Stepwise Model Selection 
final_model <- step(full_model, direction = "both")
summary(final_model)
exp(coef(final_model))  # Odds ratios

#  Predictions on Test Data (Logistic Regression)
prob_pred <- predict(final_model, newdata = test_data, type = "response")
class_pred <- ifelse(prob_pred > 0.5, "1", "0") %>% as.factor()

#  Model Evaluation
conf_matrix <- confusionMatrix(class_pred, test_data$Diabetes_binary, positive = "1")
print(conf_matrix)

# ROC Curve & AUC
roc_curve <- roc(test_data$Diabetes_binary, prob_pred)
plot(roc_curve, main = "ROC Curve for Logistic Regression")
auc(roc_curve)

# HNP Plot (Model Diagnostics)
hnp(final_model, main = "HNP Plot for Logistic Regression")

# === XGBOOST MODEL ===

# Verify class distribution (should be 50-50 as per dataset description)
table(data$Diabetes_binary)
prop.table(table(data$Diabetes_binary))

# Visualize class distribution
ggplot(data, aes(x = factor(Diabetes_binary))) +
  geom_bar(fill = c("skyblue", "salmon")) +
  labs(title = "Distribution of Diabetes Classes",
       x = "Diabetes Status (0 = No, 1 = Yes)",
       y = "Count") +
  theme_minimal()


# Relationships between key predictors and target variable
ggplot(data, aes(x = factor(Diabetes_binary), y = BMI, fill = factor(Diabetes_binary))) +
  geom_boxplot() +
  labs(title = "BMI Distribution by Diabetes Status",
       x = "Diabetes Status (0 = No, 1 = Yes)",
       y = "BMI",
       fill = "Diabetes Status") +
  theme_minimal()


# Correlation analysis for numerical variables
numeric_vars <- sapply(data, is.numeric)
correlation_matrix <- cor(diabetes_data[, numeric_vars])
print("Correlation matrix of numerical variables:")
print(correlation_matrix)


# Visualize correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "color", 
         type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


# === DATA PREPROCESSING ===

# Convert all variables to appropriate types
# For this dataset, most variables seem to be numeric but some may need to be treated as factors
# Age, Education, and Income are likely categorical/ordinal
data$Age <- as.factor(data$Age)
data$Education <- as.factor(data$Education)
data$Income <- as.factor(data$Income)
data$Sex <- as.factor(data$Sex)

# class distribution in training and testing sets
cat("Training set class distribution:\n")
prop.table(table(train_data$Diabetes_binary))
cat("Testing set class distribution:\n")
prop.table(table(test_data$Diabetes_binary))


# Preparing data for XGBoost
# XGBoost requires data in specific formats (matrix for features, vector for labels)
# Converting categorical variables to one-hot encoding 
# Prepare training data
train_labels <- train_data$Diabetes_binary
train_features <- train_data %>% select(-Diabetes_binary)


# Convert categorical variables to dummy variables
dummies <- dummyVars(" ~ .", data = train_features)
train_features_dummy <- predict(dummies, newdata = train_features)
train_matrix <- xgb.DMatrix(data = as.matrix(train_features_dummy), label = train_labels)


# Prepare testing data
test_labels <- test_data$Diabetes_binary
test_features <- test_data %>% select(-Diabetes_binary)
test_features_dummy <- predict(dummies, newdata = test_features)
test_matrix <- xgb.DMatrix(data = as.matrix(test_features_dummy), label = test_labels)



# === BUILD XGBOOST MODEL ===

# Define hyperparameters for XGBoost
xgb_params <- list(
  objective = "binary:logistic",  # Binary classification with logistic regression
  eval_metric = "logloss",        # Logarithmic loss
  eta = 0.1,                      # Learning rate
  max_depth = 6,                  # Maximum tree depth
  min_child_weight = 1,           # Minimum sum of instance weight needed in a child
  subsample = 0.8,                # Subsample ratio of the training instances
  colsample_bytree = 0.8          # Subsample ratio of columns when constructing each tree
)


# Train the XGBoost model with early stopping
xgb_model <- xgb.train(
  params = xgb_params,
  data = train_matrix,
  nrounds = 1000,                # Maximum number of boosting rounds
  watchlist = list(train = train_matrix, test = test_matrix),
  early_stopping_rounds = 50,    # Stop if performance doesn't improve for 50 rounds
  print_every_n = 50,            # Print evaluation metrics every 50 iterations
  verbose = 1                    # Print training information
)

# === MODEL EVALUATION ===

# Make predictions on test set
xgb_pred_prob <- predict(xgb_model, as.matrix(test_features_dummy))
xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, 1, 0)

# Create confusion matrix
conf_matrix <- confusionMatrix(factor(xgb_pred_class), factor(test_labels))
print("Confusion Matrix and Statistics:")
print(conf_matrix)


# Calculate other performance metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- conf_matrix$byClass["F1"]
specificity <- conf_matrix$byClass["Specificity"]

# Print performance metrics
cat("Performance Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Specificity:", specificity, "\n")

# === FEATURE IMPORTANCE ANALYSIS ===

# Get feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train_features_dummy), model = xgb_model)
print("Feature Importance:")
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix, top_n = 20, main = "Top 20 Feature Importance")

# === MODEL INTERPRETATION ===

xgb.save(xgb_model, "diabetes_xgboost_model.model")
conf_matrix$overall["Accuracy"]
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))
conf_matrix$byClass["Sensitivity"]
conf_matrix$byClass["Specificity"]
conf_matrix$byClass["Pos Pred Value"]
conf_matrix$byClass["F1"]

# Calculate ROC and AUC
roc_obj <- roc(test_labels, xgb_pred_prob)
auc_value <- auc(roc_obj)
cat("Area Under the Curve (AUC):", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, main = "ROC Curve for XGBoost Model", col = "blue")

                      # --- K nearest Neighbours --- #

# --- imports --- #
library(class)
library(dplyr)
library(caret)
library(janitor)

# --- loading data --- #
data <-janitor::clean_names(data)

# Converting the response variable into a factor variable
data$diabetes_binary <- factor(data$diabetes_binary, labels = c("No", "Yes"))

predictors <- c("high_bp", "high_chol", "gen_hlth", "age", "phys_activity", 
                "bmi", "heart_diseaseor_attack")

# Standardize/scale the relevant features
scaled_data <- scale(data[, predictors])


# Create a training and testing split (70-30)
set.seed(2)
train_index <- sample(1:nrow(data), size = 0.7 * nrow(data))

train_x <- scaled_data[train_index, ]
test_x  <- scaled_data[-train_index, ]

train_y <- data$diabetes_binary[train_index]
test_y <- data$diabetes_binary[-train_index]


# Running the K-NN model (with k = 20)
knn_pred <- knn(train_x, test_x, cl = train_y, k = 30)


# Evaluating the performance
conf_matrix <- table(Predicted = knn_pred, Actual = test_y)
print(conf_matrix)
accuracy <- mean(knn_pred == test_y)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# K = 1
knn_pred_k1 <- knn(train_x, test_x, cl = train_y, k = 1)

conf_matrix_k1 <- table(Predicted = knn_pred_k1, Actual = test_y)
print(conf_matrix_k1)
accuracy_k1 <- mean(knn_pred_k1 == test_y)
cat("Accuracy:", round(accuracy_k1 * 100, 2), "%\n")

# K = 3
knn_pred_k3 <- knn(train_x, test_x, cl = train_y, k = 3)

conf_matrix_k3 <- table(Predicted = knn_pred_k3, Actual = test_y)
print(conf_matrix_k3)
accuracy_k3 <- mean(knn_pred_k3 == test_y)
cat("Accuracy:", round(accuracy_k3 * 100, 2), "%\n")

# K = 7
knn_pred_k7 <- knn(train_x, test_x, cl = train_y, k = 7)

conf_matrix_k7 <- table(Predicted = knn_pred_k7, Actual = test_y)
print(conf_matrix_k7)
accuracy_k7 <- mean(knn_pred_k7 == test_y)
cat("Accuracy:", round(accuracy_k7 * 100, 2), "%\n")

# K = 9
knn_pred_k9 <- knn(train_x, test_x, cl = train_y, k = 9)

conf_matrix_k9 <- table(Predicted = knn_pred_k9, Actual = test_y)
print(conf_matrix_k9)
accuracy_k9 <- mean(knn_pred_k9 == test_y)
cat("Accuracy:", round(accuracy_k9 * 100, 2), "%\n")

# K = 50
knn_pred_k50 <- knn(train_x, test_x, cl = train_y, k = 50)

conf_matrix_k50 <- table(Predicted = knn_pred_k50, Actual = test_y)
print(conf_matrix_k50)
accuracy_k50 <- mean(knn_pred_k50 == test_y)
cat("Accuracy:", round(accuracy_k50 * 100, 2), "%\n")

# Creating a plot (to find the elbow [best k model])

accuracy_k <- numeric()

for (k in 1:50) {
  knn_pred <- knn(train_x, test_x, cl = train_y, k = k)
  acc <- mean(knn_pred == test_y)
  accuracy_k[k] <- acc
}

# Plot the elbow graph
plot(1:50, accuracy_k, type = "b", pch = 19, col = "blue",
     xlab = "Number of Neighbors (k)",
     ylab = "Accuracy",
     main = "Elbow Method for KNN – Accuracy vs. K")



