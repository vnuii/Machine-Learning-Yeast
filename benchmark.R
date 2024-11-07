# This script performs a benchmark analysis on multiple datasets using various machine learning classifiers.
#
# Set the seed for reproducibility and define the number of cores for parallel processing.
#
# List all files in the specified directory, remove the file extension, and exclude "Glucose".
#
# Create a cluster for parallel processing.
#
# Loop through each carbon source name:
# - Read the corresponding data file.
# - Set the row names to the assembly_ID column.
# - Clean the column names.
# - Convert the target variable to a binary factor ("growth" or "no_growth").
# - Create a classification task.
# - Define multiple learners (classifiers) with probability prediction.
# - Create a benchmark design with repeated cross-validation.
# - Execute the benchmark in parallel.
# - Save the benchmark result to a file.
library(mlr3verse)
library(data.table)
library(dplyr)
library(parallel)
set.seed(0611)
cores <- 60

carbon_names <- list.files("/share/ME-58T/y1000_metabolism/data4torch/OGcount_carbon") %>%
  gsub("_OGcount.tsv", "", .) %>%
  .[. != "Glucose"]

cl <- makeCluster(cores)

for (i in carbon_names) {
  data_dt <- fread(paste0("/share/ME-58T/y1000_metabolism/data4torch/OGcount_carbon/", i, "_OGcount.tsv"))
  rownames(data_dt) <- data_dt$assembly_ID
  i <- make.names(i)
  data_dt <- data_dt[, -1]
  colnames(data_dt) <- make.names(colnames(data_dt), unique = TRUE)
  data_dt[[i]] <- ifelse(data_dt[[i]] == 1, "growth", "no_growth")
  tsk_tmp <- as_task_classif(data_dt, target = i, positive = "growth")
  lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
  lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
  lrn_rpart <- lrn("classif.rpart", predict_type = "prob") # 分类的决策树
  lrn_svm <- lrn("classif.svm", predict_type = "prob")
  lrn_logreg <- lrn("classif.log_reg", predict_type = "prob")
  lrn_kknn <- lrn("classif.kknn", predict_type = "prob") # k-Nearest Neighbors
  design <- benchmark_grid(
    tasks = tsk_tmp,
    learners = list(
      lrn_ranger,
      lrn_xgboost,
      lrn_rpart,
      lrn_svm,
      lrn_logreg
    ),
    resamplings = rsmp("repeated_cv", folds = 10, repeats = 5)
  )
  future::plan("cluster", workers = cl)
  bmr <- benchmark(design)
  saveRDS(bmr, file = paste0("/share/ME-58T/y1000_metabolism/benchmark_10repeats/bmr_", i, ".rds"))
}

stopCluster(cl)
