# This script performs Recursive Feature Elimination (RFE) on selected carbon source datasets.
# It uses the mlr3verse package for machine learning tasks and parallel processing to optimize performance.
# The script reads data files, preprocesses them, and applies RFE using a random forest classifier.
# The results are saved as RDS files, and the runtime for each optimization is printed.

# Set seed for reproducibility.
# Define the list of carbon sources to be analyzed.
# Loop through each selected carbon source:
#   - Read the data file.
#   - Preprocess the data (set row names, remove the first column, make column names syntactically valid).
#   - Create a classification task.
#   - Define the number of cores for parallel processing.
#   - Set up the RFE optimizer.
#   - Define the random forest learner with specific parameters.
#   - Create an instance for feature selection.
#   - Record the start time.
#   - Run the optimizer using parallel processing.
#   - Save the instance results as an RDS file.
#   - Record the end time.
#   - Calculate and print the runtime.
#   - Clean up the instance and perform garbage collection.
# Recursive Feature Elimination
library(mlr3verse)
library(data.table)
library(dplyr)
library(parallel)
library(future)
set.seed(0611)

carbon_names <- list.files("/share/ME-58T/y1000_metabolism/data4torch/OGcount_carbon") %>%
  gsub("_OGcount.tsv", "", .) %>%
  .[. != "Glucose"]

# selected_carbons <- carbon_names[13:17]
selected_carbons <- carbon_names[c(3, 11, 12)]

for (i in selected_carbons) {
  data_dt <- fread(paste0("/share/ME-58T/y1000_metabolism/data4torch/OGcount_carbon/", i, "_OGcount.tsv"))
  rownames(data_dt) <- data_dt$assembly_ID
  i <- make.names(i) # make.names() is a function that makes syntactically valid names out of character vectors.
  data_dt <- data_dt[, -1]
  colnames(data_dt) <- make.names(colnames(data_dt), unique = TRUE)
  data_dt[[i]] <- ifelse(data_dt[[i]] == 1, "growth", "no_growth")
  tsk_tmp <- as_task_classif(data_dt, target = i, positive = "growth")

  cores <- 24

  optimizer <- fs("rfe",
    n_features = 1,
    feature_number = 1
  )

  lrn_ranger <- lrn("classif.ranger",
    predict_type = "prob",
    importance = "impurity",
    num.threads = 8
  )

  instance <- fsi(
    task = tsk_tmp,
    learner = lrn_ranger,
    resampling = rsmp("cv", folds = 5),
    measures = msr("classif.auc"),
    terminator = trm("none")
  )

  start_time <- Sys.time()

  plan("multisession", workers = cores)

  optimizer$optimize(instance)
  saveRDS(instance, file = paste0("/share/ME-58T/y1000_metabolism/RFE_results/rfenormal_", i, ".rds"))

  end_time <- Sys.time()

  runtime <- end_time - start_time
  print(paste("Start time:", start_time))
  print(paste("End time:", end_time))
  print(paste("Optimization runtime:", i, "_", runtime))
  rm(instance)
  gc()
}

# grep "Optimization runtime:" /share/ME-58T/y1000_metabolism/log/RFEnormal_13to17_2024.07.19.log
