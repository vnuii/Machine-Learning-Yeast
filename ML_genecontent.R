# This script performs machine learning on yeast gene content data using the mlr3 package.
# It reads in labeled data, processes it, and trains classification models to predict gene gain and loss events.
# Main Process:
# 1. For each orthologous group (OG) in the uncertain_MPPA_dt data table:
#    a. Filter out rows where the OG value is 0.5.
#    b. Identify parent nodes associated with gene gain (OG == 0) and gene loss (OG == 1).
#    c. Create data tables for gain and loss events, with labels for the first and second offspring nodes.
#    d. Train a classification model using the ranger algorithm for both gain and loss events.
#    e. Perform hyperparameter tuning using random search.
#    f. Perform cross-validation and save the resulting models to disk.
library(mlr3verse)
library(data.table)
library(dplyr)
library(parallel)
library(ape)
set.seed(0611)


label_dt <- fread("/share/ME-58T/y1000_metabolism/pastml/ML_label_dt.tsv") %>%
  .[node != "root" & parent_node != "root"]
uncertain_MPPA_dt <- fread("/share/ME-58T/y1000_metabolism/pastml/uncertain_data/uncertain_MPPA.tsv")
tree <- read.tree("/share/ME-58T/y1000_metabolism/pastml/yeast_1154.txt")

get_label <- function(node_, OG, idx) {
  offspring_node <- label_dt[parent_node == node_]$node[idx]
  label <- uncertain_MPPA_dt[node == offspring_node, .SD, .SDcols = OG] %>%
    unlist() %>%
    unname()
  return(label)
}

for (OG in colnames(uncertain_MPPA_dt[, -1])) {
  uncertain_MPPA_dt_tmp <- uncertain_MPPA_dt[, .SD, .SDcols = c("node", OG)] %>%
    .[get(OG) != 0.5]
  gain_parent <- uncertain_MPPA_dt_tmp[node %in% tree$node.label & get(OG) == 0]$node
  loss_parent <- uncertain_MPPA_dt_tmp[node %in% tree$node.label & get(OG) == 1]$node
  ML_dt_gain_1 <- uncertain_MPPA_dt[node %in% gain_parent]
  ML_dt_gain_1$label <- lapply(ML_dt_gain_1$node, get_label, OG = OG, idx = 1)
  ML_dt_gain_1$node <- paste0(ML_dt_gain_1$node, "|1")
  ML_dt_gain_2 <- uncertain_MPPA_dt[node %in% gain_parent]
  ML_dt_gain_2$label <- lapply(ML_dt_gain_2$node, get_label, OG = OG, idx = 2)
  ML_dt_gain_2$node <- paste0(ML_dt_gain_2$node, "|2")
  ML_dt_gain <- rbind(ML_dt_gain_1, ML_dt_gain_2)

  ML_dt_loss_1 <- uncertain_MPPA_dt[node %in% loss_parent]
  ML_dt_loss_1$label <- lapply(ML_dt_loss_1$node, get_label, OG = OG, idx = 1)
  ML_dt_loss_1$node <- paste0(ML_dt_loss_1$node, "|1")
  ML_dt_loss_2 <- uncertain_MPPA_dt[node %in% loss_parent]
  ML_dt_loss_2$label <- lapply(ML_dt_loss_2$node, get_label, OG = OG, idx = 2)
  ML_dt_loss_2$node <- paste0(ML_dt_loss_2$node, "|2")
  ML_dt_loss <- rbind(ML_dt_loss_1, ML_dt_loss_2)

  lts_ranger <- lts("classif.ranger.default") # 默认的调节范围
  lrn_ranger <- lrn("classif.ranger", predict_type = "prob", num.threads = 14)
  lrn_ranger$param_set$set_values(.values = lts_ranger$values)

  # gain
  rownames(ML_dt_gain) <- ML_dt_gain$node
  ML_dt_gain <- ML_dt_gain[, -1]
  ML_dt_gain[["label"]] <- ifelse(ML_dt_gain[["label"]] == 1, "one", "zero")
  tsk_gain_tmp <- as_task_classif(ML_dt_gain, target = "label", positive = "one")
  ## tuning
  lts_ranger_gain_at <- auto_tuner(
    tuner = tnr("random_search"),
    learner = lrn_ranger,
    resampling = rsmp("holdout", ratio = 0.9),
    measure = msr("classif.auc"),
    term_evals = 500
  )
  future::plan("multisession", workers = 5)
  rr <- resample(
    task = tsk_gain_tmp,
    learner = lts_ranger_gain_at,
    resampling = rsmp("cv", folds = 5),
    store_models = TRUE,
  )
  saveRDS(rr, file = paste0("/share/ME-58T/y1000_metabolism/ML_model_genecontent/ML_", OG, "_gain.rds"))

  # loss
  rownames(ML_dt_loss) <- ML_dt_loss$node
  ML_dt_loss <- ML_dt_loss[, -1]
  ML_dt_loss[["label"]] <- ifelse(ML_dt_loss[["label"]] == 1, "one", "zero")
  tsk_loss_tmp <- as_task_classif(ML_dt_loss, target = "label", positive = "one")
  ## tuning
  lts_ranger_loss_at <- auto_tuner(
    tuner = tnr("random_search"),
    learner = lrn_ranger,
    resampling = rsmp("holdout", ratio = 0.9, stratify = TRUE),
    measure = msr("classif.auc"),
    term_evals = 500
  )
  future::plan("multisession", workers = 5)
  rr <- resample(
    task = tsk_loss_tmp,
    learner = lts_ranger_loss_at,
    resampling = rsmp("cv", folds = 5, stratify = TRUE),
    store_models = TRUE,
  )
  saveRDS(rr, file = paste0("/share/ME-58T/y1000_metabolism/ML_model_genecontent/ML_", OG, "_loss.rds"))
}
