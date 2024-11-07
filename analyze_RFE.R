# This script performs feature selection and enrichment analysis on yeast metabolism data.
# It uses Recursive Feature Elimination (RFE) results to identify important features for different carbon sources,
# and then performs Gene Ontology (GO) and KEGG pathway enrichment analysis on these features.
# Additionally, it visualizes the gain and loss of gene families using PastML and plots the number of selected features
# and their corresponding AUC values for each carbon source.

# Main steps:
# 1. Load RFE results and extract the best features for each carbon source.
# 2. Perform GO enrichment analysis using the selected features.
# 3. Perform KEGG pathway enrichment analysis using the selected features.
# 4. Visualize the gain and loss of gene families using PastML.
# 5. Plot the number of selected features and their corresponding AUC values for each carbon source.

library(data.table)
library(dplyr)

files <- list.files("/share/ME-58T/y1000_metabolism/RFE_results/need_table", full.names = TRUE)

feature_dt <- data.table(carbon = character(), feature = character())
for (i in files) {
  carbon <- basename(i) %>%
    gsub("rfenormal_rfenormal_", "", .) %>%
    gsub(".tsv", "", .)
  dt_tmp <- fread(i)
  OG_max <- dt_tmp[which.max(dt_tmp$classif.auc), features]
  OG_max_c <- dt_tmp[which.max(dt_tmp$classif.auc), features] %>%
    stringr::str_split(., "\\|") %>%
    unlist()
  assign(carbon, OG_max_c)
  feature_dt_tmp <- data.table(carbon = carbon, feature = OG_max)
  feature_dt <- rbind(feature_dt, feature_dt_tmp)
}

colnames(dt_tmp)
dt_tmp[which.max(dt_tmp$classif.auc), features] %>%
  stringr::str_split(., "\\|") %>%
  unlist()

Reduce(intersect, list(lapply(feature_dt$feature, function(x) stringr::str_split(x, "\\|"))))

unlist(OG_list)

a <- stringr::str_split(feature_dt$feature, "\\|")
Reduce(intersect, list(Citrate, Sucrose, Xylose))
Reduce(intersect, list(myo.Inositol, Raffinose, Rhamnose))
intersect(myo.Inositol, Citrate)
unique(unlist(a)) # 953

## GO enrichment
library(clusterProfiler)
OG2slim <- fread("/share/ME-58T/1kp/a_main_graph/a_LTT/OG_GOslim.tsv")
slim2des <- fread("/share/ME-58T/1kp/a_main_graph/a_LTT/yeast_slim_GO_des_altID.tsv")
bg <- fread("/share/ME-58T/y1000_metabolism/data/yeast_0.1_Orthogroups.GeneCount.tsv") %>% pull(Orthogroup)
OG2slim_Order <- OG2slim[Orthogroup %in% bg]
slim2des_Order <- slim2des[id %in% OG2slim_Order$GO_slim]
i <- feature_dt$carbon[3]
for (i in feature_dt$carbon) {
  key_OG <- feature_dt[carbon == i, feature] %>%
    stringr::str_split(., "\\|") %>%
    unlist()
  enrich_out <- enricher(
    gene = key_OG,
    TERM2GENE = OG2slim_Order[, c("GO_slim", "Orthogroup")],
    TERM2NAME = slim2des_Order[, c("id", "name")],
    pAdjustMethod = "BH",
    minGSSize = 1,
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2
  )
  enrich_out_dt <- merge(enrich_out, slim2des_Order[, 1:3], by.x = "ID", by.y = "id", sort = F)
  fwrite(enrich_out_dt, paste0("/share/ME-58T/y1000_metabolism/RFE_enrich_slim/enrich_", i, ".tsv"), sep = "\t")
}

### KEGG enrichment
OG2Ghost <- fread("/share/ME-58T/y1000_metabolism/OG2Ghost.tsv", select = c("Orthogroup", "kos_unique"))
bg <- fread("/share/ME-58T/y1000_metabolism/data/yeast_0.1_Orthogroups.GeneCount.tsv") %>% pull(Orthogroup)
KO2des <- fread("/share/ME-58T/1kp/final_enrich/data/KEGG/only_Sacc_brite_path_table.tsv")

OG2Ghost_Order <- OG2Ghost[Orthogroup %in% bg] %>%
  filter(kos_unique != "")
OG2Ghost_Order_final <- data.table(Orthogroup = character(), kos = character())

for (i in OG2Ghost_Order$Orthogroup) {
  kos <- OG2Ghost_Order[Orthogroup == i, kos_unique] %>%
    stringr::str_split(., ",") %>%
    unlist()
  OG2Ghost_Order_tmp <- data.table(Orthogroup = rep(i, length(kos)), kos = kos)
  OG2Ghost_Order_final <- rbind(OG2Ghost_Order_final, OG2Ghost_Order_tmp)
}
KO2des_Order <- KO2des[KO %in% unique(OG2Ghost_Order_final$kos)]
i <- feature_dt$carbon[3]
for (i in feature_dt$carbon) {
  key_OG <- feature_dt[carbon == i, feature] %>%
    stringr::str_split(., "\\|") %>%
    unlist()
  enrich_out <- enricher(
    gene = key_OG,
    TERM2GENE = OG2slim_Order[, c("GO_slim", "Orthogroup")],
    TERM2NAME = slim2des_Order[, c("id", "name")],
    pAdjustMethod = "BH",
    minGSSize = 1,
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2
  )
  enrich_out_dt <- merge(enrich_out, slim2des_Order[, 1:3], by.x = "ID", by.y = "id", sort = F)
  fwrite(enrich_out_dt, paste0("/share/ME-58T/y1000_metabolism/RFE_enrich_slim/enrich_", i, ".tsv"), sep = "\t")
}



## PastML
library(data.table)
library(ggplot2)
library(dplyr)
library(ggside)
library(viridis)
library(ggpointdensity)

gain_loss_summary_dt_downpass <- fread("/share/ME-58T/y1000_metabolism/pastml/ML_gain_loss_summary_downpass.tsv") %>% filter(Orthogroup %in% unique(unlist(a)))
gain_loss_summary_dt_MPPA <- fread("/share/ME-58T/y1000_metabolism/pastml/ML_gain_loss_summary_MPPA.tsv") %>% filter(Orthogroup %in% unique(unlist(a)))

p_MPPA <- ggplot(gain_loss_summary_dt_downpass, aes(x = gain_num, y = loss_num)) +
  # geom_point(size = 2, alpha = 0.5, color = "grey") +
  geom_pointdensity(size = 2, adjust = 4) +
  scale_color_viridis() +
  # geom_hex(bins = 200) +
  geom_xsidehistogram() +
  geom_ysidehistogram() +
  scale_ysidex_continuous(guide = guide_axis(angle = 90)) +
  theme_classic(base_size = 16, base_family = "Arial") +
  labs(
    x = "Gain model instance number (0 -> 1 or 0 -> 0)",
    y = "Loss model instance number (1 -> 0 or 1 -> 1)",
    title = "Gene family gain or loss model instance number (MPPA)",
    color = "Density"
  ) +
  theme(plot.title = element_text(hjust = 0.5))

p_downpass <- ggplot(gain_loss_summary_dt_MPPA, aes(x = gain_num, y = loss_num)) +
  # geom_point(size = 2, alpha = 0.5, color = "grey") +
  geom_pointdensity(size = 2, adjust = 4) +
  scale_color_viridis() +
  # geom_hex(bins = 200) +
  geom_xsidehistogram() +
  geom_ysidehistogram() +
  scale_ysidex_continuous(guide = guide_axis(angle = 90)) +
  theme_classic(base_size = 16, base_family = "Arial") +
  labs(
    x = "Gain model instance number (0 -> 1 or 0 -> 0)",
    y = "Loss model instance number (1 -> 0 or 1 -> 1)",
    title = "Gene family gain or loss model instance number (DOWNPASS)",
    color = "Density"
  ) +
  theme(plot.title = element_text(hjust = 0.5))

cowplot::plot_grid(p_MPPA, p_downpass, nrow = 1, ncol = 2)



# plot the number of selected features and their corresponding AUC values for each carbon source
a <- fread("/share/ME-58T/y1000_metabolism/RFE_results/need_table/rfenormal_rfenormal_Cellobiose.tsv")
files <- list.files("/share/ME-58T/y1000_metabolism/RFE_results/need_table", full.names = TRUE)
plot_dt <- data.table(carbon = character(), OG_num = numeric(), AUC = numeric())
for (i in files) {
  carbon <- basename(i) %>%
    gsub("rfenormal_rfenormal_", "", .) %>%
    gsub(".tsv", "", .)
  dt_tmp <- fread(i)
  OG_num <- dt_tmp[which.max(dt_tmp$classif.auc), ]$features_number
  AUC <- dt_tmp[which.max(dt_tmp$classif.auc)]$classif.auc
  plot_dt_tmp <- data.table(carbon = carbon, OG_num = OG_num, AUC = AUC)
  plot_dt <- rbind(plot_dt, plot_dt_tmp)
}

carbon_color <- c(
  "Cellobiose" = "#5F4690FF", "Citrate" = "#1D6996FF", "D.Glucosamine" = "#38A6A5FF", "DL.Lactate" = "#0F8554FF",
  "Fructose" = "#73AF48FF", "Galactose" = "#EDAD08FF", "Glucose" = "#E17C05FF", "Glycerol" = "#CC503EFF",
  "L.Arabinose" = "#94346EFF", "L.Sorbose" = "#6F4070FF", "Lactose" = "#994E95FF", "Maltose" = "#666666FF",
  "Mannose" = "#019875FF", "myo.Inositol" = "#99B898FF", "Raffinose" = "#FECEA8FF", "Rhamnose" = "#FF847CFF",
  "Sucrose" = "#E84A5FFF", "Xylose" = "#C0392BFF"
)

library(ggplot2)
ggplot(plot_dt) +
  geom_bar(aes(x = carbon, y = OG_num, fill = carbon), stat = "identity") +
  geom_line(aes(x = carbon, y = AUC * 225, group = 1), color = "black", size = 0.8) +
  geom_hline(yintercept = 191.25, linetype = "dashed", color = "red", size = 0.5) +
  scale_fill_manual(values = carbon_color, guide = "none") +
  scale_y_continuous(
    name = "Gene family number",
    sec.axis = sec_axis(~ . / 225, name = "AUC")
  ) +
  theme_classic(base_size = 18, base_family = "Arial") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = NULL)
