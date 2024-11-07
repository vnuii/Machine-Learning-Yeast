# 根据酵母是否能够利用碳源或者生长速率对18个碳源进行PCA
library(data.table)
library(dplyr)
library(ggplot2)
library(ggrepel)

a <- fread("/share/ME-58T/y1000_metabolism/data/carbon18_science_853_FB.tsv")
carbon_names <- colnames(a)[-c(1:5)]
yeast_carbon_dt <- fread("/share/ME-58T/y1000_metabolism/data/metabolism_science_assemblyID_deleted.tsv",
  select = c("assembly_ID", "Order", "Carbon Breadth", "Carbon Class", carbon_names)
)
yeast_carbon_dt_nona <- na.omit(yeast_carbon_dt)
yeast_carbon_dt_nona_01 <- yeast_carbon_dt_nona
yeast_carbon_dt_nona_01[, (5:22) := lapply(.SD, function(x) ifelse(x > 0, 1, 0)), .SDcols = 5:22]
yeast_carbon_dt_nona_01_t <- t(yeast_carbon_dt_nona_01[, 5:22])

yeast_carbon_dt_nona_t <- t(yeast_carbon_dt_nona[, 5:22])
pca_rate <- prcomp(yeast_carbon_dt_nona_t, scale = TRUE)
plot_dt_rate <- data.table(carbon_names = rownames(yeast_carbon_dt_nona_t), PC1 = pca_rate$x[, 1], PC2 = pca_rate$x[, 2])
summary(pca_rate)

ggplot(plot_dt_rate, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = carbon_names), size = 3) +
  theme_classic(base_size = 18, base_family = "Arial") +
  geom_text_repel(aes(label = carbon_names), size = 4, min.segment.length = 0, seed = 42, box.padding = 0.5, max.overlaps = 15) +
  labs(x = "PC1 (60.44%)", y = "PC2 (7.84%)", title = "PCA of yeast carbon metabolism (rate, scale = TRUE)") +
  scale_color_manual(values = c(
    "#5F4690FF", "#1D6996FF", "#38A6A5FF", "#0F8554FF",
    "#73AF48FF", "#EDAD08FF", "#E17C05FF", "#CC503EFF",
    "#94346EFF", "#6F4070FF", "#994E95FF", "#666666FF",
    "#019875FF", "#99B898FF", "#FECEA8FF", "#FF847CFF",
    "#E84A5FFF", "#C0392BFF"
  )) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

pca_01 <- prcomp(yeast_carbon_dt_nona_01_t)
summary(pca_01)
plot_dt <- data.table(carbon_names = rownames(yeast_carbon_dt_nona_01_t), PC1 = pca_01$x[, 1], PC2 = pca_01$x[, 2])

plot_pca <- ggplot(plot_dt, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = carbon_names), size = 5) +
  scale_color_manual(values = c(
    "#5F4690FF", "#1D6996FF", "#38A6A5FF", "#0F8554FF",
    "#73AF48FF", "#EDAD08FF", "#E17C05FF", "#CC503EFF",
    "#94346EFF", "#6F4070FF", "#994E95FF", "#666666FF",
    "#019875FF", "#99B898FF", "#FECEA8FF", "#FF847CFF",
    "#E84A5FFF", "#C0392BFF"
  )) +
  theme_classic(base_size = 18, base_family = "Arial") +
  geom_text_repel(aes(label = carbon_names),
    size = 4,
    min.segment.length = 0, seed = 42, box.padding = 0.5
  ) +
  labs(
    x = "PC1 (42.75%)", y = "PC2 (8.17%)", color = "Carbon source",
    title = "PCA of yeast carbon metabolism (0/1)"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  ) +
  geom_vline(xintercept = -10, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red")

# yeast_carbon_dt_nona_t <- t(yeast_carbon_dt_nona[, 5:22])
# pca_rate <- prcomp(yeast_carbon_dt_nona_t, scale = FALSE)
# summary(pca_rate)
# plot_dt_rate <- data.table(carbon_names = rownames(yeast_carbon_dt_nona_t), PC1 = pca_rate$x[, 1], PC2 = pca_rate$x[, 2])

# ggplot(plot_dt_rate, aes(x = PC1, y = PC2)) +
#   geom_point(aes(color = carbon_names), size = 3) +
#   theme_classic(base_size = 18, base_family = "Arial") +
#   geom_text_repel(aes(label = carbon_names), size = 4, min.segment.length = 0, seed = 42, box.padding = 0.5) +
#   labs(x = "PC1 (60.44%)", y = "PC2 (7.84%)", title = "PCA of yeast carbon metabolism (rate)")

carbon_count_dt <- data.table(carbon = colnames(yeast_carbon_dt_nona_01[, 5:22]), count = colSums(yeast_carbon_dt_nona_01[, 5:22]))
carbon_count_dt$type <- "growth"
tmp <- data.table(carbon = colnames(yeast_carbon_dt_nona_01[, 5:22]), count = 736 - colSums(yeast_carbon_dt_nona_01[, 5:22]))
tmp$type <- "no_growth"

all <- rbind(carbon_count_dt, tmp)
all$carbon <- factor(all$carbon, levels = all[type == "growth"]$carbon[order(all[type == "growth"]$count, decreasing = TRUE)])

# Cellobiose	Citrate	D-Glucosamine	DL-Lactate	Fructose	Galactose	Glucose	Glycerol	L-Arabinose	L-Sorbose	Lactose	Maltose	Mannose	myo-Inositol	Raffinose	Rhamnose	Sucrose	Xylose
carbon_color <- c(
  "Cellobiose" = "#5F4690FF", "Citrate" = "#1D6996FF", "D-Glucosamine" = "#38A6A5FF", "DL-Lactate" = "#0F8554FF",
  "Fructose" = "#73AF48FF", "Galactose" = "#EDAD08FF", "Glucose" = "#E17C05FF", "Glycerol" = "#CC503EFF",
  "L-Arabinose" = "#94346EFF", "L-Sorbose" = "#6F4070FF", "Lactose" = "#994E95FF", "Maltose" = "#666666FF",
  "Mannose" = "#019875FF", "myo-Inositol" = "#99B898FF", "Raffinose" = "#FECEA8FF", "Rhamnose" = "#FF847CFF",
  "Sucrose" = "#E84A5FFF", "Xylose" = "#C0392BFF"
)

plot_count <- ggplot(all[type == "growth"], aes(x = carbon, y = count, fill = carbon)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = carbon_color) +
  theme_classic(base_size = 18, base_family = "Arial") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  ) +
  labs(x = "Carbon source", y = "Number of strains", title = "Carbon source utilization of yeast strains")

cowplot::plot_grid(plot_pca, plot_count, nrow = 1, labels = "AUTO", align = "hv")

count_dt <- data.table(carbon = character(), all_number = numeric(), growth_number = numeric(), uti_rate = numeric())
for (i in carbon_names) {
  tmp_dt <- yeast_carbon_dt %>% select(all_of(c("assembly_ID", "Order", "Carbon Breadth", "Carbon Class", i)))
  all_number <- sum(!is.na(tmp_dt[[i]]))
  growth_number <- sum(tmp_dt[[i]] > 0, na.rm = TRUE)
  count_dt_tmp <- data.table(carbon = i, all_number = all_number, growth_number = growth_number, uti_rate = growth_number / all_number)
  count_dt <- rbind(count_dt, count_dt_tmp)
}

ggplot(count_dt, aes(x = carbon, y = uti_rate)) +
  geom_bar(stat = "identity", fill = "grey") +
  geom_text(aes(label = scales::percent(uti_rate)), vjust = -0.5, size = 3) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "red") +
  theme_classic(base_size = 18, base_family = "Arial") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  ) +
  labs(x = "Carbon source", y = "Utilization rate", title = "Carbon source utilization rate of yeast strains")
