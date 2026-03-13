#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════
# RQ3: Input Complexity vs. Quantization Robustness
# ══════════════════════════════════════════════════════════════════
#
# Input:  rq3_merged.csv (from rq3_prepare.py)
#         Columns: task_id, benchmark, entropy, length_words, bucket,
#                  pass_fp, pass_q, degraded, helped, model, technique
#
# Output: rq3_mcnemar_results.csv
#         rq3_stratified_results.csv
#         rq3_correlation_results.csv
#         rq3_bucket_correlation_results.csv
#         rq3_combined_table.csv
#         rq3_combined_table.tex
#
# Usage:  Rscript rq3_analysis.R <merged_csv> <output_dir>

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript rq3_analysis.R <rq3_merged.csv> <output_dir>")
}

merged_csv <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(merged_csv, stringsAsFactors = FALSE)
cat("Loaded", nrow(df), "rows\n")
cat("Models:", paste(unique(df$model), collapse = ", "), "\n")
cat("Techniques:", paste(unique(df$technique), collapse = ", "), "\n")
cat("Benchmarks:", paste(unique(df$benchmark), collapse = ", "), "\n\n")


# ══════════════════════════════════════════════════════════════════
# 1. McNEMAR'S TEST
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  1. McNemar's Test: FP vs each quantization technique\n")
cat("================================================================\n\n")

mcnemar_results <- data.frame()

for (mod in unique(df$model)) {
  for (tech in unique(df$technique)) {
    sub <- df %>% filter(model == mod, technique == tech)
    if (nrow(sub) == 0) next

    a <- sum(sub$pass_fp == 1 & sub$pass_q == 1)
    b <- sum(sub$pass_fp == 1 & sub$pass_q == 0)
    c <- sum(sub$pass_fp == 0 & sub$pass_q == 1)
    d <- sum(sub$pass_fp == 0 & sub$pass_q == 0)
    n <- a + b + c + d
    pass1_fp <- (a + b) / n
    pass1_q  <- (a + c) / n

    if (b + c == 0) {
      p_val <- NA; statistic <- NA
    } else {
      mat <- matrix(c(a, b, c, d), nrow = 2)
      test <- mcnemar.test(mat)
      p_val <- test$p.value; statistic <- test$statistic
    }

    row <- data.frame(model = mod, technique = tech, n = n,
                      both_pass = a, fp_only = b, q_only = c, both_fail = d,
                      pass1_fp = round(pass1_fp, 4), pass1_q = round(pass1_q, 4),
                      delta = round(pass1_q - pass1_fp, 4),
                      statistic = round(statistic, 4), p_value = round(p_val, 6),
                      significant = ifelse(is.na(p_val), FALSE, p_val < 0.05),
                      stringsAsFactors = FALSE)
    mcnemar_results <- rbind(mcnemar_results, row)

    sig <- ifelse(is.na(p_val), "N/A",
                  ifelse(p_val < 0.001, "***",
                         ifelse(p_val < 0.01, "**",
                                ifelse(p_val < 0.05, "*", "ns"))))
    cat(sprintf("  %s / %s: FP=%.1f%% Q=%.1f%% | deg=%d help=%d | p=%s [%s]\n",
                mod, tech, pass1_fp*100, pass1_q*100, b, c,
                ifelse(is.na(p_val), "N/A", sprintf("%.4f", p_val)), sig))
  }
}

write.csv(mcnemar_results, file.path(output_dir, "rq3_mcnemar_results.csv"), row.names = FALSE)
cat("\n  Saved: rq3_mcnemar_results.csv\n\n")


# ══════════════════════════════════════════════════════════════════
# 2. STRATIFIED ANALYSIS
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  2. Stratified Analysis: Bucket x Benchmark\n")
cat("================================================================\n\n")

stratified_results <- data.frame()

for (mod in unique(df$model)) {
  for (tech in unique(df$technique)) {
    for (buck in c("High", "Low")) {
      for (bench in unique(df$benchmark)) {
        sub <- df %>% filter(model == mod, technique == tech,
                             bucket == buck, benchmark == bench)
        if (nrow(sub) == 0) next
        n <- nrow(sub)
        row <- data.frame(
          model = mod, technique = tech, bucket = buck, benchmark = bench,
          n = n,
          pass1_fp = round(sum(sub$pass_fp) / n, 4),
          pass1_q = round(sum(sub$pass_q) / n, 4),
          degraded = sum(sub$degraded),
          helped = sum(sub$helped),
          degradation_rate = round(sum(sub$degraded) / n, 4),
          delta_pass1 = round((sum(sub$pass_q) - sum(sub$pass_fp)) / n, 4),
          stringsAsFactors = FALSE)
        stratified_results <- rbind(stratified_results, row)
      }
    }
  }
}

write.csv(stratified_results, file.path(output_dir, "rq3_stratified_results.csv"), row.names = FALSE)
cat("  Saved: rq3_stratified_results.csv\n\n")


# ══════════════════════════════════════════════════════════════════
# 3. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  3. Correlation: Complexity vs. Degradation\n")
cat("================================================================\n\n")

correlation_results <- data.frame()

for (mod in unique(df$model)) {
  for (tech in unique(df$technique)) {
    sub <- df %>% filter(model == mod, technique == tech, pass_fp == 1)
    n_total <- nrow(sub)
    n_degraded <- sum(sub$degraded)

    if (n_total < 5 || n_degraded == 0 || n_degraded == n_total) {
      cat(sprintf("  %s / %s: Skipped (n=%d, deg=%d)\n", mod, tech, n_total, n_degraded))
      next
    }

    cat(sprintf("  %s / %s: %d FP-correct, %d degraded (%.1f%%)\n",
                mod, tech, n_total, n_degraded, n_degraded/n_total*100))

    for (feature in c("entropy", "length_words")) {
      cor_test <- cor.test(sub$degraded, sub[[feature]], method = "pearson")
      deg_vals <- sub[[feature]][sub$degraded == 1]
      kept_vals <- sub[[feature]][sub$degraded == 0]
      mw_test <- wilcox.test(deg_vals, kept_vals, alternative = "two.sided", exact = FALSE)

      sig_r <- ifelse(cor_test$p.value < 0.05, "*", "ns")
      sig_mw <- ifelse(mw_test$p.value < 0.05, "*", "ns")
      cat(sprintf("    %s: r=%+.4f (p=%.4f)[%s] MW p=%.4f [%s]\n",
                  feature, cor_test$estimate, cor_test$p.value, sig_r,
                  mw_test$p.value, sig_mw))

      row <- data.frame(
        model = mod, technique = tech, feature = feature,
        n_fp_correct = n_total, n_degraded = n_degraded,
        r = round(cor_test$estimate, 4), r_p_value = round(cor_test$p.value, 6),
        r_significant = cor_test$p.value < 0.05,
        mw_statistic = round(mw_test$statistic, 2),
        mw_p_value = round(mw_test$p.value, 6),
        mw_significant = mw_test$p.value < 0.05,
        mean_degraded = round(mean(deg_vals), 2),
        mean_kept = round(mean(kept_vals), 2),
        stringsAsFactors = FALSE)
      correlation_results <- rbind(correlation_results, row)
    }
    cat("\n")
  }
}

write.csv(correlation_results, file.path(output_dir, "rq3_correlation_results.csv"), row.names = FALSE)
cat("  Saved: rq3_correlation_results.csv\n\n")


# ══════════════════════════════════════════════════════════════════
# 4. PER-BUCKET CORRELATION
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  4. Per-Bucket Correlation\n")
cat("================================================================\n\n")

bucket_corr <- data.frame()

for (mod in unique(df$model)) {
  for (tech in unique(df$technique)) {
    for (buck in c("High", "Low")) {
      sub <- df %>% filter(model == mod, technique == tech, bucket == buck, pass_fp == 1)
      n_total <- nrow(sub)
      n_degraded <- sum(sub$degraded)
      if (n_total < 5 || n_degraded == 0 || n_degraded == n_total) next

      for (feature in c("entropy", "length_words")) {
        cor_test <- cor.test(sub$degraded, sub[[feature]], method = "pearson")
        deg_vals <- sub[[feature]][sub$degraded == 1]
        kept_vals <- sub[[feature]][sub$degraded == 0]
        mw_test <- wilcox.test(deg_vals, kept_vals, alternative = "two.sided", exact = FALSE)

        sig <- ifelse(cor_test$p.value < 0.05, "*", "ns")
        cat(sprintf("  %s/%s [%s] %s: r=%+.4f p=%.4f [%s]\n",
                    mod, tech, buck, feature, cor_test$estimate, cor_test$p.value, sig))

        row <- data.frame(
          model = mod, technique = tech, bucket = buck, feature = feature,
          n_fp_correct = n_total, n_degraded = n_degraded,
          r = round(cor_test$estimate, 4), r_p_value = round(cor_test$p.value, 6),
          r_significant = cor_test$p.value < 0.05,
          mw_statistic = round(mw_test$statistic, 2),
          mw_p_value = round(mw_test$p.value, 6),
          mw_significant = mw_test$p.value < 0.05,
          stringsAsFactors = FALSE)
        bucket_corr <- rbind(bucket_corr, row)
      }
    }
  }
}

write.csv(bucket_corr, file.path(output_dir, "rq3_bucket_correlation_results.csv"), row.names = FALSE)
cat("\n  Saved: rq3_bucket_correlation_results.csv\n\n")


# ══════════════════════════════════════════════════════════════════
# 5. COMBINED LaTeX TABLE
#    Layout: Model | Bucket | Technique | McEval | CoderEval | BigCodeBench | r_ent | p_ent | r_len | p_len
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  5. Generating LaTeX Table\n")
cat("================================================================\n\n")

# Pivot: benchmark degradation rates as columns
strat_wide <- stratified_results %>%
  select(model, technique, bucket, benchmark, degradation_rate) %>%
  pivot_wider(names_from = benchmark, values_from = degradation_rate, names_prefix = "deg_")

# Merge correlation results
corr_ent <- correlation_results %>%
  filter(feature == "entropy") %>%
  select(model, technique, r, r_p_value) %>%
  rename(ent_r = r, ent_p = r_p_value)

corr_len <- correlation_results %>%
  filter(feature == "length_words") %>%
  select(model, technique, r, r_p_value) %>%
  rename(len_r = r, len_p = r_p_value)

combined <- strat_wide %>%
  left_join(corr_ent, by = c("model", "technique")) %>%
  left_join(corr_len, by = c("model", "technique"))

write.csv(combined, file.path(output_dir, "rq3_combined_table.csv"), row.names = FALSE)

# Formatters
fmt_p <- function(p) {
  if (is.na(p)) return("--")
  stars <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  if (p < 0.001) return(paste0("$<$0.001", stars))
  return(paste0(sprintf("%.3f", p), stars))
}

fmt_r <- function(r) {
  if (is.na(r)) return("--")
  return(sprintf("$%+.3f$", r))
}

fmt_deg <- function(d) {
  if (is.na(d)) return("--")
  return(sprintf("%.1f", d * 100))
}

# Ordering
tech_order <- c("AWQ", "GPTQ", "GGUF", "BnB", "AQLM", "QuIP#")
model_order <- unique(combined$model)
bench_cols <- intersect(c("deg_McEval", "deg_CoderEval", "deg_BigCodeBench"), colnames(combined))
bench_labels <- gsub("deg_", "", bench_cols)
n_bench <- length(bench_cols)

# Build LaTeX lines
L <- c()
L <- c(L, "\\begin{table*}[t]")
L <- c(L, "    \\centering")
L <- c(L, paste0("    \\caption{RQ$_3$ results: per-benchmark degradation rates (\\%) within entropy buckets and correlation between input complexity and quantization-induced degradation. ",
                  "$r$ = point-biserial correlation coefficient. Significance: *$p<0.05$, **$p<0.01$, ***$p<0.001$.}"))
L <- c(L, "    \\label{tab:rq3-combined}")
L <- c(L, "    \\footnotesize")

# Column spec
col_spec <- paste0("ll l ", paste(rep("r", n_bench), collapse = ""), " c rr c rr")
L <- c(L, paste0("    \\begin{tabular}{", col_spec, "}"))
L <- c(L, "        \\toprule")

# Header
bench_mc <- paste0("\\multicolumn{", n_bench, "}{c}{\\textbf{Degradation Rate (\\%)}}")
L <- c(L, paste0("        & & & ", bench_mc,
                  " & & \\multicolumn{2}{c}{\\textbf{Entropy}}",
                  " & & \\multicolumn{2}{c}{\\textbf{Length}} \\\\"))

b_start <- 4; b_end <- 3 + n_bench
e_start <- b_end + 2; e_end <- e_start + 1
l_start <- e_end + 2; l_end <- l_start + 1
L <- c(L, paste0("        \\cmidrule(lr){", b_start, "-", b_end, "}",
                  " \\cmidrule(lr){", e_start, "-", e_end, "}",
                  " \\cmidrule(lr){", l_start, "-", l_end, "}"))

bench_hdrs <- paste(paste0("\\textbf{", bench_labels, "}"), collapse = " & ")
L <- c(L, paste0("        \\textbf{Model} & \\textbf{Bucket} & \\textbf{Technique} & ",
                  bench_hdrs,
                  " & & \\textbf{$r$} & \\textbf{$p$}",
                  " & & \\textbf{$r$} & \\textbf{$p$} \\\\"))
L <- c(L, "        \\midrule")

# Data rows
for (m_idx in seq_along(model_order)) {
  mod <- model_order[m_idx]
  mod_data <- combined %>% filter(model == mod)

  # Count rows for multirow
  mod_rows <- 0
  for (buck in c("High", "Low")) {
    bd <- mod_data %>% filter(bucket == buck) %>%
      mutate(technique = factor(technique, levels = tech_order)) %>%
      arrange(technique) %>% filter(!is.na(technique))
    mod_rows <- mod_rows + nrow(bd)
  }

  row_counter <- 0
  for (b_idx in 1:2) {
    buck <- c("High", "Low")[b_idx]
    buck_data <- mod_data %>%
      filter(bucket == buck) %>%
      mutate(technique = factor(technique, levels = tech_order)) %>%
      arrange(technique) %>% filter(!is.na(technique))
    n_tech <- nrow(buck_data)
    if (n_tech == 0) next

    for (t_idx in seq_len(n_tech)) {
      r <- buck_data[t_idx, ]
      row_counter <- row_counter + 1

      # Model cell
      if (row_counter == 1) {
        mcell <- sprintf("\\multirow{%d}{*}{\\shortstack[l]{%s}}", mod_rows, mod)
      } else {
        mcell <- ""
      }

      # Bucket cell
      if (t_idx == 1) {
        bcell <- sprintf("\\multirow{%d}{*}{%s}", n_tech, buck)
      } else {
        bcell <- ""
      }

      # Benchmark deg columns
      degs <- sapply(bench_cols, function(bc) fmt_deg(r[[bc]]))
      deg_str <- paste(degs, collapse = " & ")

      # Correlation: show only on High rows
      if (buck == "High") {
        er <- fmt_r(r$ent_r); ep <- fmt_p(r$ent_p)
        lr <- fmt_r(r$len_r); lp <- fmt_p(r$len_p)
      } else {
        er <- ""; ep <- ""; lr <- ""; lp <- ""
      }

      line <- paste0("        ", mcell, " & ", bcell, " & ", r$technique,
                      " & ", deg_str,
                      " & & ", er, " & ", ep,
                      " & & ", lr, " & ", lp, " \\\\")
      L <- c(L, line)
    }

    # cmidrule between High and Low
    if (b_idx == 1) {
      L <- c(L, paste0("        \\cmidrule(l){2-", l_end, "}"))
    }
  }

  # midrule between models
  if (m_idx < length(model_order)) {
    L <- c(L, "        \\midrule")
  }
}

L <- c(L, "        \\bottomrule")
L <- c(L, "    \\end{tabular}")
L <- c(L, "    \\vspace{-0.2cm}")
L <- c(L, "\\end{table*}")

latex_text <- paste(L, collapse = "\n")
latex_path <- file.path(output_dir, "rq3_combined_table.tex")
writeLines(latex_text, latex_path)
cat("  LaTeX table saved:", latex_path, "\n\n")
cat(latex_text, "\n\n")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

cat("================================================================\n")
cat("  SUMMARY\n")
cat("================================================================\n\n")

cat("McNemar tests:", nrow(mcnemar_results), "comparisons\n")
if (nrow(mcnemar_results) > 0) {
  n_sig <- sum(mcnemar_results$significant, na.rm = TRUE)
  cat(sprintf("  Significant: %d / %d\n", n_sig, nrow(mcnemar_results)))
}

cat("\nCorrelation tests:", nrow(correlation_results), "tests\n")
if (nrow(correlation_results) > 0) {
  n_sig_r <- sum(correlation_results$r_significant, na.rm = TRUE)
  n_sig_mw <- sum(correlation_results$mw_significant, na.rm = TRUE)
  cat(sprintf("  Significant point-biserial: %d / %d\n", n_sig_r, nrow(correlation_results)))
  cat(sprintf("  Significant Mann-Whitney:   %d / %d\n", n_sig_mw, nrow(correlation_results)))
}

cat("\nAll outputs saved to:", output_dir, "\n")