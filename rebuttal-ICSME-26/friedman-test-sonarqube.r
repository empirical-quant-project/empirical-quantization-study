# === SonarQube Friedman Test (rebuttal) ===
# Adapted from /scratch/oldhome/user/projects/NonFunc-AWQ/scripts/statistical-tests/R-scripts/friedman-test-sonarqube.r
#
# Reads wide-format CSVs (one per metric) from rebuttal/friedman/inputs/, where
# each CSV has columns: TaskID, FP, AWQ, GPTQ, GGUF, BnB, AQLM, QuIP. Runs
# Friedman's test treating TaskID as the within-subjects identifier and the
# 7 configuration columns as repeated measures.
#
# Usage:
#   cd /scratch/oldhome/user/projects/Empirical-Quantization-Study
#   Rscript rebuttal/friedman-test-sonarqube.r
#
# Output:
#   rebuttal/friedman/friedman_sonarqube_results.csv
#       columns: model, metric, p.value, signif (Yes if p<0.01)
#   rebuttal/friedman/friedman_sonarqube_results-0.05.csv
#   rebuttal/friedman/friedman_sonarqube_results-0.10.csv

rm(list = ls())

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(sub("^--file=", "", file_arg[1])) else "rebuttal"
if (script_dir == "") script_dir <- "rebuttal"

input_dir   <- file.path(script_dir, "friedman", "inputs")
output_dir  <- file.path(script_dir, "friedman")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

metrics <- c("LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC")

files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  stop(sprintf("No input CSVs in %s -- run build_friedman_inputs.py first.", input_dir))
}

all_results <- data.frame()

for (metric in metrics) {
  metric_files <- files[grepl(paste0("_", metric, "\\.csv$"), files)]
  for (f in metric_files) {
    df <- read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
    if (ncol(df) < 3) next

    # Coerce columns 2..N to numeric; drop rows with any NA
    for (j in 2:ncol(df)) df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
    df <- df[complete.cases(df[, -1, drop = FALSE]), , drop = FALSE]
    if (nrow(df) < 2) next

    # If a metric is constant across all configs (e.g., Security_Hotspots all 0),
    # Friedman's test is undefined.
    col_var <- apply(df[, -1, drop = FALSE], 2, var)
    if (all(is.na(col_var)) || all(df[, -1, drop = FALSE] == df[[2]], na.rm = TRUE)) {
      pval <- NA
    } else {
      df_long <- reshape(
        df, direction = "long",
        varying = names(df)[-1], v.names = "value",
        timevar = "config", times = names(df)[-1], idvar = "TaskID"
      )
      df_long$config <- factor(df_long$config, levels = names(df)[-1])
      pval <- tryCatch(
        friedman.test(value ~ config | TaskID, data = df_long)$p.value,
        error = function(e) NA
      )
    }

    base <- sub("\\.csv$", "", basename(f))
    # Strip the trailing _<metric>; what's left identifies the model+benchmark.
    model <- sub(paste0("_", metric, "$"), "", base)

    n_tasks <- nrow(df)
    n_configs <- ncol(df) - 1
    cat(sprintf("  [%s] %s  metric=%s  p=%s  (n_tasks=%d, k=%d)\n",
                base, model, metric,
                ifelse(is.na(pval), "NA", format(pval, digits = 4)),
                n_tasks, n_configs))

    all_results <- rbind(all_results, data.frame(
      model     = model,
      metric    = metric,
      n_tasks   = n_tasks,
      n_configs = n_configs,
      p.value   = pval
    ))
  }
}

# Significance flags at three thresholds
all_results$signif_0.01 <- ifelse(!is.na(all_results$p.value) & all_results$p.value < 0.01, "Yes", "No")
all_results$signif_0.05 <- ifelse(!is.na(all_results$p.value) & all_results$p.value < 0.05, "Yes", "No")
all_results$signif_0.10 <- ifelse(!is.na(all_results$p.value) & all_results$p.value < 0.10, "Yes", "No")

write.csv(all_results, file.path(output_dir, "friedman_sonarqube_results.csv"), row.names = FALSE)
write.csv(subset(all_results, !is.na(p.value) & p.value < 0.05),
          file.path(output_dir, "friedman_sonarqube_results-0.05.csv"), row.names = FALSE)
write.csv(subset(all_results, !is.na(p.value) & p.value < 0.10),
          file.path(output_dir, "friedman_sonarqube_results-0.10.csv"), row.names = FALSE)

cat("\nWrote:\n")
cat("  ", file.path(output_dir, "friedman_sonarqube_results.csv"), "\n")
cat("  ", file.path(output_dir, "friedman_sonarqube_results-0.05.csv"), "\n")
cat("  ", file.path(output_dir, "friedman_sonarqube_results-0.10.csv"), "\n")

# ────────────────────────────────────────────────────────────────────
#  Per-RUN Friedman test (R1_C3 output-variability check)
#  For each (config, metric), test Friedman across the 10 generation runs.
#  Determinism at temperature=0 was already verified (Std=0 on pass@1);
#  this confirms it extends to SonarCloud structural metrics.
# ────────────────────────────────────────────────────────────────────

cat("\n=== Per-run Friedman (output variability across 10 runs per config) ===\n")
runs_input_dir <- file.path(script_dir, "friedman", "inputs_runs")
runs_output_dir <- file.path(script_dir, "friedman")

run_files <- list.files(runs_input_dir, pattern = "\\.csv$", full.names = TRUE)
run_results <- data.frame()

known_metrics <- c("LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC")
for (f in run_files) {
  base <- sub("\\.csv$", "", basename(f))
  # Parse "<MODEL>_<BENCH>_<CONFIG>_<METRIC>" -- metric may itself contain '_'
  metric <- ""
  for (cand in known_metrics) {
    pat <- paste0("_", cand, "$")
    if (grepl(pat, base)) { metric <- cand; break }
  }
  if (metric == "") next
  base_no_metric <- sub(paste0("_", metric, "$"), "", base)
  parts <- strsplit(base_no_metric, "_")[[1]]
  if (length(parts) < 2) next
  cfg   <- parts[length(parts)]
  model <- paste(parts[1:(length(parts) - 1)], collapse = "_")

  df <- read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  if (ncol(df) < 3) next
  for (j in 2:ncol(df)) df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
  df <- df[complete.cases(df[, -1, drop = FALSE]), , drop = FALSE]
  if (nrow(df) < 2) next

  # Per-row variance: are the 10 runs identical for every task?
  row_var <- apply(df[, -1, drop = FALSE], 1, var)
  any_run_variation <- any(!is.na(row_var) & row_var > 0)

  if (!any_run_variation) {
    pval <- NA  # constant across runs -> Friedman undefined
    note <- "deterministic (constant across runs)"
  } else {
    df_long <- reshape(
      df, direction = "long",
      varying = names(df)[-1], v.names = "value",
      timevar = "run", times = names(df)[-1], idvar = "TaskID"
    )
    df_long$run <- factor(df_long$run, levels = names(df)[-1])
    pval <- tryCatch(
      friedman.test(value ~ run | TaskID, data = df_long)$p.value,
      error = function(e) NA
    )
    note <- sprintf("p = %s", format(pval, digits = 4))
  }

  cat(sprintf("  %-25s metric=%-18s -> %s\n", cfg, metric, note))

  run_results <- rbind(run_results, data.frame(
    model = model, config = cfg, metric = metric,
    n_tasks = nrow(df),
    n_runs  = ncol(df) - 1,
    p.value = pval,
    deterministic = !any_run_variation
  ))
}

write.csv(run_results, file.path(runs_output_dir, "friedman_sonarqube_runs_results.csv"), row.names = FALSE)
cat("\nWrote: ", file.path(runs_output_dir, "friedman_sonarqube_runs_results.csv"), "\n")
