# === Friedman test on pass@1 ===
# (1) Across configurations: TaskID is the within-subjects identifier; configs
#     {FP, AWQ, GPTQ, GGUF, BnB, AQLM, QuIP} are repeated measures (run-1 values
#     are deterministic representatives).
# (2) Per-config across runs: TaskID is the subject; Run1..Run10 are repeated
#     measures. Documents that pass@1 has zero run-to-run variance.

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(sub("^--file=", "", file_arg[1])) else "rebuttal"
if (script_dir == "") script_dir <- "rebuttal"

input_dir <- file.path(script_dir, "friedman", "inputs_pass1")
output_dir <- file.path(script_dir, "friedman")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ── (1) Across configurations ──
cat("=== pass@1 Friedman across configurations ===\n")
cfg_path <- file.path(input_dir, "Qwen2.5-Coder-7B_McEval-Python_pass1_configs.csv")
df <- read.csv(cfg_path, stringsAsFactors = FALSE, check.names = FALSE)
for (j in 2:ncol(df)) df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
df <- df[complete.cases(df[, -1, drop = FALSE]), , drop = FALSE]

col_var <- apply(df[, -1, drop = FALSE], 2, var)
all_const <- all(is.na(col_var)) || all(df[, -1, drop = FALSE] == df[[2]], na.rm = TRUE)
if (all_const) {
  pval <- NA
} else {
  df_long <- reshape(df, direction = "long",
                     varying = names(df)[-1], v.names = "value",
                     timevar = "config", times = names(df)[-1], idvar = "TaskID")
  df_long$config <- factor(df_long$config, levels = names(df)[-1])
  pval <- tryCatch(friedman.test(value ~ config | TaskID, data = df_long)$p.value,
                   error = function(e) NA)
}

cfg_results <- data.frame(
  model = "Qwen2.5-Coder-7B_McEval-Python",
  metric = "pass@1",
  n_tasks = nrow(df),
  n_configs = ncol(df) - 1,
  p.value = pval,
  signif_0.01 = ifelse(!is.na(pval) & pval < 0.01, "Yes", "No"),
  signif_0.05 = ifelse(!is.na(pval) & pval < 0.05, "Yes", "No"),
  signif_0.10 = ifelse(!is.na(pval) & pval < 0.10, "Yes", "No")
)
print(cfg_results)
write.csv(cfg_results, file.path(output_dir, "friedman_pass1_results.csv"), row.names = FALSE)

# ── (2) Per-config across runs ──
cat("\n=== pass@1 Friedman across 10 runs (per config) ===\n")
configs <- c("FP", "AWQ", "GPTQ", "GGUF", "BnB", "AQLM", "QuIP")
runs_results <- data.frame()

for (cfg in configs) {
  path <- file.path(input_dir, sprintf("Qwen2.5-Coder-7B_McEval-Python_pass1_runs_%s.csv", cfg))
  if (!file.exists(path)) next
  d <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  for (j in 2:ncol(d)) d[[j]] <- suppressWarnings(as.numeric(d[[j]]))
  d <- d[complete.cases(d[, -1, drop = FALSE]), , drop = FALSE]
  if (nrow(d) < 2) next

  row_var <- apply(d[, -1, drop = FALSE], 1, var)
  any_run_variation <- any(!is.na(row_var) & row_var > 0)

  if (!any_run_variation) {
    pval <- NA
    note <- "deterministic (constant across runs)"
  } else {
    d_long <- reshape(d, direction = "long",
                      varying = names(d)[-1], v.names = "value",
                      timevar = "run", times = names(d)[-1], idvar = "TaskID")
    d_long$run <- factor(d_long$run, levels = names(d)[-1])
    pval <- tryCatch(friedman.test(value ~ run | TaskID, data = d_long)$p.value,
                     error = function(e) NA)
    note <- sprintf("p = %s", format(pval, digits = 4))
  }
  cat(sprintf("  %-6s -> %s\n", cfg, note))

  runs_results <- rbind(runs_results, data.frame(
    model = "Qwen2.5-Coder-7B_McEval-Python",
    config = cfg, metric = "pass@1",
    n_tasks = nrow(d), n_runs = ncol(d) - 1,
    p.value = pval,
    deterministic = !any_run_variation
  ))
}

write.csv(runs_results, file.path(output_dir, "friedman_pass1_runs_results.csv"), row.names = FALSE)

cat("\nWrote:\n")
cat("  ", file.path(output_dir, "friedman_pass1_results.csv"), "\n")
cat("  ", file.path(output_dir, "friedman_pass1_runs_results.csv"), "\n")
