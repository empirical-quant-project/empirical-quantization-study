rm(list=ls())

# Install required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

# ──────────────────────────── CONFIG ────────────────────────────
BASE_DIR <- "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Analysis-and-Reports/Java/per-directory"

BENCHMARKS <- c("McEval-Java", "CoderEval-Java")
MODELS <- c("CodeLlama-7B", "Qwen2.5-Coder-7B")
TECHNIQUES <- c("AWQ", "GPTQ", "BitsAndBytes", "AQLM", "GGUF", "QUIP")
METRICS <- c("LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC")

# Output CSV path
OUTPUT_DIR <- "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/statistical-tests"

# ──────────────────────── HELPER FUNCTIONS ──────────────────────

# Check if a column is constant
is_constant <- function(x) {
  return(length(unique(x)) == 1)
}

# Wilcoxon signed-rank test (paired, two-sided)
wilcox_test <- function(x, y) {
  if (is_constant(x) & is_constant(y)) {
    return(NA)
  }
  return(wilcox.test(x, y, alternative="two.side", paired=TRUE, exact=FALSE, correct=FALSE)$p.value)
}

# Safe Cliff's delta
safe_cliff_delta <- function(x, y) {
  if (is_constant(x) & is_constant(y)) {
    return(list(estimate=0, magnitude="negligible"))
  }
  result <- cliff.delta(x, y)
  return(list(estimate=result$estimate, magnitude=as.character(result$magnitude)))
}

# Build file path
build_path <- function(benchmark, model, method) {
  filename <- paste0(benchmark, "-", model, "-", method, ".csv")
  return(file.path(BASE_DIR, filename))
}

# ──────────────────────── MAIN ANALYSIS ─────────────────────────

# Collect all results
all_results <- data.frame(
  Benchmark = character(),
  Model = character(),
  Technique = character(),
  Metric = character(),
  Wilcoxon_p = numeric(),
  Wilcoxon_p_adjusted = numeric(),
  Cliff_delta = numeric(),
  Effect_size = character(),
  stringsAsFactors = FALSE
)

for (benchmark in BENCHMARKS) {
  for (model in MODELS) {
    
    # Load FP baseline
    fp_path <- build_path(benchmark, model, "FP")
    if (!file.exists(fp_path)) {
      cat(sprintf("  ⏭  FP file not found: %s\n", fp_path))
      next
    }
    fp_data <- read.csv(fp_path, header=TRUE)
    
    cat(sprintf("\n%s\n", paste(rep("=", 80), collapse="")))
    cat(sprintf("  %s | %s | FP vs Quantized (Java)\n", benchmark, model))
    cat(sprintf("%s\n\n", paste(rep("=", 80), collapse="")))
    
    # Collect p-values for this benchmark+model group (for Holm correction)
    group_results <- list()
    
    for (technique in TECHNIQUES) {
      
      quant_path <- build_path(benchmark, model, technique)
      if (!file.exists(quant_path)) {
        cat(sprintf("  ⏭  Skipping %s (file not found)\n", technique))
        next
      }
      quant_data <- read.csv(quant_path, header=TRUE)
      
      # Ensure same length
      min_len <- min(nrow(fp_data), nrow(quant_data))
      fp_trimmed <- fp_data[1:min_len, ]
      quant_trimmed <- quant_data[1:min_len, ]
      
      cat(sprintf("  ─── FP vs %s (n=%d) ───\n", technique, min_len))
      
      for (metric in METRICS) {
        fp_vals <- fp_trimmed[[metric]]
        quant_vals <- quant_trimmed[[metric]]
        
        # Wilcoxon test
        p_val <- wilcox_test(fp_vals, quant_vals)
        
        # Cliff's delta
        cd <- safe_cliff_delta(fp_vals, quant_vals)
        
        cat(sprintf("    %s: p=%.6f, Cliff's d=%.4f (%s)\n", 
                    metric, ifelse(is.na(p_val), NA, p_val), cd$estimate, cd$magnitude))
        
        group_results[[length(group_results) + 1]] <- list(
          Benchmark = benchmark,
          Model = model,
          Technique = technique,
          Metric = metric,
          Wilcoxon_p = p_val,
          Cliff_delta = cd$estimate,
          Effect_size = cd$magnitude
        )
      }
      cat("\n")
    }
    
    # Apply Holm correction within this benchmark+model group
    p_vals <- sapply(group_results, function(x) x$Wilcoxon_p)
    p_adjusted <- p.adjust(p_vals, method="holm")
    
    for (i in seq_along(group_results)) {
      r <- group_results[[i]]
      all_results <- rbind(all_results, data.frame(
        Benchmark = r$Benchmark,
        Model = r$Model,
        Technique = r$Technique,
        Metric = r$Metric,
        Wilcoxon_p = r$Wilcoxon_p,
        Wilcoxon_p_adjusted = p_adjusted[i],
        Cliff_delta = r$Cliff_delta,
        Effect_size = r$Effect_size,
        stringsAsFactors = FALSE
      ))
    }
    
    # Print adjusted p-values summary for this group
    cat("  Holm-adjusted p-values:\n")
    for (i in seq_along(group_results)) {
      r <- group_results[[i]]
      sig <- ifelse(!is.na(p_adjusted[i]) & p_adjusted[i] < 0.05, " *", "")
      cat(sprintf("    %s | %s: raw=%.6f, adj=%.6f%s\n",
                  r$Technique, r$Metric,
                  ifelse(is.na(r$Wilcoxon_p), NA, r$Wilcoxon_p),
                  ifelse(is.na(p_adjusted[i]), NA, p_adjusted[i]),
                  sig))
    }
    cat("\n")
  }
}

# ──────────────────────── SAVE RESULTS ──────────────────────────

# Save full results
output_path <- file.path(OUTPUT_DIR, "wilcoxon_results_java.csv")
write.csv(all_results, output_path, row.names=FALSE)
cat(sprintf("\n✓ Results saved to: %s\n", output_path))

# Print summary table
cat("\n")
cat(paste(rep("=", 80), collapse=""))
cat("\n  SUMMARY: Significant results (Holm-adjusted p < 0.05)\n")
cat(paste(rep("=", 80), collapse=""))
cat("\n\n")

sig_results <- all_results[!is.na(all_results$Wilcoxon_p_adjusted) & all_results$Wilcoxon_p_adjusted < 0.05, ]
if (nrow(sig_results) > 0) {
  print(sig_results)
} else {
  cat("  No significant differences found.\n")
}

cat(sprintf("\nTotal comparisons: %d\n", nrow(all_results)))
cat(sprintf("Significant (p < 0.05 after Holm): %d\n", nrow(sig_results)))