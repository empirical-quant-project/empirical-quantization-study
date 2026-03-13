rm(list=ls())

# ──────────────────────────── CONFIG ────────────────────────────
BASE_DIR <- "/scratch/oldhome/user/projects/Empirical-Quantization-Study/Analysis-and-Reports/Python/pass@1-values"

BENCHMARKS <- list(
  "McEval-Python" = file.path(BASE_DIR, "McEval"),
  "CoderEval-Python" = file.path(BASE_DIR, "CoderEval"),
  "BCB-Python" = file.path(BASE_DIR, "BCB")
)

MODELS <- c("CodeLlama-7B", "Qwen2.5-Coder-7B")
TECHNIQUES <- c("AWQ", "GPTQ", "BitsAndBytes", "AQLM", "GGUF", "QUIP")

# Output
OUTPUT_DIR <- "/scratch/oldhome/user/projects/Empirical-Quantization-Study/statistical-tests"

# ──────────────────────── HELPER FUNCTIONS ──────────────────────

build_path <- function(bench_dir, benchmark, model, method) {
  filename <- paste0(benchmark, "-", model, "-", method, ".csv")
  return(file.path(bench_dir, filename))
}

# McNemar's test with odds ratio
run_mcnemar <- function(fp_pass, quant_pass) {
  a <- sum(fp_pass == 1 & quant_pass == 1)  # both pass
  b <- sum(fp_pass == 1 & quant_pass == 0)  # FP pass, quant fail
  c <- sum(fp_pass == 0 & quant_pass == 1)  # FP fail, quant pass
  d <- sum(fp_pass == 0 & quant_pass == 0)  # both fail
  
  contingency <- matrix(c(a, b, c, d), nrow=2, byrow=TRUE,
                         dimnames=list(FP=c("Pass", "Fail"),
                                       Quantized=c("Pass", "Fail")))
  
  # Odds ratio with 0.5 correction
  odds_ratio <- (b + 0.5) / (c + 0.5)
  
  if ((b + c) == 0) {
    return(list(p.value=NA, odds_ratio=odds_ratio, a=a, b=b, c=c, d=d))
  }
  
  # McNemar's test with continuity correction
  result <- mcnemar.test(contingency, correct=TRUE)
  
  return(list(p.value=result$p.value, odds_ratio=odds_ratio, a=a, b=b, c=c, d=d))
}

# ──────────────────────── MAIN ANALYSIS ─────────────────────────

all_results <- data.frame(
  Benchmark = character(),
  Model = character(),
  Technique = character(),
  FP_Pass = integer(),
  FP_Total = integer(),
  Quant_Pass = integer(),
  Quant_Total = integer(),
  Both_Pass = integer(),
  FP_only = integer(),
  Quant_only = integer(),
  Both_Fail = integer(),
  Odds_Ratio = numeric(),
  McNemar_p = numeric(),
  McNemar_p_adjusted = numeric(),
  stringsAsFactors = FALSE
)

for (bench_name in names(BENCHMARKS)) {
  bench_dir <- BENCHMARKS[[bench_name]]
  
  for (model in MODELS) {
    
    fp_path <- build_path(bench_dir, bench_name, model, "FP")
    if (!file.exists(fp_path)) {
      cat(sprintf("  ⏭  FP file not found: %s\n", fp_path))
      next
    }
    fp_data <- read.csv(fp_path, header=TRUE)
    
    cat(sprintf("\n%s\n", paste(rep("=", 80), collapse="")))
    cat(sprintf("  %s | %s | McNemar's Test (FP vs Quantized)\n", bench_name, model))
    cat(sprintf("%s\n\n", paste(rep("=", 80), collapse="")))
    cat(sprintf("  FP baseline: %d/%d passed (%.1f%%)\n\n",
                sum(fp_data$Pass), nrow(fp_data), sum(fp_data$Pass)/nrow(fp_data)*100))
    
    group_results <- list()
    
    for (technique in TECHNIQUES) {
      
      quant_path <- build_path(bench_dir, bench_name, model, technique)
      if (!file.exists(quant_path)) {
        cat(sprintf("  ⏭  Skipping %s (file not found)\n", technique))
        next
      }
      quant_data <- read.csv(quant_path, header=TRUE)
      
      min_len <- min(nrow(fp_data), nrow(quant_data))
      fp_trimmed <- fp_data[1:min_len, ]
      quant_trimmed <- quant_data[1:min_len, ]
      
      result <- run_mcnemar(fp_trimmed$Pass, quant_trimmed$Pass)
      
      fp_pass_count <- sum(fp_trimmed$Pass)
      quant_pass_count <- sum(quant_trimmed$Pass)
      
      cat(sprintf("  ─── FP vs %s ───\n", technique))
      cat(sprintf("    FP: %d/%d passed | %s: %d/%d passed\n",
                  fp_pass_count, min_len, technique, quant_pass_count, min_len))
      cat(sprintf("    Contingency: both_pass=%d, FP_only=%d, quant_only=%d, both_fail=%d\n",
                  result$a, result$b, result$c, result$d))
      cat(sprintf("    McNemar p-value: %s, Odds Ratio: %.4f\n\n",
                  ifelse(is.na(result$p.value), "NA (no discordant pairs)",
                         sprintf("%.6f", result$p.value)),
                  result$odds_ratio))
      
      group_results[[length(group_results) + 1]] <- list(
        Benchmark = bench_name,
        Model = model,
        Technique = technique,
        FP_Pass = fp_pass_count,
        FP_Total = min_len,
        Quant_Pass = quant_pass_count,
        Quant_Total = min_len,
        Both_Pass = result$a,
        FP_only = result$b,
        Quant_only = result$c,
        Both_Fail = result$d,
        Odds_Ratio = result$odds_ratio,
        McNemar_p = result$p.value
      )
    }
    
    # Apply Holm correction within this benchmark+model group
    p_vals <- sapply(group_results, function(x) x$McNemar_p)
    p_adjusted <- p.adjust(p_vals, method="holm")
    
    for (i in seq_along(group_results)) {
      r <- group_results[[i]]
      all_results <- rbind(all_results, data.frame(
        Benchmark = r$Benchmark,
        Model = r$Model,
        Technique = r$Technique,
        FP_Pass = r$FP_Pass,
        FP_Total = r$FP_Total,
        Quant_Pass = r$Quant_Pass,
        Quant_Total = r$Quant_Total,
        Both_Pass = r$Both_Pass,
        FP_only = r$FP_only,
        Quant_only = r$Quant_only,
        Both_Fail = r$Both_Fail,
        Odds_Ratio = r$Odds_Ratio,
        McNemar_p = r$McNemar_p,
        McNemar_p_adjusted = p_adjusted[i],
        stringsAsFactors = FALSE
      ))
    }
    
    cat("  Holm-adjusted p-values:\n")
    for (i in seq_along(group_results)) {
      r <- group_results[[i]]
      sig <- ifelse(!is.na(p_adjusted[i]) & p_adjusted[i] < 0.05, " *", "")
      cat(sprintf("    %s: raw=%s, adj=%s, OR=%.4f%s\n",
                  r$Technique,
                  ifelse(is.na(r$McNemar_p), "NA", sprintf("%.6f", r$McNemar_p)),
                  ifelse(is.na(p_adjusted[i]), "NA", sprintf("%.6f", p_adjusted[i])),
                  r$Odds_Ratio,
                  sig))
    }
    cat("\n")
  }
}

# ──────────────────────── SAVE RESULTS ──────────────────────────

output_path <- file.path(OUTPUT_DIR, "mcnemar_results_python.csv")
write.csv(all_results, output_path, row.names=FALSE)
cat(sprintf("\n✓ Results saved to: %s\n", output_path))

cat("\n")
cat(paste(rep("=", 80), collapse=""))
cat("\n  SUMMARY: Significant results (Holm-adjusted p < 0.05)\n")
cat(paste(rep("=", 80), collapse=""))
cat("\n\n")

sig_results <- all_results[!is.na(all_results$McNemar_p_adjusted) & all_results$McNemar_p_adjusted < 0.05, ]
if (nrow(sig_results) > 0) {
  print(sig_results)
} else {
  cat("  No significant differences found.\n")
}

cat(sprintf("\nTotal comparisons: %d\n", nrow(all_results)))
cat(sprintf("Significant (p < 0.05 after Holm): %d\n", nrow(sig_results)))