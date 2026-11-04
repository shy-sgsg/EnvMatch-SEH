args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected input CSV.gz, model spec CSV, and output directory")
input_path <- args[[1]]
spec_path <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(lavaan))
writeLines(as.character(packageVersion("lavaan")), file.path(out_dir, "lavaan_version.txt"))
data_all <- read.csv(gzfile(input_path), check.names = FALSE, stringsAsFactors = FALSE)
spec <- read.csv(spec_path, check.names = FALSE, stringsAsFactors = FALSE)

structures <- c("S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI")
controls <- c("M0", "M1", "M2", "M3")
base_dims <- c("physical", "chemical", "biological_activity")

make_model <- function(structure, ctrl, drop_erosion = FALSE) {
  x <- if (drop_erosion) character(0) else "z_E"
  preds <- paste(c(x, ctrl), collapse = " + ")
  if (!nzchar(preds)) preds <- "1"
  lines <- c(
    paste0("physical ~ ", preds),
    paste0("chemical ~ ", preds),
    paste0("biological_activity ~ ", preds),
    "physical ~~ chemical",
    "physical ~~ biological_activity",
    "chemical ~~ biological_activity"
  )
  if (structure != "S1_parallel_dimensions") {
    shi_preds <- c(if (structure == "S2_direct_plus_indirect_SHI" && !drop_erosion) "z_E", base_dims, ctrl)
    lines <- c(lines, paste0("z_SHI ~ ", paste(shi_preds, collapse = " + ")))
  }
  paste(lines, collapse = "\n")
}

fit_once <- function(d, model) {
  vars <- unique(unlist(regmatches(model, gregexpr("[A-Za-z][A-Za-z0-9_]*", model))))
  vars <- unique(vars[vars %in% names(d)])
  S <- stats::cov(d[, vars, drop = FALSE])
  M <- colMeans(d[, vars, drop = FALSE])
  lavaan::sem(model, sample.cov = S, sample.mean = M, sample.nobs = nrow(d), fixed.x = TRUE,
              estimator = "ML", meanstructure = TRUE,
              control = list(iter.max = 1000L), warn = FALSE)
}

fit_rows <- list(); path_rows <- list(); delta_rows <- list(); idx <- 1L; pidx <- 1L; didx <- 1L
for (dataset in unique(spec$dataset)) {
  d_all <- data_all[data_all$dataset == dataset, , drop = FALSE]
  for (control in controls) {
    meta <- spec[spec$dataset == dataset & spec$control == control, , drop = FALSE]
    ctrl <- if (nrow(meta) && nzchar(meta$controls[[1]])) strsplit(meta$controls[[1]], ";", fixed = TRUE)[[1]] else character(0)
    variables <- unique(c("z_E", base_dims, "z_SHI", ctrl))
    d <- d_all[, c("dataset", variables), drop = FALSE]
    d <- d[stats::complete.cases(d[, variables, drop = FALSE]), , drop = FALSE]
    for (structure in structures) {
      model <- make_model(structure, ctrl)
      fit <- tryCatch(fit_once(d, model), error = function(e) e)
      if (inherits(fit, "error")) {
        fit_rows[[idx]] <- data.frame(dataset=dataset, method="CB_SEM_lavaan_ML", structure=structure, control=control,
          endogenous="joint_model", n=nrow(d), r_squared=NA_real_, adjusted_r_squared=NA_real_, aic=NA_real_, bic=NA_real_,
          cfi_naive=NA_real_, tli_naive=NA_real_, rmsea_naive=NA_real_, srmr=NA_real_, df=NA_real_, status=paste0("failed:", conditionMessage(fit)),
          fit_note="lavaan covariance-based observed-variable path model; conventional iid ML indices", outcome_guided_selection=TRUE, stringsAsFactors=FALSE)
        idx <- idx + 1L
        next
      }
      fm <- tryCatch(lavaan::fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")), error=function(e) rep(NA_real_, 9L))
      r2 <- tryCatch(lavaan::inspect(fit, "rsquare"), error=function(e) numeric(0))
      r2v <- setNames(rep(NA_real_, 4L), c("r2_physical", "r2_chemical", "r2_biological_activity", "r2_SHI"))
      for (name in names(r2)) {
        dest <- if (name == "z_SHI") "r2_SHI" else paste0("r2_", name)
        if (dest %in% names(r2v)) r2v[[dest]] <- unname(r2[[name]])
      }
      main_r2 <- if (structure == "S1_parallel_dimensions") mean(r2v[1:3], na.rm=TRUE) else r2v[["r2_SHI"]]
      if (!is.finite(main_r2)) main_r2 <- NA_real_
      fit_rows[[idx]] <- data.frame(dataset=dataset, method="CB_SEM_lavaan_ML", structure=structure, control=control,
        endogenous="joint_model", n=nrow(d), r_squared=main_r2, adjusted_r_squared=NA_real_, aic=unname(fm["aic"]), bic=unname(fm["bic"]),
        chi_square_naive=unname(fm["chisq"]), df=unname(fm["df"]), cfi_naive=unname(fm["cfi"]), tli_naive=unname(fm["tli"]),
        rmsea_naive=unname(fm["rmsea"]), srmr=unname(fm["srmr"]), r2_physical=r2v[["r2_physical"]], r2_chemical=r2v[["r2_chemical"]],
        r2_biological_activity=r2v[["r2_biological_activity"]], r2_SHI=r2v[["r2_SHI"]], status=ifelse(lavaan::lavInspect(fit,"converged"),"ok","not_converged"),
        fit_note="lavaan observed-variable CB-SEM; conventional iid ML fit indices; spatial dependence not accounted for in fit indices",
        outcome_guided_selection=TRUE, stringsAsFactors=FALSE)
      idx <- idx + 1L
      std <- tryCatch(lavaan::standardizedSolution(fit, type="std.all", se=TRUE, zstat=TRUE, pvalue=TRUE), error=function(e) NULL)
      if (!is.null(std) && nrow(std)) {
        keep <- std$op == "~" | (std$op == "~~" & std$lhs != std$rhs & std$lhs %in% base_dims & std$rhs %in% base_dims)
        std <- std[keep, , drop=FALSE]
        if (nrow(std)) {
          path_rows[[pidx]] <- data.frame(dataset=dataset, method="CB_SEM_lavaan_ML", structure=structure, control=control,
            from=std$rhs, to=std$lhs, path_type=ifelse(std$op == "~~", "residual_covariance", "structural"),
            beta_standardized=std$est.std, se=std$se, ci_low=std$ci.lower, ci_high=std$ci.upper, p_value_nominal=std$pvalue,
            interval_status="lavaan iid ML delta-method interval", r_squared=NA_real_, adjusted_r_squared=NA_real_,
            n=nrow(d), spatial_blocks_5km=NA_integer_, se_type="lavaan ML; iid; not spatial-block robust",
            design_condition_number=NA_real_, outcome_guided_selection=TRUE, stringsAsFactors=FALSE)
          pidx <- pidx + 1L
        }
      }
      without <- tryCatch(fit_once(d, make_model(structure, ctrl, drop_erosion=TRUE)), error=function(e) NULL)
      if (!is.null(without)) {
        r2_without <- tryCatch(lavaan::inspect(without, "rsquare"), error=function(e) numeric(0))
        for (outcome in if (structure == "S1_parallel_dimensions") base_dims else "z_SHI") {
          full_value <- if (outcome %in% names(r2)) unname(r2[[outcome]]) else NA_real_
          base_value <- if (outcome %in% names(r2_without)) unname(r2_without[[outcome]]) else 0
          delta_rows[[didx]] <- data.frame(dataset=dataset, method="CB_SEM_lavaan_ML", structure=structure, control=control,
            outcome=outcome, n=nrow(d), r2_full=full_value, r2_without_erosion=base_value,
            delta_r2_erosion_conditional=full_value-base_value,
            interpretation="same-row conditional R2 increment from erosion paths; not causal contribution", outcome_guided_selection=TRUE,
            branch="lavaan nested model comparison", stringsAsFactors=FALSE)
          didx <- didx + 1L
        }
      }
    }
  }
}
write.csv(do.call(rbind, fit_rows), file.path(out_dir, "lavaan_model_fit.csv"), row.names=FALSE, na="")
write.csv(do.call(rbind, path_rows), file.path(out_dir, "lavaan_path_coefficients.csv"), row.names=FALSE, na="")
write.csv(do.call(rbind, delta_rows), file.path(out_dir, "lavaan_erosion_increment.csv"), row.names=FALSE, na="")
