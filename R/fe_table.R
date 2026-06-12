#' Generate fixed effects tables
#'
#' @param ... One or more model objects of \code{felm} or \code{feglm} class
#'   fitted with \code{return_fe = TRUE} in \code{control}.
#' @param n Number of levels to show per fixed effect dimension. Use
#'   \code{Inf} to show all levels. The default is 5.
#' @param coef_digits Number of digits for fixed effect coefficients. The
#'   default is 3.
#' @param latex Whether to output as LaTeX code. The default is \code{FALSE}.
#' @param model_names Optional vector of custom model names.
#' @param caption Optional caption for the table (LaTeX only).
#' @param label Optional label for cross-referencing (LaTeX only).
#' @param position LaTeX float position specifier (LaTeX only). The default
#'   is \code{"htbp"}.
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' ross2004_subset <- ross2004_subset[ross2004_subset$ltrade >
#'   quantile(ross2004_subset$ltrade, 0.75), ]
#'
#' m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
#' m2 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)
#'
#' fe_table(m1, m2, model_names = c("Linear", "Poisson"))
#' @return A formatted fixed effects table of class \code{summary_table}.
#' @export
fe_table <- function(
  ...,
  n = 5,
  coef_digits = 3,
  latex = FALSE,
  model_names = NULL,
  caption = NULL,
  label = NULL,
  position = "htbp"
) {
  models <- list(...)

  # Validate model classes
  valid_classes <- c("felm", "feglm")
  invisible(lapply(seq_along(models), function(i) {
    if (!inherits(models[[i]], valid_classes)) {
      stop("Model ", i, " is not a felm or feglm object")
    }
  }))

  # Validate fixed_effects presence
  invisible(lapply(seq_along(models), function(i) {
    if (is.null(models[[i]]$fixed_effects)) {
      stop(
        "Model ", i, " has no fixed effects. ",
        "Refit with control = list(return_fe = TRUE)."
      )
    }
  }))

  # Validate n
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop("`n` must be a positive number or Inf")
  }

  # Set model names
  if (is.null(model_names)) {
    model_names <- names(models)
    if (is.null(model_names) || all(model_names == "")) {
      model_names <- paste0("(", seq_along(models), ")")
    }
  } else {
    if (length(model_names) != length(models)) {
      stop("Length of model_names must match the number of models")
    }
  }

  # Collect all FE dimension names in order of first appearance
  fe_dims <- unique(unlist(lapply(models, function(m) names(m$fixed_effects))))

  # For each dimension build a list with levels, formatted matrix, and counts
  dim_data <- lapply(fe_dims, function(dim) {
    # Union of levels in model order (first-seen order preserved)
    all_levels <- unique(unlist(lapply(models, function(m) {
      fe <- m$fixed_effects[[dim]]
      if (!is.null(fe)) names(fe) else NULL
    })))

    total <- length(all_levels)
    shown_levels <- if (is.finite(n)) head(all_levels, n) else all_levels
    n_shown <- length(shown_levels)

    # Matrix [n_shown x n_models] of formatted coefficient strings
    mat <- do.call(cbind, lapply(models, function(m) {
      fe <- m$fixed_effects[[dim]]
      vapply(shown_levels, function(lvl) {
        if (!is.null(fe) && lvl %in% names(fe)) {
          formatC(fe[[lvl]], digits = coef_digits, format = "f")
        } else {
          NA_character_
        }
      }, character(1))
    }))

    # Ensure mat is always a matrix (cbind of single values gives a vector)
    if (!is.matrix(mat)) mat <- matrix(mat, nrow = n_shown)

    list(dim = dim, levels = shown_levels, mat = mat, total = total, n_shown = n_shown)
  })

  if (latex) {
    fe_format_latex(dim_data, model_names, caption, label, position)
  } else {
    fe_format_console(dim_data, model_names)
  }
}

fe_format_console <- function(dim_data, model_names) {
  col_names <- c("Level", model_names)

  sections <- lapply(dim_data, function(d) {
    # Full display matrix: [n_shown x (1 + n_models)]
    data_mat <- cbind(d$levels, d$mat)

    # Column widths: max content width (including header) + 2 padding
    all_text <- rbind(col_names, data_mat)
    col_widths <- apply(all_text, 2, function(col) {
      max(nchar(ifelse(is.na(col), "", as.character(col)))) + 2L
    })

    # Center-aligned header
    header <- paste0(
      "| ",
      paste0(
        mapply(function(name, width) {
          padding <- width - nchar(name)
          left_pad <- max(0L, floor(padding / 2))
          right_pad <- max(0L, ceiling(padding / 2))
          paste0(strrep(" ", left_pad), name, strrep(" ", right_pad))
        }, col_names, col_widths),
        collapse = " | "
      ),
      " |"
    )

    # Separator
    separator <- paste0(
      "|",
      paste0(sapply(col_widths, function(w) strrep("-", w + 2L)), collapse = "|"),
      "|"
    )

    # Data rows
    rows <- vapply(seq_len(nrow(data_mat)), function(i) {
      cells <- vapply(seq_along(col_names), function(j) {
        cell <- data_mat[i, j]
        if (is.na(cell)) {
          formatC("", width = col_widths[j], format = "s", flag = " ")
        } else if (j == 1L) {
          formatC(cell, width = col_widths[j], format = "s", flag = "-")
        } else {
          formatC(cell, width = col_widths[j], format = "s", flag = " ")
        }
      }, character(1))
      paste0("| ", paste(cells, collapse = " | "), " |")
    }, character(1))

    trailer <- if (d$n_shown < d$total) {
      sprintf("[showing first %d of %d levels]", d$n_shown, d$total)
    } else {
      character(0)
    }

    lines <- c(sprintf("FE: %s", d$dim), "", header, separator, rows, trailer)
    paste(lines, collapse = "\n")
  })

  obj <- list(content = paste(sections, collapse = "\n\n"), type = "console")
  class(obj) <- "summary_table"
  obj
}

fe_format_latex <- function(dim_data, model_names, caption, label, position) {
  n_models <- length(model_names)
  n_cols <- 1L + n_models
  col_spec <- paste0("l", strrep("c", n_models))

  read_tmpl <- function(name) {
    path <- system.file("templates", name, package = "capybara", mustWork = TRUE)
    paste(readLines(path, warn = FALSE), collapse = "\n")
  }

  fill_tmpl <- function(tmpl, vals) {
    for (key in names(vals)) {
      tmpl <- gsub(paste0("{{", key, "}}"), vals[[key]], tmpl, fixed = TRUE)
    }
    tmpl
  }

  tmpl <- list(
    table       = read_tmpl("latex_table.tex"),
    row         = read_tmpl("row.tex"),
    midrule_row = read_tmpl("midrule_row.tex"),
    caption     = read_tmpl("caption.tex"),
    label       = read_tmpl("label.tex")
  )

  make_row <- function(cols) {
    fill_tmpl(tmpl$row, list(cols = paste(cols, collapse = " & ")))
  }
  make_midrule_row <- function(cols) {
    fill_tmpl(tmpl$midrule_row, list(cols = paste(cols, collapse = " & ")))
  }

  # Column header row
  body <- c(make_row(c("Level", model_names)), "\\midrule")

  # One section per FE dimension
  for (d in dim_data) {
    dim_esc <- gsub("_", "\\_", d$dim, fixed = TRUE)
    section_header <- c(paste0("\\textit{", dim_esc, "}"), rep("", n_models))
    body <- c(body, make_midrule_row(section_header))

    for (i in seq_len(d$n_shown)) {
      lvl_esc <- gsub("_", "\\_", d$levels[i], fixed = TRUE)
      row_vals <- c(lvl_esc, ifelse(is.na(d$mat[i, ]), "", d$mat[i, ]))
      body <- c(body, make_row(row_vals))
    }

    if (d$n_shown < d$total) {
      note <- paste0(
        "\\multicolumn{", n_cols, "}{l}{\\footnotesize ",
        sprintf("showing first %d of %d levels", d$n_shown, d$total),
        "} \\\\"
      )
      body <- c(body, note)
    }
  }

  body_text <- paste(body, collapse = "\n")

  in_quarto_tbl <- nzchar(Sys.getenv("QUARTO_BIN_PATH")) &&
    isTRUE(getOption("knitr.in.progress")) &&
    tryCatch(
      {
        lbl <- knitr::opts_current$get("label")
        !is.null(lbl) && nzchar(lbl) && grepl("^tbl-", lbl)
      },
      error = function(e) FALSE
    )

  content <- if (in_quarto_tbl) {
    paste0(
      "\\centering\n",
      "\\begin{tabular}{", col_spec, "}\n",
      "\\toprule\n",
      body_text, "\n",
      "\\bottomrule\n",
      "\\end{tabular}"
    )
  } else {
    caption_text <- if (is.null(caption)) {
      ""
    } else {
      fill_tmpl(tmpl$caption, list(caption = caption))
    }
    label_text <- if (is.null(label) || !nzchar(label)) {
      ""
    } else {
      fill_tmpl(tmpl$label, list(label = label))
    }
    fill_tmpl(tmpl$table, list(
      position  = position,
      caption   = caption_text,
      label     = label_text,
      col_spec  = col_spec,
      body      = body_text,
      footnotes = ""
    ))
  }

  obj <- list(content = content, type = "latex")
  class(obj) <- "summary_table"
  obj
}
