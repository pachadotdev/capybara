#' Generate formatted regression tables
#'
#' @param ... One or more model objects of \code{felm} or \code{feglm} class.
#' @param coef_digits Number of digits for coefficients. The default is 3.
#' @param se_digits Number of digits for standard errors. The default is 3.
#' @param stars Whether to include significance stars. The default is \code{TRUE}.
#' @param latex Whether to output as LaTeX code. The default is \code{FALSE}.
#' @param model_names Optional vector of custom model names
#' @param caption Optional caption for the table (LaTeX only)
#' @param label Optional label for cross-referencing (LaTeX only)
#' @param position LaTeX float position specifier (LaTeX only). The default is
#'   \code{"htbp"}.
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999 &
#'   ross2004$ltrade > quantile(ross2004$ltrade, 0.75), ]
#'
#' m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
#' m2 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)
#'
#' summary_table(m1, m2, model_names = c("Linear", "Poisson"))
#' @return A formatted table
#' @export
summary_table <- function(
  ...,
  coef_digits = 3,
  se_digits = 3,
  stars = TRUE,
  latex = FALSE,
  model_names = NULL,
  caption = NULL,
  label = NULL,
  position = "htbp"
) {
  # Collect models
  models <- list(...)

  # Check that all models are felm or feglm
  valid_classes <- c("felm", "feglm")
  invisible(lapply(seq_along(models), function(i) {
    if (!inherits(models[[i]], valid_classes)) {
      stop("Model ", i, " is not a felm or feglm object")
    }
  }))

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

  # Extract coefficients and standard errors
  # vcov is precomputed during fitting (either inverse Hessian or sandwich)
  # Use coef_table which is pre-computed in the model object
  coef_list <- lapply(models, function(m) {
    ct <- m$coef_table
    setNames(as.vector(ct[, 1]), rownames(ct))
  })

  se_list <- lapply(models, function(m) {
    ct <- m$coef_table
    setNames(as.vector(ct[, 2]), rownames(ct))
  })

  p_list <- lapply(models, function(m) {
    ct <- m$coef_table
    setNames(as.vector(ct[, 4]), rownames(ct))
  })

  # Get all unique variable names across models
  all_vars <- unique(unlist(lapply(models, function(m) {
    rownames(m$coef_table)
  })))

  # Create a data frame for the results
  result_df <- data.frame(
    Variable = all_vars,
    stringsAsFactors = FALSE
  )

  # Format coefficients for each model
  invisible(lapply(seq_along(models), function(i) {
    coefs <- coef_list[[i]]
    ses <- se_list[[i]]
    pvals <- p_list[[i]]

    model_col <- vapply(
      all_vars,
      function(var) {
        if (var %in% names(coefs)) {
          coef_raw <- coefs[var]
          se_raw <- ses[var]
          p_val <- pvals[var]

          # Handle NA coefficients
          if (is.na(coef_raw)) {
            return("NA")
          }

          coef_val <- formatC(coef_raw, digits = coef_digits, format = "f")
          se_val <- formatC(se_raw, digits = se_digits, format = "f")

          if (stars && !is.na(p_val)) {
            star <- ""
            if (p_val < 0.01) {
              star <- "**"
            } else if (p_val < 0.05) {
              star <- "*"
            } else if (p_val < 0.1) {
              star <- "+"
            }

            sprintf("%s%s\n(%s)", coef_val, star, se_val)
          } else {
            sprintf("%s\n(%s)", coef_val, se_val)
          }
        } else {
          NA_character_
        }
      },
      character(1)
    )

    result_df[[model_names[i]]] <<- model_col
  }))

  # Fixed effects
  fe_rows <- list()
  fe_names <- unique(unlist(lapply(models, function(m) {
    if (!is.null(m$nms_fe)) names(m$nms_fe) else NULL
  })))

  if (length(fe_names) > 0) {
    fe_rows <- setNames(
      lapply(fe_names, function(fe) {
        c(
          fe,
          sapply(models, function(m) {
            if (!is.null(m$nms_fe) && fe %in% names(m$nms_fe)) "Yes" else "No"
          })
        )
      }),
      fe_names
    )
  }

  # SE type row (from vcov_type stored on each model)
  vcov_label_map <- c(
    "iid" = "IID",
    "hetero" = "Heteroskedastic-robust",
    "cluster" = "Cluster-robust",
    "m-estimator" = "Cluster-robust (M-est.)",
    "m-estimator-dyadic" = "Dyadic-robust",
    "dyadic" = "Dyadic-robust"
  )
  se_type_row <- c(
    "SE type",
    sapply(models, function(m) {
      vt <- m[["vcov_type"]]
      if (is.null(vt)) {
        ""
      } else {
        lbl <- vcov_label_map[vt]
        if (is.na(lbl)) vt else lbl
      }
    })
  )

  # Add model statistics
  stats_rows <- list()

  obs_row <- c(
    "N",
    sapply(models, function(m) {
      nobs_vec <- m$nobs
      if (inherits(m, "felm")) {
        format(as.numeric(nobs_vec["nobs_full"]), big.mark = ",")
      } else {
        if (is.vector(nobs_vec) && length(nobs_vec) > 1) {
          if ("nobs" %in% names(nobs_vec)) {
            format(as.numeric(nobs_vec["nobs"]), big.mark = ",")
          } else {
            format(as.numeric(nobs_vec[1]), big.mark = ",")
          }
        } else {
          format(as.numeric(nobs_vec), big.mark = ",")
        }
      }
    })
  )

  # Check if any model is a GLM

  r2_label <- ifelse(latex, "$R^2$", "R-squared")

  r2_row <- c(
    r2_label,
    sapply(models, function(m) {
      if (!is.null(m$r_squared)) {
        formatC(m$r_squared, digits = 3, format = "f")
      } else {
        ""
      }
    })
  )

  # Output in the requested format

  # Only include r2_row / se_type_row when at least one model has a value
  stat_rows_optional <- list()
  if (any(r2_row[-1] != "")) stat_rows_optional <- c(stat_rows_optional, list(r2_row))
  if (any(se_type_row[-1] != "")) stat_rows_optional <- c(stat_rows_optional, list(se_type_row))

  result2_df <- rbind(
    c("", rep("", length(models))), # for spacing
    c("Fixed effects ", rep("", length(models))), # for spacing
    do.call(rbind, fe_rows),
    c("", rep("", length(models))), # for spacing
    obs_row,
    if (length(stat_rows_optional) > 0) do.call(rbind, stat_rows_optional) else NULL
  )

  # Set column names from result_df
  col_names <- colnames(result_df)
  colnames(result2_df) <- col_names

  # Format the output and return it directly (no print call)
  res <- if (latex) {
    format_latex_table(result_df, result2_df, stars, label, caption, position)
  } else {
    format_console_table(result_df, result2_df, stars)
  }

  res
}

# Console formatter for clean ascii tables
format_console_table <- function(result_df, result2_df, stars) {
  # Convert to data frame and ensure column names are preserved
  full_df <- rbind(as.matrix(result_df), result2_df)
  colnames(full_df) <- colnames(result_df) # Make sure column names are properly set

  # Calculate column widths for proper alignment
  col_widths <- apply(full_df, 2, function(col) {
    # Split coefficient/SE pairs and find max width
    max_width <- Reduce(
      function(acc, i) {
        cell <- as.character(col[i])
        if (!is.na(cell) && grepl("\n", cell)) {
          parts <- strsplit(cell, "\n")[[1]]
          max(acc, max(nchar(parts)))
        } else {
          acc
        }
      },
      seq_along(col),
      init = max(nchar(as.character(col)), na.rm = TRUE)
    )
    max_width + 2 # Add padding
  })

  # Create header with center alignment
  header_names <- as.character(colnames(result_df))
  header <- paste0(
    "| ",
    paste0(
      mapply(
        function(name, width) {
          # Center align header text
          name <- as.character(name) # Ensure it's a simple string
          padding <- width - nchar(name)
          left_pad <- max(0L, floor(padding / 2))
          right_pad <- max(0L, ceiling(padding / 2))
          paste0(
            paste(rep(" ", left_pad), collapse = ""),
            name,
            paste(rep(" ", right_pad), collapse = "")
          )
        },
        header_names,
        col_widths
      ),
      collapse = " | "
    ),
    " |"
  )

  # Create separator with proper width
  separator <- paste0(
    "|",
    paste0(
      mapply(
        function(width) {
          paste0(rep("-", width + 2), collapse = "")
        },
        col_widths
      ),
      collapse = "|"
    ),
    "|"
  )
  separator <- gsub(" ", "-", separator)

  # Generate table rows
  table_rows <- unlist(lapply(1:nrow(full_df), function(i) {
    # First detect how many lines we need for this row
    lines_needed <- max(
      1L,
      vapply(
        1:ncol(full_df),
        function(j) {
          cell <- as.character(full_df[i, j])
          if (!is.na(cell) && grepl("\n", cell)) {
            length(strsplit(cell, "\n")[[1]])
          } else {
            1L
          }
        },
        integer(1)
      )
    )

    # Create an array to hold all lines for this table row
    vapply(
      1:lines_needed,
      function(line) {
        line_cells <- vapply(
          1:ncol(full_df),
          function(j) {
            cell <- as.character(full_df[i, j])

            if (is.na(cell)) {
              formatC("", width = col_widths[j], format = "s", flag = " ")
            } else if (grepl("\n", cell)) {
              # Split multi-line cells
              parts <- strsplit(cell, "\n")[[1]]

              if (line <= length(parts)) {
                # This line exists in the cell
                if (j == 1) {
                  # Left align first column
                  formatC(
                    parts[line],
                    width = col_widths[j],
                    format = "s",
                    flag = "-"
                  )
                } else {
                  # Right align other columns
                  formatC(
                    parts[line],
                    width = col_widths[j],
                    format = "s",
                    flag = " "
                  )
                }
              } else {
                # Fill with empty space
                formatC("", width = col_widths[j], format = "s", flag = " ")
              }
            } else {
              # Single line cell - only show on first line
              if (line == 1) {
                if (j == 1) {
                  # Left align first column
                  formatC(cell, width = col_widths[j], format = "s", flag = "-")
                } else {
                  # Right align other columns
                  formatC(cell, width = col_widths[j], format = "s", flag = " ")
                }
              } else {
                formatC("", width = col_widths[j], format = "s", flag = " ")
              }
            }
          },
          character(1)
        )

        # Assemble this line
        paste0("| ", paste(line_cells, collapse = " | "), " |")
      },
      character(1)
    )
  }))

  # Add legend
  if (stars) {
    legend <- "\nStandard errors in parenthesis\nSignificance levels: ** p < 0.01; * p < 0.05; + p < 0.10"
  } else {
    legend <- ""
  }

  # Create content and metadata separately
  table_content <- paste(
    c(header, separator, table_rows, legend),
    collapse = "\n"
  )

  # Create a new S3 object with better structure
  obj <- list(
    content = table_content,
    type = "console"
  )
  class(obj) <- "summary_table"

  obj
}

# LaTeX formatter
format_latex_table <- function(
  result_df,
  result2_df,
  stars,
  label = NULL,
  caption = NULL,
  position = "htbp"
) {
  full_df <- rbind(as.matrix(result_df), result2_df)
  n_cols <- ncol(full_df)
  col_spec <- paste0("l", paste(rep("c", n_cols - 1), collapse = ""))

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

  # Read all templates once
  tmpl <- list(
    table       = read_tmpl("latex_table.tex"),
    row         = read_tmpl("row.tex"),
    midrule_row = read_tmpl("midrule_row.tex"),
    footnotes   = read_tmpl("footnotes.tex"),
    caption     = read_tmpl("caption.tex"),
    label       = read_tmpl("label.tex")
  )

  make_row <- function(cols) fill_tmpl(tmpl$row, list(cols = paste(cols, collapse = " & ")))
  make_midrule_row <- function(cols) fill_tmpl(tmpl$midrule_row, list(cols = paste(cols, collapse = " & ")))

  # --- Build body (header + coefficient rows + stat rows) ---

  body <- c(make_row(colnames(full_df)), "\\midrule")

  # Coefficient rows
  coef_rows <- unlist(lapply(seq_len(nrow(result_df)), function(i) {
    row <- result_df[i, ]
    var_name <- gsub("_", "\\_", as.character(row[1]), fixed = TRUE)

    coef_values <- character(ncol(row))
    coef_values[1] <- var_name
    se_values <- character(ncol(row))
    se_values[1] <- ""

    cell_results <- lapply(2:ncol(row), function(j) {
      cell <- as.character(row[j])
      if (is.na(cell) || cell == "") {
        list(coef = "", se = "")
      } else if (grepl("\n", cell)) {
        parts <- strsplit(cell, "\n")[[1]]
        coef_with_stars <- parts[1]
        if (grepl("\\*\\*$", coef_with_stars)) {
          coef_with_stars <- sub("\\*\\*$", "$^{**}$", coef_with_stars)
        } else if (grepl("\\*$", coef_with_stars)) {
          coef_with_stars <- sub("\\*$", "$^{*}$", coef_with_stars)
        } else if (grepl("\\+$", coef_with_stars)) coef_with_stars <- sub("\\+$", "$^{+}$", coef_with_stars)
        list(coef = coef_with_stars, se = parts[2])
      } else {
        list(coef = as.character(cell), se = "")
      }
    })

    coef_values[2:ncol(row)] <- vapply(cell_results, function(x) x$coef, character(1))
    se_values[2:ncol(row)] <- vapply(cell_results, function(x) x$se, character(1))

    out <- make_row(coef_values)
    if (any(nchar(se_values) > 0)) out <- c(out, make_row(se_values))
    out
  }))

  body <- c(body, coef_rows)

  # Stat rows — midrule_row for section headers, plain row otherwise
  midrule_triggers <- c("Fixed effects", "N")
  stat_rows <- apply(result2_df, 1, function(row) {
    if (all(row == "")) {
      return(NULL)
    }
    row_esc <- gsub("_", "\\_", row, fixed = TRUE)
    cols <- ifelse(is.na(row_esc), "", row_esc)
    if (trimws(row[1]) %in% midrule_triggers) make_midrule_row(cols) else make_row(cols)
  })
  body <- c(body, stat_rows[!vapply(stat_rows, is.null, logical(1))])

  body_text <- paste(body, collapse = "\n")

  # --- Fill main template ---
  # When inside a Quarto chunk with a tbl- label, Quarto itself creates the
  # \begin{table}\caption{...}\label{tbl-xxx}...\end{table} wrapper (driven by
  # the tbl-cap and label chunk options).  If we also emit a full table float,
  # the result is two nested floats: an empty "Table N" from Quarto and the real
  # "Table N+1: Caption" from us.  In that context we must output only the inner
  # tabular content and let Quarto supply the outer environment.
  in_quarto_tbl <- nzchar(Sys.getenv("QUARTO_BIN_PATH")) &&
    isTRUE(getOption("knitr.in.progress")) &&
    tryCatch(
      {
        lbl <- knitr::opts_current$get("label")
        !is.null(lbl) && nzchar(lbl) && grepl("^tbl-", lbl)
      },
      error = function(e) FALSE
    )

  footnotes_text <- if (stars) fill_tmpl(tmpl$footnotes, list(n_cols = n_cols)) else ""

  content <- if (in_quarto_tbl) {
    # Emit only the tabular — Quarto wraps it with caption, label, and position.
    paste0(
      "\\centering\n",
      "\\begin{tabular}{", col_spec, "}\n",
      "\\toprule\n",
      body_text, "\n",
      "\\bottomrule\n",
      if (nzchar(footnotes_text)) paste0(footnotes_text, "\n") else "",
      "\\end{tabular}"
    )
  } else {
    caption_text <- if (is.null(caption)) "" else fill_tmpl(tmpl$caption, list(caption = caption))
    label_text <- if (is.null(label) || !nzchar(label)) "" else fill_tmpl(tmpl$label, list(label = label))
    fill_tmpl(tmpl$table, list(
      position  = position,
      caption   = caption_text,
      label     = label_text,
      col_spec  = col_spec,
      body      = body_text,
      footnotes = footnotes_text
    ))
  }

  obj <- list(content = content, type = "latex")
  class(obj) <- "summary_table"
  obj
}
