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
#' @examples
#' m1 <- felm(mpg ~ wt | cyl, mtcars)
#' m2 <- fepoisson(mpg ~ wt | cyl, mtcars)
#' summary_table(m1, m2, model_names = c("Linear", "Poisson"))
#' @return A formatted table
#' @export
summary_table <- function(...,
                          coef_digits = 3,
                          se_digits = 3,
                          stars = TRUE,
                          latex = FALSE,
                          model_names = NULL,
                          caption = NULL,
                          label = NULL) {
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
    if (!is.null(m$coef_table)) {
      as.vector(m$coef_table[, 1])
    } else {
      as.vector(m$coefficients)
    }
  })
  
  se_list <- lapply(models, function(m) {
    if (!is.null(m$coef_table)) {
      as.vector(m$coef_table[, 2])
    } else {
      as.vector(sqrt(diag(m$vcov)))
    }
  })
  
  p_list <- lapply(models, function(m) {
    if (!is.null(m$coef_table)) {
      as.vector(m$coef_table[, 4])
    } else {
      # Calculate p-values from coefficients and standard errors
      z <- m$coefficients / sqrt(diag(m$vcov))
      as.vector(2 * pnorm(-abs(z)))
    }
  })

  # Set names for the lists
  for (i in seq_along(coef_list)) {
    var_names <- names(models[[i]]$coefficients)
    names(coef_list[[i]]) <- var_names
    names(se_list[[i]]) <- var_names
    names(p_list[[i]]) <- var_names
  }

  # Get all unique variable names across models
  all_vars <- unique(unlist(lapply(models, function(m) {
    names(m$coefficients)
  })))

  # Create a data frame for the results
  result_df <- data.frame(
    Variable = all_vars,
    stringsAsFactors = FALSE
  )

  # Format coefficients for each model
  invisible(lapply(seq_along(models), function(i) {
    model_col <- vapply(all_vars, function(var) {
      if (var %in% names(coef_list[[i]])) {
        coef_val <- formatC(coef_list[[i]][var], digits = coef_digits, format = "f")
        se_val <- formatC(se_list[[i]][var], digits = se_digits, format = "f")

        if (stars) {
          p_val <- p_list[[i]][var]
          star <- ""
          if (p_val < 0.01) {
            star <- "**"
          } else if (p_val < 0.05) {
            star <- "*"
          } else if (p_val < 0.1) star <- "+"

          sprintf("%s%s\n(%s)", coef_val, star, se_val)
        } else {
          sprintf("%s\n(%s)", coef_val, se_val)
        }
      } else {
        NA_character_
      }
    }, character(1))

    result_df[[model_names[i]]] <<- model_col
  }))

  # Fixed effects
  fe_rows <- list()
  fe_names <- unique(unlist(lapply(models, function(m) {
    if (!is.null(m$nms_fe)) names(m$nms_fe) else NULL
  })))

  if (length(fe_names) > 0) {
    fe_rows <- setNames(lapply(fe_names, function(fe) {
      c(fe, sapply(models, function(m) {
        if (!is.null(m$nms_fe) && fe %in% names(m$nms_fe)) "Yes" else "No"
      }))
    }), fe_names)
  }

  # Add model statistics
  stats_rows <- list()

  obs_row <- c("N", sapply(models, function(m) {
    if (inherits(m, "felm")) {
      format(as.numeric(m$nobs["nobs_full"]), big.mark = ",")
    } else {
      if (is.vector(m$nobs) && length(m$nobs) > 1) {
        if ("nobs" %in% names(m$nobs)) {
          format(as.numeric(m$nobs["nobs"]), big.mark = ",")
        } else {
          format(as.numeric(m$nobs[1]), big.mark = ",")
        }
      } else {
        format(as.numeric(m$nobs), big.mark = ",")
      }
    }
  }))

  # Check if any model is a GLM (uses pseudo R-squared)

  has_glm <- any(sapply(models, function(m) inherits(m, "feglm")))
  r2_label <- if (has_glm) {
    if (latex) "Pseudo $R^2$" else "Pseudo R-squared"
  } else {
    if (latex) "$R^2$" else "R-squared"
  }

  r2_row <- c(r2_label, sapply(models, function(m) {
    if (inherits(m, "felm")) {
      formatC(m$r.squared, digits = 3, format = "f")
    } else if (inherits(m, "feglm") && !is.null(m$pseudo.rsq)) {
      formatC(m$pseudo.rsq, digits = 3, format = "f")
    } else {
      ""
    }
  }))

  # Output in the requested format

  result2_df <- rbind(
    c("", rep("", length(models))), # for spacing
    c("Fixed effects ", rep("", length(models))), # for spacing
    do.call(rbind, fe_rows),
    c("", rep("", length(models))), # for spacing
    obs_row,
    r2_row
  )

  colnames(result2_df) <- colnames(result_df)

  # Format the output and return it directly (no print call)
  res <- if (latex) {
    format_latex_table(result_df, result2_df, stars, label, caption)
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
    max_width <- Reduce(function(acc, i) {
      if (!is.na(col[i]) && grepl("\n", col[i])) {
        parts <- strsplit(col[i], "\n")[[1]]
        max(acc, max(nchar(parts)))
      } else {
        acc
      }
    }, seq_along(col), init = max(nchar(col), na.rm = TRUE))
    max_width + 2 # Add padding
  })

  # Create header with center alignment
  header_names <- as.character(colnames(result_df))
  header <- paste0("| ", paste0(
    mapply(function(name, width) {
      # Center align header text
      name <- as.character(name) # Ensure it's a simple string
      padding <- width - nchar(name)
      left_pad <- floor(padding / 2)
      right_pad <- ceiling(padding / 2)
      paste0(
        paste(rep(" ", left_pad), collapse = ""),
        name,
        paste(rep(" ", right_pad), collapse = "")
      )
    }, header_names, col_widths),
    collapse = " | "
  ), " |")

  # Create separator with proper width
  separator <- paste0("|", paste0(
    mapply(function(width) {
      paste0(rep("-", width + 2), collapse = "")
    }, col_widths),
    collapse = "|"
  ), "|")
  separator <- gsub(" ", "-", separator)

  # Generate table rows
  table_rows <- unlist(lapply(1:nrow(full_df), function(i) {
    # First detect how many lines we need for this row
    lines_needed <- max(1L, vapply(1:ncol(full_df), function(j) {
      cell <- full_df[i, j]
      if (!is.na(cell) && grepl("\n", cell)) {
        length(strsplit(cell, "\n")[[1]])
      } else {
        1L
      }
    }, integer(1)))

    # Create an array to hold all lines for this table row
    vapply(1:lines_needed, function(line) {
      line_cells <- vapply(1:ncol(full_df), function(j) {
        cell <- full_df[i, j]

        if (is.na(cell)) {
          formatC("", width = col_widths[j], format = "s", flag = " ")
        } else if (grepl("\n", cell)) {
          # Split multi-line cells
          parts <- strsplit(cell, "\n")[[1]]

          if (line <= length(parts)) {
            # This line exists in the cell
            if (j == 1) {
              # Left align first column
              formatC(parts[line], width = col_widths[j], format = "s", flag = "-")
            } else {
              # Right align other columns
              formatC(parts[line], width = col_widths[j], format = "s", flag = " ")
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
      }, character(1))

      # Assemble this line
      paste0("| ", paste(line_cells, collapse = " | "), " |")
    }, character(1))
  }))

  # Add legend
  if (stars) {
    legend <- "\nStandard errors in parenthesis\nSignificance levels: ** p < 0.01; * p < 0.05; + p < 0.10"
  } else {
    legend <- ""
  }

  # Create content and metadata separately
  table_content <- paste(c(header, separator, table_rows, legend), collapse = "\n")

  # Create a new S3 object with better structure
  obj <- list(
    content = table_content,
    type = "console"
  )
  class(obj) <- "summary_table"

  obj
}

# LaTeX formatter
format_latex_table <- function(result_df, result2_df, stars, label = NULL,
                               caption = NULL) {
  # Convert to data frame
  full_df <- rbind(as.matrix(result_df), result2_df)

  # Create LaTeX code
  n_cols <- ncol(full_df)

  # Start with empty vector for LaTeX code
  latex <- character(0)

  # Include table environment if label or caption is provided
  include_environment <- !is.null(label) || !is.null(caption)
  if (include_environment) {
    latex <- c(latex, "\\begin{table}[htbp]", "\\centering")
    if (!is.null(caption)) {
      latex <- c(latex, paste0("\\caption{", caption, "}"))
    }
    if (!is.null(label)) {
      latex <- c(latex, paste0("\\label{", label, "}"))
    }
  }

  # Add tabular environment
  latex <- c(
    latex,
    paste0("\\begin{tabular}{l", paste(rep("c", n_cols - 1), collapse = ""), "}"),
    "\\toprule"
  )

  # Header row
  latex <- c(latex, paste(colnames(full_df), collapse = " & "), "\\\\")

  # Midrule
  latex <- c(latex, "\\midrule")

  # Process coefficients
  latex_rows <- unlist(lapply(1:nrow(result_df), function(i) {
    row <- result_df[i, ]

    # First process variable name (escape underscores for LaTeX)
    var_name <- as.character(row[1])
    var_name <- gsub("_", "\\_", var_name, fixed = TRUE)

    # Create coefficient row
    coef_values <- character(ncol(row))
    coef_values[1] <- var_name # Variable name

    # Create SE row (empty in first column)
    se_values <- character(ncol(row))
    se_values[1] <- "" # Empty first column

    # Fill in coefficient and SE values for each model
    cell_results <- lapply(2:ncol(row), function(j) {
      cell <- row[j]

      if (is.na(cell) || cell == "") {
        list(coef = "", se = "")
      } else if (grepl("\n", cell)) {
        # Split into coef and SE
        parts <- strsplit(as.character(cell), "\n")[[1]]
        # Convert stars to LaTeX superscripts
        # Only replace stars at the end of the string
        coef_with_stars <- parts[1]
        if (grepl("\\*\\*$", coef_with_stars)) {
          coef_with_stars <- sub("\\*\\*$", "$^{**}$", coef_with_stars)
        } else if (grepl("\\*$", coef_with_stars)) {
          coef_with_stars <- sub("\\*$", "$^{*}$", coef_with_stars)
        } else if (grepl("\\+$", coef_with_stars)) {
          coef_with_stars <- sub("\\+$", "$^{+}$", coef_with_stars)
        }
        list(coef = coef_with_stars, se = parts[2])
      } else {
        list(coef = as.character(cell), se = "")
      }
    })

    coef_values[2:ncol(row)] <- vapply(cell_results, function(x) x$coef, character(1))
    se_values[2:ncol(row)] <- vapply(cell_results, function(x) x$se, character(1))

    # Return coefficient row and SE row (if it has content)
    result <- c(paste(coef_values, collapse = " & "), "\\\\")
    if (any(nchar(se_values) > 0)) {
      result <- c(result, paste(se_values, collapse = " & "), "\\\\")
    }
    result
  }))
  
  latex <- c(latex, latex_rows)

  # Midrule before stats
  # latex <- c(latex, "\\midrule")

  # Add stat rows (escape underscores for LaTeX)
  stat_rows <- apply(result2_df, 1, function(row) {
    if (all(row == "")) {
      return(NULL)
    }
    row <- gsub("_", "\\_", row, fixed = TRUE)
    row_text <- paste(ifelse(is.na(row), "", row), collapse = " & ")
    paste0(row_text, " \\\\")
  })
  latex <- c(latex, stat_rows[!sapply(stat_rows, is.null)])

  latex <- gsub(
    "^Fixed effects ",
    "\\\\midrule Fixed effects ", latex
  )

  latex <- gsub(
    "^N ",
    "\\\\midrule N ", latex
  )

  # Table footer
  latex <- c(latex, "\\bottomrule")

  # Add significance legend
  if (stars) {
    latex <- c(latex, paste0(
      "\\multicolumn{", n_cols, "}{l}{\\footnotesize Standard errors in parentheses} \\\\ \n",
      "\\multicolumn{", n_cols, "}{l}{\\footnotesize Significance levels: ",
      "$**\\: p < 0.01;\\: *\\: p < 0.05;\\: +\\: p < 0.10$}"
    ))
  }

  # Close environments
  latex <- c(latex, "\\end{tabular}")

  if (include_environment) {
    latex <- c(latex, "\\end{table}")
  }

  # Create a new S3 object with better structure
  obj <- list(
    content = paste(latex, collapse = "\n"),
    type = "latex"
  )
  class(obj) <- "summary_table"
  obj
}
