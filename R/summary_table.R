#' Generate formatted regression tables
#'
#' @param ... One or more model objects of \code{felm} or \code{feglm} class.
#' @param coef_digits Number of digits for coefficients. The default is 3.
#' @param se_digits Number of digits for standard errors. The default is 3.
#' @param stars Whether to include significance stars. The default is \code{TRUE}.
#' @param latex Whether to output as LaTeX code. The default is \code{FALSE}.
#' @param model_names Optional vector of custom model names
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
                          model_names = NULL) {
  # Collect models
  models <- list(...)

  # Check that all models are felm or feglm
  valid_classes <- c("felm", "feglm")
  for (i in seq_along(models)) {
    if (!inherits(models[[i]], valid_classes)) {
      stop("Model ", i, " is not a felm or feglm object")
    }
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

  # Extract coefficients and standard errors
  summaries <- lapply(models, summary)
  coef_list <- lapply(summaries, function(m) m$coefficients[, 1])
  se_list <- lapply(summaries, function(m) m$coefficients[, 2])
  p_list <- lapply(summaries, function(m) m$coefficients[, 4])

  for (i in seq_along(coef_list)) {
    names(coef_list[[i]]) <- rownames(summaries[[i]]$coefficients)
    names(se_list[[i]]) <- rownames(summaries[[i]]$coefficients)
    names(p_list[[i]]) <- rownames(summaries[[i]]$coefficients)
  }

  # Get all unique variable names across models
  all_vars <- unique(unlist(lapply(summaries, function(m) rownames(m$coefficients))))

  # Create a data frame for the results
  result_df <- data.frame(
    Variable = all_vars,
    stringsAsFactors = FALSE
  )

  # Format coefficients for each model
  for (i in seq_along(models)) {
    model_col <- rep(NA_character_, nrow(result_df))

    for (j in seq_along(all_vars)) {
      var <- all_vars[j]
      if (var %in% names(coef_list[[i]])) {
        coef_val <- formatC(coef_list[[i]][var], digits = coef_digits, format = "f")
        se_val <- formatC(se_list[[i]][var], digits = se_digits, format = "f")

        if (stars) {
          p_val <- p_list[[i]][var]
          star <- ""
          if (p_val < 0.001) {
            star <- "***"
          } else if (p_val < 0.01) {
            star <- "**"
          } else if (p_val < 0.05) {
            star <- "*"
          } else if (p_val < 0.1) star <- "."

          model_col[j] <- sprintf("%s%s\n(%s)", coef_val, star, se_val)
        } else {
          model_col[j] <- sprintf("%s\n(%s)", coef_val, se_val)
        }
      }
    }

    result_df[[model_names[i]]] <- model_col
  }

  # Fixed effects
  fe_rows <- list()
  fe_names <- unique(unlist(lapply(models, function(m) {
    if (!is.null(m$nms_fe)) names(m$nms_fe) else NULL
  })))

  if (length(fe_names) > 0) {
    for (fe in fe_names) {
      fe_row <- c(fe, sapply(models, function(m) {
        if (!is.null(m$nms_fe) && fe %in% names(m$nms_fe)) "Yes" else "No"
      }))
      fe_rows[[fe]] <- fe_row
    }
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

  r2_row <- c(if (latex) "$R^2$" else "R-squared", sapply(models, function(m) {
    if (inherits(m, "felm")) {
      formatC(summary(m)$r.squared, digits = 3, format = "f")
    } else if (inherits(m, "feglm") && !is.null(summary(m)$pseudo.rsq)) {
      formatC(summary(m)$pseudo.rsq, digits = 3, format = "f")
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
    format_latex_table(result_df, result2_df, stars)
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
    max_width <- max(nchar(col), na.rm = TRUE)
    # Account for newlines by checking each part
    for (i in seq_along(col)) {
      if (!is.na(col[i]) && grepl("\n", col[i])) {
        parts <- strsplit(col[i], "\n")[[1]]
        max_width <- max(max_width, max(nchar(parts)))
      }
    }
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
  table_rows <- character(0)

  for (i in 1:nrow(full_df)) {
    # First detect how many lines we need for this row
    lines_needed <- 1 # At least 1 line per row
    for (j in 1:ncol(full_df)) {
      cell <- full_df[i, j]
      if (!is.na(cell) && grepl("\n", cell)) {
        parts <- strsplit(cell, "\n")[[1]]
        lines_needed <- max(lines_needed, length(parts))
      }
    }

    # Create an array to hold all lines for this table row
    row_lines <- character(lines_needed)

    # For each column, fill the appropriate lines
    for (line in 1:lines_needed) {
      line_cells <- character(ncol(full_df))

      for (j in 1:ncol(full_df)) {
        cell <- full_df[i, j]

        if (is.na(cell)) {
          cell_text <- formatC("", width = col_widths[j], format = "s", flag = " ")
        } else if (grepl("\n", cell)) {
          # Split multi-line cells
          parts <- strsplit(cell, "\n")[[1]]

          if (line <= length(parts)) {
            # This line exists in the cell
            if (j == 1) {
              # Left align first column
              cell_text <- formatC(parts[line], width = col_widths[j], format = "s", flag = "-")
            } else {
              # Right align other columns
              cell_text <- formatC(parts[line], width = col_widths[j], format = "s", flag = " ")
            }
          } else {
            # Fill with empty space
            cell_text <- formatC("", width = col_widths[j], format = "s", flag = " ")
          }
        } else {
          # Single line cell - only show on first line
          if (line == 1) {
            if (j == 1) {
              # Left align first column
              cell_text <- formatC(cell, width = col_widths[j], format = "s", flag = "-")
            } else {
              # Right align other columns
              cell_text <- formatC(cell, width = col_widths[j], format = "s", flag = " ")
            }
          } else {
            cell_text <- formatC("", width = col_widths[j], format = "s", flag = " ")
          }
        }

        line_cells[j] <- cell_text
      }

      # Assemble this line
      row_lines[line] <- paste0("| ", paste(line_cells, collapse = " | "), " |")
    }

    # Combine all lines for this row
    table_rows <- c(table_rows, row_lines)
  }

  # Add legend
  if (stars) {
    legend <- "\nStandard errors in parenthesis\nSignificance levels: *** p < 0.001; ** p < 0.01; * p < 0.05; . p < 0.1"
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
format_latex_table <- function(result_df, result2_df, stars, include_environment = FALSE) {
  # Convert to data frame
  full_df <- rbind(as.matrix(result_df), result2_df)

  # Create LaTeX code
  n_cols <- ncol(full_df)

  # Start with empty vector for LaTeX code
  latex <- character(0)

  # Only include table environment if requested
  if (include_environment) {
    latex <- c(
      latex,
      "\\begin{table}[htbp]",
      "\\centering",
      "\\caption{Regression Results}"
    )
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
  for (i in 1:nrow(result_df)) {
    row <- result_df[i, ]
    
    # First process variable name
    var_name <- as.character(row[1])
    
    # Create coefficient row
    coef_values <- character(ncol(row))
    coef_values[1] <- var_name  # Variable name
    
    # Create SE row (empty in first column)
    se_values <- character(ncol(row))
    se_values[1] <- ""  # Empty first column
    
    # Fill in coefficient and SE values for each model
    for (j in 2:ncol(row)) {
      cell <- row[j]
      
      if (is.na(cell) || cell == "") {
        coef_values[j] <- ""
        se_values[j] <- ""
      } else if (grepl("\n", cell)) {
        # Split into coef and SE
        parts <- strsplit(as.character(cell), "\n")[[1]]
        coef_values[j] <- parts[1]  # Coefficient with stars
        se_values[j] <- parts[2]    # SE with parentheses
      } else {
        coef_values[j] <- as.character(cell)
        se_values[j] <- ""
      }
    }
    
    # Add the coefficient row
    latex <- c(latex, paste(coef_values, collapse = " & "), "\\\\")
    
    # Add the SE row if it has any content
    if (any(nchar(se_values) > 0)) {
      latex <- c(latex, paste(se_values, collapse = " & "), "\\\\")
    }
  }

  # Midrule before stats
  # latex <- c(latex, "\\midrule")

  # Add stat rows
  stat_rows <- apply(result2_df, 1, function(row) {
    if (all(row == "")) {
      return(NULL)
    }
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
      "$^{***}\\: p < 0.001;\\: ^{**}\\: p < 0.01;\\: ^{*}\\: p < 0.05;\\: ^{.}\\: p < 0.1$}"
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
