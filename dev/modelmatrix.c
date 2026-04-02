// called from R as  .Externals2(C_modelmatrix, t, data)
SEXP modelmatrix(SEXP call, SEXP op, SEXP args, SEXP rho)
{
    SEXP expr, factors, terms, vars, vnames, assign;
    SEXP xnames, tnames, rnames;
    SEXP count, contrast, contr1, contr2, nlevs, ordered, columns, x;
    SEXP variable, var_i;
    int fik, first, i, j, k, kk, ll, n, nc, nterms, nVar;
    int intrcept, jstart, jnext, risponse, indx, rhs_response;
    double dk, dnc;
    char buf[BUFSIZE]="\0";
    char *bufp;
    const char *addp;
    R_xlen_t nn;

    args = CDR(args);

    /* Get the "terms" structure and extract */
    /* the intercept and response attributes. */
    terms = CAR(args); // = 't' in R's calling code

    intrcept = asLogical(getAttrib(terms, install("intercept")));
    if (intrcept == NA_INTEGER)
	intrcept = 0;

    risponse = asLogical(getAttrib(terms, install("response")));
    if (risponse == NA_INTEGER)
	risponse = 0;

    /* Get the factor pattern matrix.  We duplicate this because */
    /* we may want to alter it if we are in the no-intercept case. */
    /* Note: the values of "nVar" and "nterms" are the REAL number of */
    /* variables in the model data frame and the number of model terms. */

    nVar = nterms = 0;		/* -Wall */
    PROTECT(factors = duplicate(getAttrib(terms, install("factors"))));
    if (length(factors) == 0) {
	/* if (intrcept == 0)
	   error("invalid model (zero parameters).");*/
	nVar = 1;
	nterms = 0;
    }
    else if (isInteger(factors) && isMatrix(factors)) {
	nVar = nrows(factors);
	nterms = ncols(factors);
    }
    else error(_("invalid '%s' argument"), "terms");

    /* Get the variable names from the factor matrix */

    vnames = getAttrib(factors, R_DimNamesSymbol);
    if (length(factors) > 0) {
	if (length(vnames) < 1 ||
	    (nVar - intrcept > 0 && !isString(VECTOR_ELT(vnames, 0))))
	    error(_("invalid '%s' argument"), "terms");
	vnames = VECTOR_ELT(vnames, 0);
    }

    /* Get the variables from the model frame.  First perform */
    /* elementary sanity checks.  Notes:  1) We need at least */
    /* one variable (lhs or rhs) to compute the number of cases. */
    /* 2) We don't type-check the response. */

    vars = CADR(args);
    if (!isNewList(vars) || length(vars) < nVar)
	error(_("invalid model frame"));
    if (length(vars) == 0)
	error(_("do not know how many cases"));

    nn = n = nrows(VECTOR_ELT(vars, 0));
    /* This could be generated, so need to protect it */
    PROTECT(rnames = getAttrib(vars, R_RowNamesSymbol));

    /* This section of the code checks the types of the variables
       in the model frame.  Note that it should really only check
       the variables if they appear in a term in the model.
       Because it does not, we need to allow other types here, as they
       might well occur on the LHS.
       The R code converts all character variables in the model frame to
       factors, so the only types that ought to be here are logical,
       integer (including factor), numeric and complex.
     */

    PROTECT(variable = allocVector(VECSXP, nVar));
    PROTECT(nlevs = allocVector(INTSXP, nVar));
    PROTECT(ordered = allocVector(LGLSXP, nVar));
    PROTECT(columns = allocVector(INTSXP, nVar));

    for (i = 0; i < nVar; i++) {
	var_i = SET_VECTOR_ELT(variable, i, VECTOR_ELT(vars, i));
	if (nrows(var_i) != n)
	    error(_("variable lengths differ (found for variable %d)"), i);
	if (isOrdered_int(var_i)) {
	    LOGICAL(ordered)[i] = 1;
	    if((INTEGER(nlevs)[i] = nlevels(var_i)) < 1)
		error(_("variable %d has no levels"), i+1);
	    /* will get updated later when contrasts are set */
	    INTEGER(columns)[i] = ncols(var_i);
	}
	else if (isUnordered_int(var_i)) {
	    LOGICAL(ordered)[i] = 0;
	    if((INTEGER(nlevs)[i] = nlevels(var_i)) < 1)
		error(_("variable %d has no levels"), i+1);
	    /* will get updated later when contrasts are set */
	    INTEGER(columns)[i] = ncols(var_i);
	}
	else if (isLogical(var_i)) {
	    LOGICAL(ordered)[i] = 0;
	    INTEGER(nlevs)[i] = 2;
	    INTEGER(columns)[i] = ncols(var_i);
	}
	else if (isNumeric(var_i)) {
	    SET_VECTOR_ELT(variable, i, coerceVector(var_i, REALSXP));
	    var_i = VECTOR_ELT(variable, i);
	    LOGICAL(ordered)[i] = 0;
	    INTEGER(nlevs)[i] = 0;
	    INTEGER(columns)[i] = ncols(var_i);
	}
	else {
	    LOGICAL(ordered)[i] = 0;
	    INTEGER(nlevs)[i] = 0;
	    INTEGER(columns)[i] = ncols(var_i);
	}
/*	else
	    error(_("invalid variable type for '%s'"),
	    CHAR(STRING_ELT(vnames, i))); */
    }

    /* If there is no intercept we look through the factor pattern */
    /* matrix and adjust the code for the first factor found so that */
    /* it will be coded by dummy variables rather than contrasts. */

    if (!intrcept) {
	for (j = 0; j < nterms; j++) {
	    for (i = risponse; i < nVar; i++) {
		if (INTEGER(nlevs)[i] > 1
		    && INTEGER(factors)[i + j * nVar] > 0) {
		    INTEGER(factors)[i + j * nVar] = 2;
		    goto alldone;
		}
	    }
	}
    }
 alldone:
    ;

    /* Compute the required contrast or dummy variable matrices. */
    /* We set up a symbolic expression to evaluate these, substituting */
    /* the required arguments at call time.  The calls have the following */
    /* form: (contrast.type nlevs contrasts) */

    PROTECT(contr1 = allocVector(VECSXP, nVar));
    PROTECT(contr2 = allocVector(VECSXP, nVar));

    PROTECT(expr = allocLang(3));
    SETCAR(expr, install("contrasts"));
    SETCADDR(expr, allocVector(LGLSXP, 1));

    /* FIXME: We need to allow a third argument to this function */
    /* which allows us to specify contrasts directly.  That argument */
    /* would be used here in exactly the same way as the below. */
    /* I.e. we would search the list of constrast specs before */
    /* we try the evaluation below. */

    for (i = 0; i < nVar; i++) {
	if (INTEGER(nlevs)[i]) {
	    k = 0;
	    for (j = 0; j < nterms; j++) {
		if (INTEGER(factors)[i + j * nVar] == 1)
		    k |= 1;
		else if (INTEGER(factors)[i + j * nVar] == 2)
		    k |= 2;
	    }
	    SETCADR(expr, VECTOR_ELT(variable, i));
	    if (k & 1) {
		LOGICAL(CADDR(expr))[0] = 1;
		SET_VECTOR_ELT(contr1, i, eval(expr, rho));
	    }
	    if (k & 2) {
		LOGICAL(CADDR(expr))[0] = 0;
		SET_VECTOR_ELT(contr2, i, eval(expr, rho));
	    }
	}
    }

    /* By convention, an rhs term identical to the response generates nothing
       in the model matrix (but interactions involving the response do). */

    rhs_response = -1;
    if (risponse > 0) /* there is a response specified */
	for (j = 0; j < nterms; j++)
	    if (INTEGER(factors)[risponse - 1 + j * nVar]) {
		for (i = 0, k = 0; i < nVar; i++)
		    k += INTEGER(factors)[i + j * nVar] > 0;
		if (k == 1) {
		    rhs_response = j;
		    break;
		}
	    }


    /* We now have everything needed to build the design matrix. */
    /* The first step is to compute the matrix size and to allocate it. */
    /* Note that "count" holds a count of how many columns there are */
    /* for each term in the model and "nc" gives the total column count. */

    PROTECT(count = allocVector(INTSXP, nterms));
    if (intrcept)
	dnc = 1;
    else
	dnc = 0;
    for (j = 0; j < nterms; j++) {
	if (j == rhs_response) {
	    warning(_("the response appeared on the right-hand side and was dropped"));
	    INTEGER(count)[j] = 0;  /* need this initialised */
	    continue;
	}
	dk = 1;	/* accumulate in a double to detect overflow */
	for (i = 0; i < nVar; i++) {
	    if (INTEGER(factors)[i + j * nVar]) {
		if (INTEGER(nlevs)[i]) {
		    switch(INTEGER(factors)[i + j * nVar]) {
		    case 1:
			dk *= ncols(VECTOR_ELT(contr1, i));
			break;
		    case 2:
			dk *= ncols(VECTOR_ELT(contr2, i));
			break;
		    }
		}
		else dk *= INTEGER(columns)[i];
	    }
	}
	if (dk > INT_MAX) error(_("term %d would require %.0g columns"), j+1, dk);
	INTEGER(count)[j] = (int) dk;
	dnc = dnc + dk;
    }
    if (dnc > INT_MAX) error(_("matrix would require %.0g columns"), dnc);
    nc = (int) dnc;

    /* Record which columns of the design matrix are associated */
    /* with which model terms. */

    PROTECT(assign = allocVector(INTSXP, nc));
    k = 0;
    if (intrcept) INTEGER(assign)[k++] = 0;
    for (j = 0; j < nterms; j++) {
	if(INTEGER(count)[j] <= 0)
	    warning(_("problem with term %d in model.matrix: no columns are assigned"),
		      j+1);
	for (i = 0; i < INTEGER(count)[j]; i++)
	    INTEGER(assign)[k++] = j+1;
    }


    /* Create column labels for the matrix columns. */

    PROTECT(xnames = allocVector(STRSXP, nc));


    /* Here we loop over the terms in the model and, within each */
    /* term, loop over the corresponding columns of the design */
    /* matrix, assembling the names. */

    /* FIXME : The body within these two loops should be embedded */
    /* in its own function. */

    k = 0;
    if (intrcept)
	SET_STRING_ELT(xnames, k++, mkChar("(Intercept)"));

    for (j = 0; j < nterms; j++) {
	if (j == rhs_response) continue;
	for (kk = 0; kk < INTEGER(count)[j]; kk++) {
	    first = 1;
	    indx = kk;
	    bufp = &buf[0];
	    for (i = 0; i < nVar; i++) {
		ll = INTEGER(factors)[i + j * nVar];
		if (ll) {
		    var_i = VECTOR_ELT(variable, i);
		    if (!first)
			bufp = AppendString(bufp, ":");
		    first = 0;
		    if (isFactor(var_i) || isLogical(var_i)) {
			if (ll == 1) {
			    x = ColumnNames(VECTOR_ELT(contr1, i));
			    ll = ncols(VECTOR_ELT(contr1, i));
			}
			else {
			    x = ColumnNames(VECTOR_ELT(contr2, i));
			    ll = ncols(VECTOR_ELT(contr2, i));
			}
			addp = translateChar(STRING_ELT(vnames, i));
			if(strlen(buf) + strlen(addp) < BUFSIZE)
			    bufp = AppendString(bufp, addp);
			else
			    warning(_("term names will be truncated"));
			if (x == R_NilValue) {
			    if(strlen(buf) + 10 < BUFSIZE)
				bufp = AppendInteger(bufp, indx % ll + 1);
			    else
				warning(_("term names will be truncated"));
			} else {
			    addp = translateChar(STRING_ELT(x, indx % ll));
			    if(strlen(buf) + strlen(addp) < BUFSIZE)
				bufp = AppendString(bufp, addp);
			    else
				warning(_("term names will be truncated"));
			}
		    } else if (isComplex(var_i)) {
			error(_("complex variables are not currently allowed in model matrices"));
		    } else if(isNumeric(var_i)) { /* numeric */
			x = ColumnNames(var_i);
			ll = ncols(var_i);
			addp = translateChar(STRING_ELT(vnames, i));
			if(strlen(buf) + strlen(addp) < BUFSIZE)
			    bufp = AppendString(bufp, addp);
			else
			    warning(_("term names will be truncated"));
			if (ll > 1) {
			    if (x == R_NilValue) {
				if(strlen(buf) + 10 < BUFSIZE)
				    bufp = AppendInteger(bufp, indx % ll + 1);
				else
				    warning(_("term names will be truncated"));
			    } else {
				addp = translateChar(STRING_ELT(x, indx % ll));
				if(strlen(buf) + strlen(addp) < BUFSIZE)
				    bufp = AppendString(bufp, addp);
				else
				    warning(_("term names will be truncated"));
			    }
			}
		    } else
			error(_("variables of type '%s' are not allowed in model matrices"),
			      R_typeToChar(var_i));
		    indx /= ll;
		}
	    }
	    SET_STRING_ELT(xnames, k++, mkChar(buf));
	}
    }

    /* Allocate and compute the design matrix. */

    PROTECT(x = allocMatrix(REALSXP, n, nc));
    double *rx = REAL(x);

#if defined(R_MEMORY_PROFILING) && defined(USE_RINTERNALS)
    // RTRACE and SET_RTRACE macros are only available with USE_RINTERNALS
    if (RTRACE(vars)){
       memtrace_report(vars, x);
       SET_RTRACE(x, 1);
    }
#endif

    /* a) Begin with a column of 1s for the intercept. */

    if ((jnext = jstart = intrcept) != 0)
	for (i = 0; i < n; i++)
	    rx[i] = 1.0;

    /* b) Now loop over the model terms */

    contrast = R_NilValue;	/* -Wall */
    for (k = 0; k < nterms; k++) {
	if (k == rhs_response) continue;
	for (i = 0; i < nVar; i++) {
	    if (INTEGER(columns)[i] == 0)
		continue;
	    var_i = VECTOR_ELT(variable, i);
#if defined(R_MEMORY_PROFILING) && defined(USE_RINTERNALS)
	    // RTRACE and SET_RTRACE macros are only available with
	    // USE_RINTERNALS
	    if (RTRACE(var_i)){
	       memtrace_report(var_i, x);
	       SET_RTRACE(x, 1);
	    }
#endif
	    fik = INTEGER(factors)[i + k * nVar];
	    if (fik) {
		switch(fik) {
		case 1:
		    contrast = coerceVector(VECTOR_ELT(contr1, i), REALSXP);
		    break;
		case 2:
		    contrast = coerceVector(VECTOR_ELT(contr2, i), REALSXP);
		    break;
		}
		PROTECT(contrast);
		if (jnext == jstart) {
		    if (INTEGER(nlevs)[i] > 0) {
			int adj = isLogical(var_i)?1:0;
			// avoid overflow of jstart * nn PR#15578
			firstfactor(&rx[jstart * nn], n, jnext - jstart,
				    REAL(contrast), nrows(contrast),
				    ncols(contrast), INTEGER(var_i), adj);
			jnext = jnext + ncols(contrast);
		    }
		    else {
			firstvar(&rx[jstart * nn], n, jnext - jstart,
				 REAL(var_i), n, ncols(var_i));
			jnext = jnext + ncols(var_i);
		    }
		}
		else {
		    if (INTEGER(nlevs)[i] > 0) {
			int adj = isLogical(var_i)?1:0;
			addfactor(&rx[jstart * nn], n, jnext - jstart,
				  REAL(contrast), nrows(contrast),
				  ncols(contrast), INTEGER(var_i), adj);
			jnext = jnext + (jnext - jstart)*(ncols(contrast) - 1);
		    }
		    else {
			addvar(&rx[jstart * nn], n, jnext - jstart,
			       REAL(var_i), n, ncols(var_i));
			jnext = jnext + (jnext - jstart) * (ncols(var_i) - 1);
		    }
		}
		UNPROTECT(1);
	    }
	}
	jstart = jnext;
    }
    PROTECT(tnames = allocVector(VECSXP, 2));
    SET_VECTOR_ELT(tnames, 0, rnames);
    SET_VECTOR_ELT(tnames, 1, xnames);
    setAttrib(x, R_DimNamesSymbol, tnames);
    setAttrib(x, install("assign"), assign);
    UNPROTECT(14);
    return x;
}
// modelmatrix()
