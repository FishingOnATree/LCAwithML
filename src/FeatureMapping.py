__author__ = 'Rays'

import csv
import datetime
import numpy as np
import pandas as pd
import itertools
import re


GRADE_DICT = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
HOME_OWNERSHIP_LIST = ["OWN", "RENT", "MORTGAGE", "OTHER"]
X_COLUMNS = ["acc_now_delinq", "acc_open_past_24mths", "annual_inc", "avg_cur_bal",
             "collections_12_mths_ex_med", "credit_history_in_month", "delinq_2yrs", "delinq_amnt", "dti_new",
             "emp_length_new", "fico_range_high", "fico_range_low", "inq_last_6mths",
             "loan_amnt", "grade_new", "inq_last_6mths", "installment",
             "int_rate", "loan_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
             "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_last_delinq",
             "mths_since_last_major_derog", "mths_since_last_record", "mths_since_recent_revol_delinq", "num_actv_bc_tl",
             "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
             "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_30dpd",
             "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "open_acc", "pct_tl_nvr_dlq",
             "percent_bc_gt_75", "pub_rec_bankruptcies", "total_acc", "revol_bal",
             "tax_liens", "term", "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim",
             "total_acc", "total_rev_hi_lim", "verify_status"]
PURPOSE_LIST = [("car", "Car financing"),
                ("credit_card",  "Credit card refinancing"),
                ("debt_consolidation", "Debt consolidation"),
                ("educational", ""),
                ("home_improvement", "Home improvement"),
                ("house", "Home buying"),
                ("major_purchase", "Major purchase"),
                ("medical", "Medical expenses"),
                ("moving", "Moving and relocation"),
                ("other", "Other"),
                ("renewable_energy", ""),
                ("small_business", "Business"),
                ("vacation", "Vacation"),
                ("wedding", "Wedding expenses")]

VERIFIED_STATUS = ["Verified", "Source Verified"]


def map_term(x):
    if str(x).strip().startswith("60"):
        return 60
    else:
        return 36


def map_emp_length(x):
    pat = r"(\d{1,2}).*year.*"
    if re.match(pat, x):
        return int(re.findall(pat, x)[0])
    else:
        return 0


def map_loan_status(y):
    return 0 if y in ["Charged Off", "Default", "Late (31-120 days)"] else 1


def map_purpose(p):
    feature = [1 if p in x else 0 for x in PURPOSE_LIST]
    assert sum(feature) == 1
    return feature


def map_home_ownership(p):
    # this works but if we use: DictEncoding:
    # HOME_OWNERSHIP_TYPE = [{"home_ownership": "OWN"},
    #                    {"home_ownership": "RENT"},
    #                    {"home_ownership": "MORTGAGE"},
    #                    {"home_ownership": "OTHER"}]
    # v = DictVectorizer(sparse=False)
    # X = v.fit_transform(HOME_OWNERSHIP_TYPE)
    #v.transform({"home_ownership":"OWN"})
    status = "OTHER" if p not in HOME_OWNERSHIP_LIST else p
    feature = [1 if x == status else 0 for x in HOME_OWNERSHIP_LIST]
    assert sum(feature) == 1
    return feature


def map_dti(row):
    if not pd.isnull(row["dti_joint"]):
        result = row["dti_joint"]
    elif not pd.isnull(row["dti"]):
        result = row["dti"]
    else:
        result = 0
    return result


def map_credit_length(row):
    return (row["issue_d"].year - row["earliest_cr_line"].year) * 12 + row["issue_d"].month - row["earliest_cr_line"].month


def map_verify_status(row):
    # because LC doesn't have the same column name for historical and new loan data, we have to hack it.
    if "verification_status" in row and row["verification_status"] in VERIFIED_STATUS:
        return 1
    else:
        if ("verification_status_joint" in row and row["verification_status_joint"] in VERIFIED_STATUS) or \
                ("verified_status_joint" in row and row["verified_status_joint"] in VERIFIED_STATUS):
            return 1
    return 0


def map_dates(content):
    try:
        # format for historical data
        date = datetime.datetime.strptime(content, "%b-%Y")
    except ValueError:
        # format for new loans
        date = datetime.datetime.strptime(content, "%m-%d-%Y %H:%M:%S")
    return date


def map_percent(s):
    if isinstance(s, basestring):
        return float(s.strip('%'))/100
    else:
        return float(s)


def load_data(raw_data_file):
    df = pd.read_csv(raw_data_file, sep=",", engine="python", quoting=csv.QUOTE_ALL)

    # drop NA to have more features available without tot_cur_bal -
    #       idea is we don't lend to those we don't know how much he owes outside
    df = df[np.isfinite(df["tot_cur_bal"])]
    df = df[np.isfinite(df["pct_tl_nvr_dlq"])]
    df = df[np.isfinite(df["percent_bc_gt_75"])]

    # fix existing data
    df["bc_open_to_buy"] = df["bc_open_to_buy"].fillna(0.0)
    df["bc_util"] = df["bc_util"].fillna(100.0)
    df["int_rate"] = df["int_rate"].map(map_percent)
    # df["revol_util"] = df["revol_util"].map(map_percent)
    df["mo_sin_old_il_acct"] = df["mo_sin_old_il_acct"].fillna(-1)

    # diff format for historical and current
    df["earliest_cr_line"] = df["earliest_cr_line"].map(map_dates)
    # issue_d N/A in new loans, but we just need it to calculate credit_history
    if "issue_d" in df:
        df["issue_d"] = df["issue_d"].map(map_dates)
    else:
        df["issue_d"] = datetime.datetime.today()

    # X recalculated terms
    df["dti_new"] = df.apply(map_dti, axis=1)
    df["grade_new"] = df["grade"].map(GRADE_DICT)
    df["credit_history_in_month"] = df.apply(map_credit_length, axis=1)
    df["emp_length_new"] = df["emp_length"].map(map_emp_length)
    df["verify_status"] = df.apply(map_verify_status, axis=1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_delinq", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_major_derog", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_record", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_recent_revol_delinq", -1)
    df["term"] = df["term"].map(map_term)
    # Y
    if "loan_status" not in df:
        df["loan_status"] = -1
    else:
        df["loan_status"] = df["loan_status"].map(map_loan_status)

    #re-order column names for easier processing
    df = df.reindex_axis(sorted(df.columns), axis=1)
    return df


def map_na_by_correlation(df, source_col, target_col, correlation):
    index = df[target_col].isnull()
    df.loc[index, target_col] = df.loc[index, source_col]*correlation


def shape_list(data):
    return np.asarray(list(itertools.chain(*data))).reshape(data.shape[0], len(data[0]))


def map_features(df):
    x = np.append(df[X_COLUMNS].values,
                  shape_list(df["home_ownership"].map(map_home_ownership).values),
                  axis=1)
    x = np.append(x,
                  shape_list(df["purpose"].map(map_purpose).values),
                  axis=1)
    y = df["loan_status"].values
    return x, y
