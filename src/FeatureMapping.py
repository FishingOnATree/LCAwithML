__author__ = 'Rays'

import csv
import datetime
import numpy as np
import pandas as pd
import re


GRADE_DICT = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
HOME_OWNERSHIP_LIST = ["OWN", "RENT", "MORTGAGE", "OTHER"]
PURPOSE_LIST = ["car", "credit_card", "debt_consolidation", "educational", "home_improvement", "house",
                "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business",
                "vacation", "wedding"]
VERIFIED_STATUS = ["Verified", "Source Verified"]


def extract_col_pos_dict(headers):
    return {h: i for i, h in enumerate(headers)}


def split_raw_data(line):
    return line.replace("\n", "").split("\",\"")


def map_number(x, num_type):
    return num_type(x) if x else num_type(0)


def map_two_number(x1, x2, num_type):
    if x2:
        return num_type(x2)
    elif x1:
        return num_type(x1)
    else:
        num_type(0)


def map_verification_status(x1, x2):
    if x1 in VERIFIED_STATUS or x2 in VERIFIED_STATUS:
        return 1
    else:
        return 0


def map_term(x):
    if x.strip().startswith("60"):
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
    feature = [1 if x == p else 0 for x in PURPOSE_LIST]
    assert sum(feature) == 1
    return feature


def map_home_ownership(p):
    status = "OTHER" if p not in HOME_OWNERSHIP_LIST else p
    feature = [1 if x == status else 0 for x in HOME_OWNERSHIP_LIST]
    assert sum(feature) == 1
    return feature


def map_time_diff_in_month(issued_date_str, cr_time_str):
    if issued_date_str:
        issued_date = datetime.datetime.strptime(issued_date_str, "%b-%Y")
    else:
        issued_date = datetime.today()
    cr_time = datetime.datetime.strptime(cr_time_str, "%b-%Y")
    return (issued_date.year - cr_time.year) * 12 + issued_date.month - cr_time.month


def mapping(fields, col_pos_dict):
    xi = list()
    # numerical values in raw files, x1 - x16
    xi.append(map_number(fields[col_pos_dict["acc_now_delinq"]], int))
    xi.append(map_number(fields[col_pos_dict["all_util"]], float))
    xi.append(map_number(fields[col_pos_dict["annual_inc"]], float))
    xi.append(map_number(fields[col_pos_dict["collections_12_mths_ex_med"]], int))
    xi.append(map_number(fields[col_pos_dict["delinq_2yrs"]], int))
    xi.append(map_two_number(fields[col_pos_dict["dti_joint"]], fields[col_pos_dict["dti"]], float))
    xi.append(map_number(fields[col_pos_dict["fico_range_high"]], int))
    xi.append(map_number(fields[col_pos_dict["fico_range_low"]], int))
    xi.append(map_number(fields[col_pos_dict["inq_last_6mths"]], int))
    xi.append(map_number(fields[col_pos_dict["loan_amnt"]], float))
    xi.append(map_number(fields[col_pos_dict["mths_since_last_delinq"]], int))
    xi.append(map_number(fields[col_pos_dict["mths_since_last_major_derog"]], int))
    xi.append(map_number(fields[col_pos_dict["mths_since_last_record"]], int))
    xi.append(map_number(fields[col_pos_dict["open_acc"]], int))
    xi.append(map_number(fields[col_pos_dict["pub_rec"]], int))
    xi.append(map_number(fields[col_pos_dict["total_acc"]], int))
    # direct transformation, x17 - x22
    xi.append(GRADE_DICT[fields[col_pos_dict["grade"]]])
    xi.append(map_time_diff_in_month(fields[col_pos_dict["issue_d"]], fields[col_pos_dict["earliest_cr_line"]]))
    xi.append(map_emp_length(fields[col_pos_dict["emp_length"]]))
    xi.append(map_term(fields[col_pos_dict["term"]]))
    xi.append(map_verification_status(fields[col_pos_dict["verification_status_joint"]], fields[col_pos_dict["verification_status"]]))
    # 1-to-k transformation
#    xi.append((fields[col_pos_dict["purpose"]]))  # to be handled
    xi.extend(map_purpose(fields[col_pos_dict["purpose"]]))
#    xi.append((fields[col_pos_dict["home_ownership"]]))  # to be handled
    xi.extend(map_home_ownership(fields[col_pos_dict["home_ownership"]]))
    # mapping y
    yi = map_loan_status(fields[col_pos_dict["loan_status"]])
    return yi, xi


def map_dti(row):
    if not pd.isnull(row["dti_joint"]):
        result = row["dti_joint"]
    elif not pd.isnull(row["dti"]):
        result = row["dti"]
    else:
        result = 0
    return result


def map_credit_length(row):
    #TODO - when read CSV, we can use a default date parser to parse date
    if row["issue_d"]:
        issued_date = datetime.datetime.strptime(row["issue_d"], "%b-%Y")
    else:
        issued_date = datetime.now()
    cr_time = datetime.datetime.strptime(row["earliest_cr_line"], "%b-%Y")
    return (issued_date.year - cr_time.year) * 12 + issued_date.month - cr_time.month


def map_verify_status(row):
    if row["verification_status_joint"] in VERIFIED_STATUS or row["verification_status"] in VERIFIED_STATUS:
        return 1
    else:
        return 0

X_COLUMNS = ["acc_now_delinq", "annual_inc", "collections_12_mths_ex_med", "delinq_2yrs",
             "fico_range_high", "fico_range_low", "inq_last_6mths", "loan_amnt", "mths_since_last_delinq",
             "mths_since_last_major_derog", "mths_since_last_record", "open_acc", "pub_rec", "total_acc",
             "dti_new", "grade_new", "credit_history_in_month", "emp_length_new", "verify_status", "term"]



def load_data(raw_data_file):
    df = pd.read_csv(raw_data_file, sep=",", engine="python", quoting=csv.QUOTE_ALL)
    df = df.rename(columns={"\"id": "id"})

    # X recalculated terms
    df["dti_new"] = df.apply(map_dti, axis=1)
    df["grade_new"] = df["grade"].map(GRADE_DICT)
    df["credit_history_in_month"] = df.apply(map_credit_length, axis=1)
    df["emp_length_new"] = df["emp_length"].map(map_emp_length)
    df["verify_status"] = df.apply(map_verify_status, axis=1)
    df["revol_util"].replace('%','',regex=True).astype('float')/100
    ## need to replace isnull, NaN, "null"
    ## TODO replace them with a good NEVER-HAPPENED value... one idea is to use negative credit history in months
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_delinq", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_major_derog", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_last_record", -1)
    map_na_by_correlation(df, "credit_history_in_month", "mths_since_recent_revol_delinq", -1)
    df["term"] = df["term"].map(map_term)
    # Y
    df["loan_status"] = df["loan_status"].map(map_loan_status)

    # TODO drop NA to have more features available
    # df = df[np.isfinite(df["total_rev_hi_lim"])]
    #re-order column names for easier processing
    df = df.reindex_axis(sorted(df.columns), axis=1)
    return df


def map_na_by_correlation(df, source_col, target_col, correlation):
    index = df[target_col].isnull()
    df.loc[index, target_col] = df.loc[index, source_col]*correlation


def map_features_new(df):
    #TODO one-hot encoding
    x = df[X_COLUMNS].values
    home_ownership_features = df["home_ownership"].map(map_home_ownership)
    purpose_features = df["purpose"].map(map_purpose)
    y = df["loan_status"].values
    return x, y


def map_features(raw_data_file):
    col_pos_dict = {}
    count = 0
    x = []
    y = []
    with open(raw_data_file) as f:
        for line in f:
            if line.startswith("\"id\""):
                headers = split_raw_data(line)
                col_pos_dict = extract_col_pos_dict(headers)
            elif line.startswith("\""):
                try:
                    fields = split_raw_data(line)
                    if len(fields) > 0:
                        yi, xi = mapping(fields, col_pos_dict)
                        x.append(xi)
                        y.append(yi)
                        count += 1
                except ValueError:
                    print("ValueError")
    print("%d row mapped" % count)
    x = np.array(x)
    y = np.array(y)
    return x, y
