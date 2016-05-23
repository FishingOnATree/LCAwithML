__author__ = 'Rays'

import datetime
import numpy as np
import re


GRADE_DICT = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
PURPOSE_LIST = ["car", "credit_card", "debt_consolidation", "educational", "home_improvement", "house",
                "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business",
                "vacation", "wedding"]
HOME_OWNERSHIP_LIST = ["OWN", "RENT", "MORTGAGE", "OTHER"]

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
    if x1 or x2 in ["Verified", "Source Verified"]:
        return 1
    else:
        return 0


def map_term(x):
    if x.strip().startswith("60"):
        return 1
    else:
        return 0


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
    xi.append((fields[col_pos_dict["purpose"]]))  # to be handled
    xi.extend(map_purpose(fields[col_pos_dict["purpose"]]))
    xi.append((fields[col_pos_dict["home_ownership"]]))  # to be handled
    xi.extend(map_home_ownership(fields[col_pos_dict["home_ownership"]]))
    # mapping y
    yi = map_loan_status(fields[col_pos_dict["loan_status"]])
    return xi, yi


def map_features(raw_data_file):
    col_pos_dict = {}
    count = 0
    with open(raw_data_file) as f:
        for line in f:
            if line.startswith("\"id\""):
                headers = split_raw_data(line)
                col_pos_dict = extract_col_pos_dict(headers)
            elif line.startswith("\""):
                fields = split_raw_data(line)
                if len(fields) > 0:
                    yi, xi = mapping(fields, col_pos_dict)
                    count += 1
    f.close()
    print(count)

#map_features("data/traing_data_small.csv")