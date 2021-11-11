#!/user/bin/env python
# Author: Wenbo Duan
# Email: pv19120@bristol.ac.uk
# Time: 10/11/2021
# File: ultils.py

from os import name
import re
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

def record_process(func):
    def wrapper(*args):
        output = func(*args)
        if len(output) != 2:
            with open("log.txt", "a+") as f:
                f.write("\nInitial tableau\n")
                tableau = str(output)
                f.write(tableau)
                f.write("\n")
        else:
            with open("log.txt", "a+") as f:
                f.write("\nnext pivot\n")
                tableau = str(output[0])  
                f.write(tableau)
                f.write("\n")
        return output
    return wrapper


def save_answer(answers:list, initial_tableau:pd.DataFrame):
    # initialize the output file:
    # 1. repilcated the question description
    question = ""
    with open("input.txt", "r") as f:
        question = f.read()
    with open("solution.txt", "a+") as f:
        f.write("\n############################ QUESTION ############################\n")
        f.write(question)
        f.write("\n\n############################ ANSWER ############################\n")
    # 2. indicate the type of the answer
    # 2.1 multipe optimal
    if len(answers) > 1:
        with open("solution.txt", "a+") as f:
            f.write("\nThis question has -MULTIPLE OPTIMALS-\n")
        solutions = []
        for index, answer in enumerate(answers):
            # find the solutions of original variables (e.g. x_1, x_2)
            key_sol = dict(filter(lambda keys: re.search(r'x', keys[0]) , answer.items()))
            solutions.append(key_sol)
            # record current solution
            with open("solution.txt", "a+") as f:
                df = pd.DataFrame(answer, index=["scenario {}".format(str(index))])
                f.write("\n")
                f.write(str(df))
                f.write("\n")
        # synthesis multiple answer
        variables_str = " , ".join(solutions[0].keys())
        answer_1 = " , ".join(str(v) for v in solutions[0].values())
        answer_2 = " , ".join(str(v) for v in solutions[1].values())
        with open("solution.txt", "a+") as f:
            f.write("\n\nTherefore, the final solution is\n({}) = C({}) + (1-C)({})\nIn which 0 <= C <= 1\n".format(variables_str, answer_1, answer_2))
    # 2.2 single solution
    else:
         with open("solution.txt", "a+") as f:
            df = pd.DataFrame(answers[0], index=["answer"])
            f.write(str(df))
            f.write("\n")

            
    # inidcate the type of variables
    df = pd.DataFrame(answers[0], index=["answer"])
    variables = " ".join(list(df))
    # the order of variable analysis output is PREDEFIND !
    name_list =  ["original variables", "slack variables", "surplus variables", "artificial variables"]
    var_analysis = []
    var_analysis.append(re.findall(r'x_\d', variables))
    slack_variables, surplus_variables = [], []
    for var in list(initial_tableau):
        if re.search(r's_', var):
            if initial_tableau[var].sum() == 1:
               
                slack_variables.append(var)
            elif initial_tableau[var].sum() == -1:
                surplus_variables.append(var)
            else:
                a = initial_tableau[var].sum()
                raise AttributeError
    var_analysis.append(slack_variables)
    var_analysis.append(surplus_variables)
    var_analysis.append(re.findall(r'a_\d', variables))

    assert len(name_list) == len(var_analysis), 'check your answer!'

    with open("solution.txt", "a+") as f:
        for index, var in enumerate(var_analysis):
            if len(var) >0:
                f.write("\n{} : {}".format(name_list[index],",".join(var)))