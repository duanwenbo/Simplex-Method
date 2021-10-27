#!/user/bin/env python
# Author: Wenbo Duan
# Email: pv19120@bristol.ac.uk
# Time: 22/10/2021
# File: Initialization.py


import re
import os

def _detect_previous_doc():
    if os.path.exists("log.txt"):
        os.remove("log.txt")
    if os.path.exists("solution.txt"):
        os.remove("solution.txt")

def extract_info(input_path) -> dict:
    """
    input_path: the file path of the input file, str
    return: a dictionary containing optimization_type, objective_function and constraints
    Extracting key info from the text file to initialize the problem
    """

    _detect_previous_doc()

    initial_info = {}
    with open(input_path, 'r') as f:
        data = f.read()

    opt_type = re.search(r'(?<=type:\s").*?(?=")', data).group()
    initial_info["optimization_type"] = opt_type

    objective_function = re.search(r'(?<=objective_function:\s").*?(?=")', data).group()
    initial_info["objective_function"] = objective_function

    redundance = re.search(r'^.*constraints:\s',data,re.DOTALL).group()
    constraints = data.replace(redundance,"")
    constraints = re.findall(r'(?<=").*?(?=")', constraints, re.DOTALL)
    constraints = [constraint for constraint in constraints if constraint != '\n']
    initial_info["constraints"] = constraints

    return initial_info


if __name__ == "__main__":
    input_path = "input.txt" # the file path of your input file
    extract_info(input_path)
