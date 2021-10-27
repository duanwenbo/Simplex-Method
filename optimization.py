#!/user/bin/env python
# Author: Wenbo Duan
# Email: pv19120@bristol.ac.uk
# Time: 25/10/2021
# File: Optimization.py

from initalization import extract_info
import pandas as pd
import numpy as np
import re


def record_process(func):
    def wrapper(*args):
        output = func(*args)
        if len(output) != 2:
            with open("log.txt", "a+") as f:
                f.write("Initial tableau\n")
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

def record_result(func):
    def wrapper(*args):
        answer = func(*args)
        df = pd.DataFrame(answer, index=["answer"])
        variables = " ".join(list(df))
        original_variables = ",".join(re.findall(r'x_\d', variables))
        slack_variables = ",".join(re.findall(r's_\d', variables))
        question = ""
        with open("input.txt", "r") as f:
            question = f.read()
        with open("solution.txt", "a+") as f:
            f.write("\n############################ Question ############################\n")
            f.write(question)
            f.write("\n\n############################ ANSWER ############################\n")
            f.write(str(df))
            f.write("\nwhere:\noriginal variables: {}\nslack variables: {}".format(original_variables, slack_variables))
    return wrapper
        
class LP_Solver:
    def __init__(self, key_info:str):
        self.constraints = key_info['constraints']
        self.objective_func = key_info['objective_function']
        self.type = key_info['optimization_type']
    
    def _read_coefficient(self, constraint:str, key_variable:str) -> float:
        """
        read the coefficient of a specified variable in an equation
        """
        constraint = constraint.replace(" ","")
        if key_variable == "solution":
            coefficient = re.search(r'(?<=\=).*$', constraint).group()
        else:
            if re.search(key_variable, constraint):
                coefficient = re.search(r'\d*?(?={})'.format(key_variable), constraint).group()
                if coefficient == "":
                    coefficient = 1
            else:
                coefficient = 0
        return(float(coefficient))

    def slack_var(self) -> list:
        """
        change inequaity to equality
        """
        equality_constraints = []
        inequality_count = 1
        for constraint in self.constraints:
            if re.search(r'\<', constraint):
                constraint = re.sub(r'\<', "+ s_{}".format(inequality_count), constraint)
                inequality_count += 1
            elif re.search(r'\>', constraint):
                constraint = re.sub(r'\<', "- s_{}".format(inequality_count), constraint)
                inequality_count += 1
            else:
                pass
            equality_constraints.append(constraint)
        return equality_constraints
    
    @record_process
    def tableau (self, equality_constraints:list) -> pd.DataFrame:
        """
        create a standard tableau for LP optimization
        """
        # find all different variables to create a tableau head
        tableau_head = []
        for constraint in equality_constraints:
            variables = re.findall(r'x_\d|s_\d', constraint)
            tableau_head.extend(variables)
        tableau_head = list(set(tableau_head))  # Deduplication
        tableau_head.sort(key=lambda x: (x[0],-float(x[2])), reverse=True )
        tableau_head.append("solution")
        
        # initialize a null dataframe(tableau) with only header
        df = pd.DataFrame(columns=tableau_head)

        # find coefficients by columns, fill the tableau
        for key_variable in tableau_head:
            coefficients = []
            for constraint in equality_constraints:
               coefficient = self._read_coefficient(constraint, key_variable) 
               coefficients.append(coefficient)
            df["{}".format(key_variable)] = coefficients
        
        # add the objective function row 
        coefficient = [self._read_coefficient(self.objective_func, var) for var in tableau_head[:-1]]
        # transform to standard tableau format: -c^T x + Z = 0
        coefficient = [ (-i if i != 0.0 else i) for i in coefficient]
        coefficient.append(0.)
        df.loc[df.shape[0]] = coefficient
        return df
    
    @record_process
    def single_pivot(self, tableau:pd.DataFrame):
        def _pivot_column() -> int:
            # determine which column to be pivoted
            objective_row = tableau.iloc[-1].to_numpy()
            pivot_column, = np.where(np.argsort(objective_row) == 0)[0]
            return pivot_column
        
        def _pivot_row(pivot_column:int) -> int:
            # determine which row to be pivoted
            tableau['ratio'] = tableau.iloc[:-1,-1] /  tableau.iloc[:-1,pivot_column]
            pivot_row, = np.where(np.argsort(tableau['ratio'].to_numpy()) == 0)[0]
            return pivot_row

        column_index = _pivot_column()
        row_index = _pivot_row(column_index)
        tableau = tableau.drop(columns=['ratio'])
        pivot_key = tableau.iloc[row_index, column_index]
        # check if the pivot_key is a basic variable
        if pivot_key != 1.:
            tableau.iloc[row_index,:] = (tableau.iloc[row_index,:] / pivot_key)

        # start pivoting on this column iteratively
        for index, value in enumerate(tableau.iloc[:,column_index]):
            if index == row_index:
                pass
            else:
                tableau.iloc[index,:] -= (tableau.iloc[row_index,:]*tableau.iloc[index,column_index])

        # check if the pivot is finished
        for value in tableau.iloc[-1,:]:
            if value < 0:
                done = False
                break
            else:
                done = True 
        return tableau, done
    
    @record_result
    def read_result(self, tableau:pd.DataFrame, objective_row:np.array):
        answer = {}
        for index in range(tableau.shape[1]-1):
            # check if the column is a unit vector
            current_column = tableau.iloc[:, index]
            modulus = np.linalg.norm(current_column.to_numpy())
            if modulus != 1.:
                answer["{}".format(tableau.columns[index])] = 0
            else:
                answer_row = current_column[current_column==1.].index[0]
                answer["{}".format(tableau.columns[index])] = tableau.iloc[answer_row,-1]
        
        # find the Z value according to the answer of independent variables
        # reshape objective vector as 1x4 array
        objective_vector = np.expand_dims(objective_row, axis=0)
        # reshape answer vector as 4x1 array
        answer_vector = np.array(list(answer.values()))
        answer_vector = np.expand_dims(answer_vector, axis=1)
        # Z = C^T * X
        z = np.dot(objective_vector, answer_vector)[0][0]
        answer['z'] = z
        return answer
    
    def solve(self):
        """
        main function to execute algorithm
        """
        equality_constraints = self.slack_var()
        tableau = self.tableau(equality_constraints)
        objective_row = (tableau.iloc[-1,:-1]*-1).to_numpy()
        done = False
        while not done:
            next_tableau, done = self.single_pivot(tableau)
            tableau = next_tableau
        self.read_result(tableau,objective_row)
        print("finished!")
     
if __name__ == "__main__":
    path = "input.txt"
    info = extract_info(input_path=path)
    lp_solver = LP_Solver(info)
    lp_solver.solve()
