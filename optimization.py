#!/user/bin/env python
# Author: Wenbo Duan
# Email: pv19120@bristol.ac.uk
# Time: 25/10/2021
# File: Optimization.py

from initalization import extract_info
from utils import record_process, save_answer
import pandas as pd
import numpy as np
import re

class LP_Solver:
    def __init__(self, key_info:dict):
        self.constraints = key_info['constraints']
        self.objective_func = key_info['objective_function']
        self.type = key_info['optimization_type']
        self.non_basic_var_column = []  # for multiple opt use
    
    def _read_coefficient(self, constraint:str, key_variable:str) -> float:
        """
        read the coefficient of a specified variable in an equation
        """
        constraint = constraint.replace(" ","")
        if key_variable == "solution":
            coefficient = re.search(r'(?<=\=).*$', constraint).group()
        else:
            if re.search(key_variable, constraint):
                coefficient = re.search(r'\d*\.?\d*?(?={})'.format(key_variable), constraint).group()
                if coefficient == "":
                    coefficient = 1
            else:
                coefficient = 0
        return(float(coefficient))

    def _slack_var(self) -> list:
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
    
    def _unit_vector(self, column:pd.DataFrame) -> bool:
        """
        determine if a column is an unit vector
        This is used to 1. read the result; 2. determine the multiple optimal
        """
        modulus = np.linalg.norm(column.to_numpy())
        if modulus != 1.:
            return False
        else:
            return True

    def _read_result(self, tableau:pd.DataFrame, objective_row:np.array) -> dict:
        """
        read one basic solution from the tableau
        """
        answer = {}
        for index in range(tableau.shape[1]-1):
            # check if the column is a unit vector
            current_column = tableau.iloc[:, index]
            if not self._unit_vector(current_column):
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
    
    def _multiple_optimal(self, tableau:pd.DataFrame) -> int:
        """
        Check if the tableau has multiple optimal
        if yes, return the column index of the other pivot column
        if no, return -1 as the flag of impossible
        """
        multiple_optimal_index = -1
        # if it has non-basic variables
        if len(self.non_basic_var_column) != 0:
            # check if the indicator of this non-basic var is 1 
            # (condition of multiple optimal)
            for index, column in enumerate(self.non_basic_var_column):
                if tableau.iloc[-1,column] == 0:
                    multiple_optimal_index = column
                    del self.non_basic_var_column[index]
        return multiple_optimal_index


    def _detect_non_basic_vars(self, tableau:pd.DataFrame) -> list:
        """
        Detect the position of non-basic variables in the tableau
        This is used for finding the multiple optimal on non-basic var column later
        """
        variables = list(tableau)
        key_var_num = len(list(filter(lambda var: re.search(r'x', var) , variables)))
        non_basic_vars_index = []
        for index in range(key_var_num):
            current_column = tableau.iloc[:,index]
            if not self._unit_vector(current_column):
                non_basic_vars_index.append(index)
        return non_basic_vars_index
    
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
    def single_pivot(self, tableau:pd.DataFrame, multi_opt_index:int):
        def _pivot_column() -> int:
            # pivot on the column with the most negative element
            objective_row = tableau.iloc[-1].to_numpy()
            pivot_column = np.argsort(objective_row)[0]
            return pivot_column
        
        def _pivot_row(pivot_column:int) -> int:
            # determine which row to be pivoted
            tableau['ratio'] = tableau.iloc[:-1,-1] /  tableau.iloc[:-1,pivot_column]
            default_picking_position = 0  # 0 stands for the minimum one
            # count how many negative ratio this column has.
            negative_num = len([i for i in tableau['ratio'].to_list() if i < 0])
            # the final target is the index of smallest positive number in this column
            picking_position = default_picking_position + negative_num
            pivot_row = np.argsort(tableau['ratio'].to_numpy())[picking_position]
            return pivot_row

        if multi_opt_index == -1:
            column_index = _pivot_column()
        else:
            column_index = multi_opt_index
        row_index = _pivot_row(column_index)
        
        tableau = tableau.drop(columns=['ratio'])
        pivot_key = tableau.iloc[row_index, column_index]
        # check if the pivot_key is unit 1
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
    
    def solve(self):
        """
        main function to execute algorithm
        """
        answers = []
        equality_constraints = self._slack_var()
        tableau = self.tableau(equality_constraints)
        objective_row = (tableau.iloc[-1,:-1]*-1).to_numpy()
        done = False
        first_done = True
        multi_opt_flag = -1
        while not done:
            # case1: Maximization problem with one solution
            next_tableau, done = self.single_pivot(tableau, multi_opt_flag)
            tableau = next_tableau
            if done:
                # for the first time, record non-basic varible column position
                if first_done:
                    self.non_basic_var_column = self._detect_non_basic_vars(tableau)
                    first_done = False
                # Check the multiple optimal
                multi_opt_flag = self._multiple_optimal(tableau)
                if multi_opt_flag >= 0:
                    # case2: if multiple optimal detected, continue pivoting
                    answers.append(self._read_result(tableau, objective_row))
                    done = False
        answers.append(self._read_result(tableau,objective_row))
        save_answer(answers)
        print("finished!")


if __name__ == "__main__":
    path = "input.txt"
    info = extract_info(input_path=path)
    lp_solver = LP_Solver(info)
    lp_solver.solve()
