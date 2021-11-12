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
import copy

class LP_Solver:
    def __init__(self, key_info:dict):
        self.constraints = key_info['constraints']
        self.objective_func = key_info['objective_function']
        self.type = key_info['optimization_type']
        self.non_basic_var_column = []  # for multiple opt use
        self.inital_tableau = ""  # contains objective function info
    
    def _read_coefficient(self, constraint:str, key_variable:str) -> float:
        """
        read the coefficient of a specified variable in an equation
        """
        constraint = constraint.replace(" ","")
        if key_variable == "solution":
            coefficient = re.search(r'(?<=\=).*$', constraint).group()
        else:
            if re.search(key_variable, constraint):
                coefficient = re.search(r'-*?\d*\.?\d*?(?={})'.format(key_variable), constraint).group()
                if coefficient == "":
                    coefficient = 1
                elif coefficient == "-":
                    coefficient = -1
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
                constraint = re.sub(r'\>', "- s_{}".format(inequality_count), constraint)
                inequality_count += 1
            else:
                pass
            equality_constraints.append(constraint)
        return equality_constraints
    
    @record_process
    def _artificial_vars(self, tableau:pd.DataFrame):

        @record_process
        def _positive_sol(negative_row:int, tableau:pd.DataFrame):
            # ensure the solution column is positive, try to find an initial feasiable solution 
            tableau.iloc[negative_row,:] *= -1
            return tableau
        
        # check if the initial tabeau is sufficient or not to get a feasible solution
        def _detect_sufficiency(tableau:pd.DataFrame):
            sufficieny = True
            # find if the solution column contains negative or not
            negative_sol_index = []
            for index, num in enumerate(tableau.solution.to_list()):
                if num < 0:
                    negative_sol_index.append(index)
            # if it contains, regulate the related solution to positive
            if len(negative_sol_index) > 0:
                for index in negative_sol_index:
                    tableau = _positive_sol(index, tableau)
            # check if we can read an initial feasible solution from the tableau
            feasible_basic_sol = self._read_result(tableau)
            solutions = list(feasible_basic_sol.values())[:-1] # if it has negative solution
            if sorted(solutions)[0] < 0:
                sufficieny = False
            else:
                sufficieny = True
            
            for constraint in self.constraints:
                if re.search(r'[^\>\<]\=', constraint):
                    sufficieny = False
            
            return sufficieny

        if not _detect_sufficiency(tableau):
            # if the tableau need artificial vars
            # the number of artificial vars = total number equations CURRENTLY
            artificial_var_num = len(self.constraints)
            # append an indentity matrix in the original tableau
            for i in range(artificial_var_num):
                tableau.insert(tableau.shape[1]-1, "a_{}".format(i), 0)
                tableau.iloc[i,-2] = 1
                # change object row as well to show hand-done arithmetic more clear
                if self.type == "maximisation":
                    tableau.iloc[-1,-2] = 10
                else:
                    tableau.iloc[-1,-2] = -10
        return tableau
    
    @record_process
    def _normalize_artificial_column(self, tableau:pd.DataFrame) -> pd.DataFrame:
        """"
        make artificial vars column to a unit vector column if it contains artificial vars
        """
        # 1. find the position of artificial variables
        artificial_coulmn_index = []
        for index, variable in enumerate(list(tableau)):
            if re.search(r'a', variable):
                artificial_coulmn_index.append(index)
        if len(artificial_coulmn_index) > 0:
            # 2. do pivoting on these positions to make an unit vector column
            for index, column in enumerate (artificial_coulmn_index):
                tableau.iloc[-1,:] -= tableau.iloc[index, :]*tableau.iloc[-1,column]
        return tableau
    
    def _unit_vector(self, column:pd.DataFrame) -> bool:
        """
        determine if a column is an unit vector
        This is used for 1. reading the result; 2. determining multiple optimal
        """
        modulus = np.linalg.norm(column.to_numpy())
        if modulus != 1.:
            return False
        else:
            return True

    def _read_result(self, tableau:pd.DataFrame) -> dict:
        """
        read one basic feasible solution from the tableau
        """
        answer = {}
        for index in range(tableau.shape[1]-1):
            # check if the column is a unit vector
            current_column = tableau.iloc[:, index]
            if not self._unit_vector(current_column):
                answer["{}".format(tableau.columns[index])] = 0
            else:
                answer_row = current_column[current_column !=0].index[0]
                answer["{}".format(tableau.columns[index])] = tableau.iloc[answer_row,-1] * tableau.iloc[answer_row, index]
        
        # find the Z value according to Z = C^T * X
        # reshape answer vector as nx1 array
        answer_vector = np.array(list(answer.values()))
        answer_vector = np.expand_dims(answer_vector, axis=1)

        # reshape objective vector as 1xn array
        objective_row = (self.inital_tableau.iloc[-1,:-1]*-1).to_numpy()
        # solve the length difference problem caused by artificial vars
        len_difference = len(answer_vector) - len(objective_row)
        objective_row = np.append(objective_row, [0]*len_difference)
        objective_vector = np.expand_dims(objective_row, axis=0)
       
        # Z = C^T * X
        z = np.dot(objective_vector, answer_vector)[0][0]
        answer['z'] = z
        return answer
    
    def _multiple_optimal(self, tableau:pd.DataFrame) -> int:
        """
        Check if the tableau has multiple optimal
        if yes, return the index of the  pivot column
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
            objective_row = tableau.iloc[-1].to_numpy()[:-1]
            # check if the input is correct, if not, output the hint 
            assert self.type == "maximisation" or self.type == "minimisation", "please input 'maximisation' or 'minimisation' "
            # pivot on the column with the most negative/positive element
            if self.type == "maximisation":
                pivot_column = np.argsort(objective_row)[0]
            elif self.type == "minimisation":
                pivot_column = np.argsort(objective_row)[-1]
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
        for value in tableau.iloc[-1,:-1]:
            if self.type == "maximisation":  # for maximisation
                if value < 0:
                    done = False
                    break
                else:
                    done = True 
            else:  # for minimisation
                if value > 0:
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
        # 1. chaneg inequalities to equalities
        equality_constraints = self._slack_var()
        # 2. initalize the tableau
        tableau = self.tableau(equality_constraints)
        self.inital_tableau = copy.copy(tableau) # store this version tableau
        # 3. check if artificial variables needed
        tableau = self._artificial_vars(tableau)
        # 4. normalize the artificial columns if it contains
        tableau = self._normalize_artificial_column(tableau)

        done = False
        first_done = True
        multi_opt_flag = -1

        while not done:
            # case1: normal problem with one solution
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
                    answers.append(self._read_result(tableau))
                    done = False
        answers.append(self._read_result(tableau))
        save_answer(answers, self.inital_tableau)
        print("finished!")


if __name__ == "__main__":
    path = "input.txt"
    info = extract_info(input_path=path)
    lp_solver = LP_Solver(info)
    lp_solver.solve()
