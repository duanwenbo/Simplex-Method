from optimization import LP_Solver
from initalization import extract_info


def optimize(question_path):
    info = extract_info(question_path)
    lp_solver = LP_Solver(info)
    lp_solver.solve()

if __name__ == "__main__":
    question_path = 'input.txt'
    optimize(question_path)
