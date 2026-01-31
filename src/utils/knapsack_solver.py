
import numpy as np
from ortools.linear_solver import pywraplp

def solve_knapsack_ortools(values, weights, budget):
    """
    Solves the 0/1 Knapsack problem using Google OR-Tools (MIP solver).
    Args:
        values (np.ndarray): Array of item values.
        weights (np.ndarray): Array of item weights.
        budget (float): Total budget capacity.
    Returns:
        np.ndarray: Binary array of selected items (1=selected, 0=not).
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('GLOP') 
    
    n = len(values)
    x = {}
    for i in range(n):
        x[i] = solver.IntVar(0, 1, f'x_{i}')
        
    # Constraint: sum(w_i * x_i) <= budget
    constraint = solver.RowConstraint(0, float(budget), 'budget')
    for i in range(n):
        constraint.SetCoefficient(x[i], float(weights[i]))
        
    # Objective: Maximize sum(v_i * x_i)
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], float(values[i]))
    objective.SetMaximization()
    
    status = solver.Solve()
    
    selected = np.zeros(n, dtype=np.int32)
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(n):
            if x[i].solution_value() > 0.5:
                selected[i] = 1
    return selected
