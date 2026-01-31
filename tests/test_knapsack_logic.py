
import unittest
import numpy as np
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel

class MockModel(MSMEQEExpansionModel):
    def __init__(self):
        pass

class TestKnapsackLogic(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()

    def test_greedy_vs_knapsack_divergence(self):
        """
        Classic Case where Greedy fails but Knapsack succeeds.
        Budget W = 10.
        Item A: v=10, w=10 (Density=1). Greedy picks this first. Remainder=0. Total=10.
        Item B-K: v=6, w=1 (Density=6). 10 items.
        Knapsack should pick B-K. Total=60.
        Greedy should pick A. Total=10.
        """
        budget = 10.0
        
        # Item A (The "Trap" for Greedy)
        # Items B-K (The "Gold" for Knapsack)
        values = np.array([10.0] + [6.0] * 10)
        weights = np.array([10.0] + [1.0] * 10)
        
        # 1. Run Greedy Logic (Simulated)
        # Sort by Value Descending
        indices = np.argsort(-values)
        greedy_selection = np.zeros(len(values))
        rem = budget
        for idx in indices:
            if weights[idx] <= rem:
                greedy_selection[idx] = 1
                rem -= weights[idx]
        
        greedy_value = np.sum(values * greedy_selection)
        print(f"\nGreedy Valid Selection: {greedy_selection}")
        print(f"Greedy Total Value: {greedy_value}")
        
        # 2. Run Actual Knapsack Implementation using OR-Tools
        from src.utils.knapsack_solver import solve_knapsack_ortools
        knapsack_selection = solve_knapsack_ortools(values, weights, budget)
        
        knapsack_value = np.sum(values * knapsack_selection)
        print(f"Knapsack Valid Selection: {knapsack_selection}")
        print(f"Knapsack Total Value: {knapsack_value}")
        
        # 3. Assert Divergence
        self.assertEqual(greedy_value, 10.0, "Greedy logic should pick the single heavy item")
        self.assertEqual(knapsack_value, 60.0, "Knapsack logic should pick the many light items")
        self.assertNotEqual(greedy_selection[0], knapsack_selection[0], "Selection vectors must differ")
        print("SUCCESS: Divergence Proven.")

if __name__ == '__main__':
    unittest.main()
