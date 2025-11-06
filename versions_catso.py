#import iohinspector
import ioh
import numpy as np
#import pandas as pd
#import polars as pl
#from tqdm import tqdm 
import matplotlib.pyplot as plt
#import os


class CatSo_only_restart:

    def __init__(self, budget: int = 50_000, mutation_rate: float = 0.1, restart_rate: float = 0.002,  budget_restart: int = 500, **kwargs): # mutation rate, random restart rate, random restart budget rate
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.budget_restart = budget_restart

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        # Initialize with random sample
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() < self.restart_rate:
                
                budget_restart = self.budget_restart
                candidate_restart = np.random.randint(0, 2, size=problem.meta_data.n_variables)
                result_restart = problem(candidate_restart)

                while budget_restart > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found: # We do have to stay in budget
                    
                    budget_restart -= 1
                    new_candidate_restart = candidate_restart

                    for i in range(len(candidate)):
                        if np.random.random() < self.mutation_rate:
                            new_candidate_restart[i] = 1 - new_candidate_restart[i]  # Flip bit (0->1, 1->0)
                    
                    new_result_restart = problem(new_candidate_restart)

                    if new_result_restart >= result_restart:
                        candidate_restart = new_candidate_restart
                        result_restart = new_result_restart
            # Restart finished

            else : # no restart
                # Mutation operator : each bit is flipped with proba "mutation_rate"
                candidate = problem.state.current_best.x.copy()

                for i in range(len(candidate)):
                    if np.random.random() < self.mutation_rate:
                        candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

                problem(candidate)
        

OneMax = ioh.get_problem(1, instance=1, dimension=300, problem_class=ioh.ProblemClass.PBO)
CatSo_only_restart()(OneMax)
print(f"Have we stayed in budget ? eval = {OneMax.state.evaluations} < 50 000 ? best found = {OneMax.state.current_best.y}")