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
        
"""
OneMax = ioh.get_problem(1, instance=1, dimension=300, problem_class=ioh.ProblemClass.PBO)
CatSo_only_restart()(OneMax)
print(f"Have we stayed in budget ? eval = {OneMax.state.evaluations} < 50 000 ? best found = {OneMax.state.current_best.y}")
"""



class CatSo_var_budget_restart:

    def __init__(self, budget: int = 50_000, mutation_rate: float = 0.1, restart_rate: float = 0.002,  mean_budget_restart: float = 500, var_budget_restart: float = 5, **kwargs): # mutation rate, random restart rate, random restart budget rate
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.mean_budget_restart = mean_budget_restart
        self.var_budget_restart = var_budget_restart

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        # Initialize with random sample
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)
        count_restart = 0
        count_useful_restart = 0

        # For debugging purposes
        # TO COMMENT
        # print(f"CatSo called w budget {self.budget}, mut {self.mutation_rate}, restart rate {self.restart_rate}, budget_restart N({self.mean_budget_restart},{self.var_budget_restart})\n")

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() < self.restart_rate:
                
                budget_restart = np.floor(np.random.normal(loc = self.mean_budget_restart, scale = self.var_budget_restart))

                #print(f"budget restart : {budget_restart}")

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
                
                # for stats :
                count_restart +=1
                if result_restart == (problem.state.current_best.y): 
                    count_useful_restart += 1
            # Restart finished

            else : # no restart
                # Mutation operator : each bit is flipped with proba "mutation_rate"
                candidate = problem.state.current_best.x.copy()

                for i in range(len(candidate)):
                    if np.random.random() < self.mutation_rate:
                        candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

                problem(candidate)

        
        # print(f"ratio restarts : {count_restart} / {self.budget} = {count_restart / self.budget}")
        # print(f"ratio useful restarts : {count_useful_restart} / {count_restart} = {count_useful_restart / count_restart}")

"""
OneMax = ioh.get_problem(1, instance=1, dimension=300, problem_class=ioh.ProblemClass.PBO)
CatSo_var_budget_restart()(OneMax)
print(f"Have we stayed in budget ? eval = {OneMax.state.evaluations} < 50 000 ?\n Best found = {OneMax.state.current_best.y}")
"""


#MaxCut = ioh.get_problem("MaxCut", instance=1, dimension=10, problem_class=ioh.ProblemClass.GRAPH)
"""
fids = [k if "MaxCut" in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
fids = [fid for fid in fids if fid is not None]
for fid in fids:
        problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        print("=" * 20)
        print(f"{problem.meta_data.name}")
        CatSo_var_budget_restart(restart_rate=25/50000, mean_budget_restart=200, var_budget_restart=5)(problem)
        print(f"Have we stayed in budget ? eval = {problem.state.evaluations} < 50 000 ?")
        print(f"Best found = {problem.state.current_best.y}")
        print(f"Optimum : {problem.optimum.y}")
        print(f"Number of variables: {problem.meta_data.n_variables}")
        #print(f"Lower bounds: {problem.bounds.lb}")
        #print(f"Upper bounds: {problem.bounds.ub}")
        print("=" * 20)
"""