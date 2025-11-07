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
        candidate = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(candidate)

        self.mutation_rate = 1/problem.meta_data.n_variables

        # For stats :
        """
        count_restart = 0
        count_useful_restart = 0
        """

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
                """
                count_restart +=1
                if result_restart == (problem.state.current_best.y): 
                    count_useful_restart += 1
                """
            # Restart finished

            else : # no restart
                # Mutation operator : each bit is flipped with proba "mutation_rate"
                candidate = problem.state.current_best.x.copy()

                for i in range(len(candidate)):
                    if np.random.random() < self.mutation_rate:
                        candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

                problem(candidate)

        """
        print(f"ratio restarts : {count_restart} / {self.budget} = {count_restart / self.budget}")
        if count_restart > 0 :
            print(f"ratio useful restarts : {count_useful_restart} / {count_restart} = {count_useful_restart / count_restart}")
        """

"""
OneMax = ioh.get_problem(1, instance=1, dimension=300, problem_class=ioh.ProblemClass.PBO)
CatSo_var_budget_restart()(OneMax)
print(f"Have we stayed in budget ? eval = {OneMax.state.evaluations} < 50 000 ?\n Best found = {OneMax.state.current_best.y}")
"""


#MaxCut = ioh.get_problem("MaxCut", instance=1, dimension=10, problem_class=ioh.ProblemClass.GRAPH)

def test_CatSo_var_MaxCut_instances():
    fids = [k if "MaxCut" in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
    fids = [fid for fid in fids if fid is not None]

    for fid in fids:
            problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
            """
            print("=" * 20)
            print(f"{problem.meta_data.name}")
            """
            CatSo_var_budget_restart(restart_rate=25/50000, mean_budget_restart=200, var_budget_restart=5)(problem)
            """
            print(f"Have we stayed in budget ? eval = {problem.state.evaluations} < 50 000 ?")
            print(f"Best found = {problem.state.current_best.y}")
            print(f"Optimum : {problem.optimum.y}")
            print(f"Number of variables: {problem.meta_data.n_variables}")
            #print(f"Lower bounds: {problem.bounds.lb}")
            #print(f"Upper bounds: {problem.bounds.ub}")
            print("=" * 20)
            """

# for stats
def test_CatSo_var_MaxCut_bench(restart_rate, mean_budget_restart, var_budget_restart, budget):
    fids = [k if "MaxCut" in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
    fids = [fid for fid in fids if fid is not None]
    problem = ioh.get_problem(fids[1], problem_class=ioh.ProblemClass.GRAPH)
    best_found_list = []
    print("=" * 20)
    for i in range(20):
        CatSo_var_budget_restart(budget=budget, restart_rate=restart_rate, mean_budget_restart=mean_budget_restart, var_budget_restart=var_budget_restart)(problem)
        best_found_list.append(problem.state.current_best.y)
        problem.reset()
    print(f"Best found avg : {sum(best_found_list) / len(best_found_list)}")
    print("=" * 20)


test_CatSo_var_MaxCut_bench(20/1000, 1000/10, 5, 1000)






class CatSo_pop2_bias_rate:

    def __init__(self, budget: int = 50_000, mutation_rate: float = 0.1, restart_rate: float = 0.002,  mean_budget_restart: float = 500, var_budget_restart: float = 5, bias_rate: float = 0.9, **kwargs): # mutation rate, random restart rate, random restart budget rate
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.mean_budget_restart = mean_budget_restart
        self.var_budget_restart = var_budget_restart
        self.bias_rate = bias_rate

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        self.mutation_rate = 1/problem.meta_data.n_variables

        # Initialize with random sample
        favorite = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        backup = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        result_fav = problem(favorite)
        result_backup = problem(backup)

        if result_backup == problem.state.current_best.y: # backup is better than fav, we swap
            copy = favorite
            favorite = backup
            backup = copy

        # init candidate
        candidate = favorite


        # For stats :
        """
        count_restart = 0
        count_useful_restart = 0
        cost_restart = 0
        count_backup_picked = 0
        count_backup_better = 0
        """

        # For debugging purposes
        # TO COMMENT
        # print(f"CatSo called w budget {self.budget}, mut {self.mutation_rate}, restart rate {self.restart_rate}, budget_restart N({self.mean_budget_restart},{self.var_budget_restart})\n")

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() < self.restart_rate:
                
                """
                print("-" * 10)
                print("restart !")
                """
                
                budget_restart = np.floor(np.random.normal(loc = self.mean_budget_restart, scale = self.var_budget_restart))


                # for stats :
                """
                if problem.state.evaluations + budget_restart >= self.budget :
                    cost_restart += self.budget - problem.state.evaluations
                else :
                    cost_restart += budget_restart
                """
                # print(f"budget restart : {budget_restart}")

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
                
                # for stats TO COMMENT :
                # count_restart +=1
                
                """
                print(f"result restart = {result_restart}")
                print(f"result favorite = {result_fav}, result backup = {result_backup}")
                """

                if result_restart >= result_backup :
                    # for stats TO COMMENT :
                    # count_useful_restart += 1

                    if result_restart >= result_fav :
                        # the restart is even better than the favorite. It becomes the fav, and the fav becomes the backup
                        backup = favorite
                        result_backup = result_fav
                        favorite = candidate_restart
                        result_fav = result_restart

                        # print(f"restart super useful ! result favorite = {result_fav}, result backup = {result_backup}")

                    else :
                        # the restart is better than the backup. It replaces it
                        backup = candidate_restart
                        result_backup = result_restart

                        # print(f"restart useful ! result backup = {result_backup}")

            # Restart finished

            else : # no restart
                if np.random.random() < self.bias_rate: # we pick the favorite
                    candidate = favorite

                    for i in range(len(candidate)):
                        if np.random.random() < self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                    
                    new_result = problem(candidate)

                    if new_result >= result_fav : # new candidate is better than fav, we keep it
                        favorite = candidate
                        result_fav = new_result
                    
                    """
                    print("favorite picked")
                    print(f"result favorite = {result_fav}, result backup = {result_backup}")
                    """

                else : # we pick the backup
                    # for stats TO COMMENT :
                    # count_backup_picked += 1

                    candidate = backup

                    for i in range(len(candidate)):
                        if np.random.random() < self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                    
                    new_result = problem(candidate)

                    if new_result >= result_backup : # new candidate is better than current backup, we keep it
                        if new_result >= result_fav : # the backup has become better than the favorite. We swap
                            # for stats TO COMMENT :
                            # count_backup_better += 1

                            backup = favorite
                            favorite = candidate
                            result_backup = result_fav
                            result_fav = new_result
                        else : # we get a better a backup
                            backup = candidate
                            result_backup = new_result

                    """
                    print("backup picked !!!!!")
                    print(f"result favorite = {result_fav}, result backup = {result_backup}")
                    """
                    
        """
        print(f"ratio restarts : {count_restart} / {self.budget} = {count_restart / self.budget}")
        if count_restart > 0 :
            print(f"ratio useful restarts : {count_useful_restart} / {count_restart} = {count_useful_restart / count_restart}")
        print(f"cost restart {cost_restart}")
        if self.budget - cost_restart > 0 :
            print(f"backup picked : {count_backup_picked} / {self.budget - cost_restart} = {count_backup_picked / (self.budget - cost_restart)}")
        else : 
            print("\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!! cost_restart = budget !!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n")
        """




def test_debug(budget):
    OneMax = ioh.get_problem(1, instance=1, dimension=300, problem_class=ioh.ProblemClass.PBO)
    CatSo_pop2_bias_rate(budget=budget, restart_rate=1/5, mean_budget_restart=2, var_budget_restart=0)(OneMax)
    print(f"Have we stayed in budget ? eval = {OneMax.state.evaluations} <= {budget} ?\n Best found = {OneMax.state.current_best.y}")


#test_debug(40)
    