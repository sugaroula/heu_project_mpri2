import iohinspector
import ioh
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os



class CatherinotSotiriou:

    def __init__(self, budget: int = 10_000, mutation_rate: float = 0.1, restart_rate: float = 0.001,  mean_budget_restart: float = 100, var_budget_restart: float = 20, bias_rate: float = 0.9, best_param: bool = True):
        self.budget = budget
        self.mutation_rate = mutation_rate # probability to flip each bit
        self.restart_rate = restart_rate # probability to start a restart at each iteration
        self.mean_budget_restart = mean_budget_restart # mean for the budget allocated to a restart
        self.var_budget_restart = var_budget_restart # var for the budget allocated to a restart
        self.bias_rate = bias_rate # bias towards favorite
        self.best_param = best_param # if best_param = True then all the parameters are automatically set to what we found were the optimal values for the parameters

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        size = problem.meta_data.n_variables

        if self.best_param :
            self.mutation_rate = 1/size  # best chance at flipping only one bit at once
            self.restart_rate = 10/self.budget
            self.mean_budget_restart = self.budget/100
            self.var_budget_restart = self.mean_budget_restart/5
            self.bias_rate = 0.9

        print(f"{self.bias_rate}, {self.restart_rate}, {self.mean_budget_restart}, {self.var_budget_restart}, {self.budget}")

        # Initialize our favorite and our backup with random samples
        favorite = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        backup = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        result_fav = problem(favorite)
        result_backup = problem(backup)

        if result_backup >= result_fav: # backup is better than favorite, we swap
            copy = favorite
            favorite = backup
            backup = copy
            result_copy = result_fav
            result_fav = result_backup
            result_backup = result_copy

        # For stats : TO REMOVE
        
        count_restart = 0
        count_useful_restart = 0
        cost_restart = 0
        count_backup_picked = 0
        count_backup_better = 0
        

        # For debugging purposes
        # TO REMOVE
        # print(f"CatSo called w budget {self.budget}, mut {self.mutation_rate}, restart rate {self.restart_rate}, budget_restart N({self.mean_budget_restart},{self.var_budget_restart})\n")

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() <= self.restart_rate:
                
                # We determine the budget allocated to the restart. It follows a gaussian distribution of mean and var specified in the parameters
                budget_restart = np.floor(np.random.normal(loc = self.mean_budget_restart, scale = self.var_budget_restart))


                # for stats : REMOVE
                
                if problem.state.evaluations + budget_restart >= self.budget :
                    cost_restart += self.budget - problem.state.evaluations
                else :
                    cost_restart += budget_restart
                

                # We randomly sample a candidate for restart :
                candidate_restart = np.random.randint(0, 2, size=problem.meta_data.n_variables)
                result_restart = problem(candidate_restart)

                # We exploit our restart
                while budget_restart > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found: # We do have to stay in budget. We have to check in case the restart was started near the end
                    
                    budget_restart -= 1
                    new_candidate_restart = candidate_restart.copy()

                    for i in range(size):
                        if np.random.random() <= self.mutation_rate:
                            new_candidate_restart[i] = 1 - new_candidate_restart[i]  # Flip bit (0->1, 1->0)
                    
                    new_result_restart = problem(new_candidate_restart)

                    if new_result_restart >= result_restart:
                        candidate_restart = new_candidate_restart
                        result_restart = new_result_restart
                
                # for stats TO COMMENT : REMOVE
                count_restart +=1
                

                # At the end of our restart, we see if it was useful :
                if result_restart >= result_backup :
                    # for stats TO COMMENT : REMOVE
                    count_useful_restart += 1

                    if result_restart >= result_fav :
                        # the restart is even better than the favorite ! It becomes the new favorite, and the favorite becomes the backup
                        backup = favorite
                        result_backup = result_fav
                        favorite = candidate_restart
                        result_fav = result_restart

                    else :
                        # the restart is better than the backup. It replaces it
                        backup = candidate_restart
                        result_backup = result_restart
            # Restart finished

            else : # no restart
                if np.random.random() <= self.bias_rate: # we pick the favorite
                    candidate = favorite.copy()

                    for i in range(size):
                        if np.random.random() <= self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                    
                    new_result = problem(candidate)

                    if new_result >= result_fav : # new candidate is better than fav, we keep it
                        favorite = candidate
                        result_fav = new_result


                else : # we pick the backup
                    # for stats TO COMMENT : REMOVE
                    count_backup_picked += 1

                    candidate = backup.copy()

                    for i in range(len(candidate)):
                        if np.random.random() <= self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                    
                    new_result = problem(candidate)

                    if new_result >= result_backup : # new candidate is better than current backup, we keep it
                        if new_result >= result_fav : # the backup has become better than the favorite. We swap
                            # for stats TO COMMENT : REMOVE
                            count_backup_better += 1

                            backup = favorite
                            favorite = candidate
                            result_backup = result_fav
                            result_fav = new_result

                        else : # we get a better a backup
                            backup = candidate
                            result_backup = new_result

        print(f"ratio restarts : {count_restart} / {self.budget} = {count_restart / self.budget}")
        if count_restart > 0 :
            print(f"ratio useful restarts : {count_useful_restart} / {count_restart} = {count_useful_restart / count_restart}")
        print(f"cost restart {cost_restart}")
        if self.budget - cost_restart > 0 :
            print(f"backup picked : {count_backup_picked} / {self.budget - cost_restart} = {count_backup_picked / (self.budget - cost_restart)}")
        else : 
            print("\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!! cost_restart = budget !!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n")
        


def test_CatSo_pop2_bias_plot_MaxCut(bias_rate, restart_rate, mean_budget_restart, var_budget_restart, budget):
    fids = [k if "MaxCut" in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
    fids = [fid for fid in fids if fid is not None]
    problem = ioh.get_problem(fids[1], problem_class=ioh.ProblemClass.GRAPH)
    print("=" * 20)
    CatherinotSotiriou()(problem)


#test_CatSo_pop2_bias_plot_MaxCut(0.7, 0.001, 500, 20, 10000)
budget = 50000
test_CatSo_pop2_bias_plot_MaxCut(0.9, 10/budget, budget/10, (budget/10)/50, budget)
