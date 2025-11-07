import iohinspector
import ioh
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os



class CatherinotSotiriou:

    def __init__(self, budget: int = 10_000, restart_rate: float = 0.001,  mean_budget_restart: float = 100, var_budget_restart: float = 20, bias_rate: float = 0.9, best_param: bool = True):
        self.budget = budget
        self.restart_rate = restart_rate # probability to start a restart at each iteration
        self.mean_budget_restart = mean_budget_restart # mean for the budget allocated to a restart
        self.var_budget_restart = var_budget_restart # var for the budget allocated to a restart
        self.bias_rate = bias_rate # bias towards favorite
        self.best_param = best_param # if best_param = True then all the parameters are automatically set to what we found were the optimal values for the parameters

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        size = problem.meta_data.n_variables
        self.mutation_rate = 1/size  # best chance at flipping only one bit at once

        if self.best_param :
            self.restart_rate = 10/self.budget
            self.mean_budget_restart = self.budget/100
            self.var_budget_restart = self.mean_budget_restart/5
            self.bias_rate = 0.9

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

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() <= self.restart_rate:
                
                # We determine the budget allocated to the restart. It follows a gaussian distribution of mean and var specified in the parameters
                budget_restart = np.floor(np.random.normal(loc = self.mean_budget_restart, scale = self.var_budget_restart))
                
                # We randomly sample a candidate for restart :
                candidate_restart = np.random.randint(0, 2, size=problem.meta_data.n_variables)
                result_restart = problem(candidate_restart)

                # We exploit our restart
                while budget_restart > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found: # We do have to stay in budget. We have to check in case the restart was started near the end
                    
                    budget_restart -= 1
                    new_candidate_restart = candidate_restart.copy()

                    no_bit_flipped = True
                    for i in range(size):
                        if np.random.random() <= self.mutation_rate:
                            new_candidate_restart[i] = 1 - new_candidate_restart[i]  # Flip bit (0->1, 1->0)
                            no_bit_flipped = False
                    if no_bit_flipped :
                        i = np.random.randint(0, size)
                        new_candidate_restart[i] = 1 - new_candidate_restart[i]

                    
                    new_result_restart = problem(new_candidate_restart)

                    if new_result_restart >= result_restart:
                        candidate_restart = new_candidate_restart
                        result_restart = new_result_restart
                

                # At the end of our restart, we see if it was useful :
                if result_restart >= result_backup :
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

                    no_bit_flipped = True
                    for i in range(size):
                        if np.random.random() <= self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                            no_bit_flipped = False
                    if no_bit_flipped :
                        i = np.random.randint(0, size)
                        candidate[i] = 1 - candidate[i]
                    
                    new_result = problem(candidate)

                    if new_result >= result_fav : # new candidate is better than fav, we keep it
                        favorite = candidate
                        result_fav = new_result


                else : # we pick the backup
                    candidate = backup.copy()

                    no_bit_flipped = True
                    for i in range(len(candidate)):
                        if np.random.random() <= self.mutation_rate:
                            candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
                            no_bit_flipped = False
                    if no_bit_flipped :
                        i = np.random.randint(0, size)
                        candidate[i] = 1 - candidate[i]
                    
                    new_result = problem(candidate)

                    if new_result >= result_backup : # new candidate is better than current backup, we keep it
                        if new_result >= result_fav : # the backup has become better than the favorite. We swap
                            backup = favorite
                            favorite = candidate
                            result_backup = result_fav
                            result_fav = new_result

                        else : # we get a better a backup
                            backup = candidate
                            result_backup = new_result

        


