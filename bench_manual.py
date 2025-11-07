#import iohinspector
import ioh
import numpy as np
#import pandas as pd
#import polars as pl
#from tqdm import tqdm 
import matplotlib.pyplot as plt
#import os


def list_graph_problems():
    """Print all GRAPH problems (fid -> name)."""
    gp = ioh.ProblemClass.GRAPH.problems  # dict: fid -> name
    print(f"GRAPH problems available: {len(gp)}")
    for fid, name in sorted(gp.items()):
        print(f"{fid:4d}  {name}")


def find_fids_by_keywords(*keywords, limit_per_kw=1):
    """
    Return a dict {kw: [fid,...]} where fid names contain kw (case-insensitive).
    limit_per_kw limits how many fids we pick for each keyword.
    """
    gp = ioh.ProblemClass.GRAPH.problems
    out = {}
    for kw in keywords:
        kw_l = kw.lower()
        fids = [fid for fid, name in gp.items() if kw_l in name.lower()]
        out[kw] = fids[:limit_per_kw]
    return out


def get_graph_problem(fid, instance=1):
    return ioh.get_problem(fid, instance=instance, problem_class=ioh.ProblemClass.GRAPH)



def run_reps(problem, algo_cls, n_reps=3, **kwargs):
    traces = []
    for _ in range(n_reps):
        problem.reset()
        algo = algo_cls(**kwargs)
        trace = algo(problem)       # <-- now returns np.array(evals, best_y)
        traces.append(trace)
    return traces

def plot_mean_traces(traces, title):
    # unify onto a common x-grid for a mean curve
    xmax = int(max(t[-1,0] for t in traces if t.size))
    grid = np.linspace(1, xmax, 200).astype(int)
    interps = []
    for t in traces:
        xs, ys = t[:,0], t[:,1]
        yg = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        interps.append(yg)
    mean_curve = np.mean(np.vstack(interps), axis=0)

    plt.figure(figsize=(10,6))
    for y in interps:
        plt.plot(grid, y, alpha=0.2)
    plt.plot(grid, mean_curve, lw=2, label="mean")
    plt.xlabel("Evaluations"); plt.ylabel("Best-so-far")
    plt.title(title); plt.grid(True); plt.legend(); plt.show()


def benchmark_submodular_problem_class(algorithm, n_reps: int = 5, budget: int = 500, problems="MaxCut", root_dir: str = ".", **kwargs):
    print(1)
    fids = [k if problems in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
    print(fids)
    fids = [fid for fid in fids if fid is not None]
    logger = ioh.logger.Analyzer(
        root=root_dir,                  
        folder_name=f"{problems}_{algorithm.__name__}",         
        algorithm_name=algorithm.__name__, 
    )

    # plotting results :
    fig, ax = plt.subplots(figsize=(16, 9))

    # for statistics of finding the optimum :
    count_fail = 0
    count_success = 0
    time_to_success = []
    best_sol_found = []

    for k, v in kwargs.items():
        logger.add_experiment_attribute(k, str(v))
    for fid in fids:
        problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        problem.attach_logger(logger)

        for _ in range(n_reps):
            algorithm(budget=budget, **kwargs)(problem)
            # problem.reset()

            # statistics : 
            if problem.state.optimum_found :
                count_success += 1
                time_to_success.append(problem.state.evaluations)
            else : 
                count_fail += 1
                best_sol_found.append(problem.state.current_best.y)


            # plotting : does not work
            # I would really like to plot so I can see how the algorithm behaves
            """
            logger.close()
            # the mistake might stem from here. I closed the logger in the first it. The next iterations are not going to like it. But I have to close the logger before calling the manager and I have to do all of that before the problem is reset don't I ?
            
            manager = iohinspector.DataManager()
            manager.overview
            manager.add_folder(f"{problems}_{algorithm.__name__}")
            df = manager.load(True, True)

            dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean'])
            
            ax.set_xlim(1, 5000)
            ax.set_title(f"Test on instance {problem.meta_data.instance} of {problem.meta_data.name} (ID {problem.meta_data.problem_id})")
            """

            # resetting :
            problem.reset()


        # at the end : we show the results
    logger.close()
    """
    logger.close()
    
    manager = iohinspector.DataManager()
    manager.overview
    manager.add_folder(f"{problems}_{algorithm.__name__}")
    df = manager.load(True, True)

    dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean'])
    
    ax.set_xlim(1, 5000)
    ax.set_title("Test on "^problems)
    # plotting :
    #ax.get_legend().remove() # to be more readable
    # plotting legend for this problem
    
    plt.show()
    """

    # printing success rate :
    print("=" * 60)
    print(f"\nSuccess count : {count_success}\nFail count : {count_fail}\n")
    print("=" * 60)
    if count_success > 0 :
        print(f"Time to success :\n{time_to_success}")
    print("=" * 60)
    print(f"Best solution found when fail : {best_sol_found}")

    
"""
    for i, problem in enumerate(problems):
        logger = ioh.logger.Analyzer(
                root=os.getcwd(),                  # Store data in the current working directory
                folder_name=(f"PBO_Test_EA_{problem.meta_data.instance}_{problem.meta_data.name}"),         # in a folder named: 'PBO_RS_Test'
                algorithm_name="BinaryEA", # meta-data for the algorithm used to generate these results
        )

        print(f"\n{problem.meta_data.name} (Instance {problem.meta_data.instance}) (Target: {problem.optimum}):")
        print("-" * 40)

        logger.add_experiment_attribute("mutation_rate", f"{0.1}") # Add as experiment attribute
        #Adds column ????
        
        problem.attach_logger(logger)
        run_experiment(problem, BinaryEvolutionaryAlgorithm(2000, 0.1), n_runs=15, verbose=False)
        logger.close()
        
        manager = iohinspector.DataManager()
        manager.overview
        manager.add_folder(f"PBO_Test_EA_{problem.meta_data.instance}_{problem.meta_data.name}")
        df = manager.load(True, True)
        
        # Plot results for OneMax (function_id = 1)
        dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
        ax.set_xlim(1, 5000)
        ax.set_title(f"Test on instance {problem.meta_data.instance} of {problem.meta_data.name} (ID {problem.meta_data.problem_id})")

        problem.reset()  # Reset for next use

    plt.show()

"""



def benchmark_compare_BinaryEA_PBO_pb(algorithm, n_reps: int = 5, budget: int = 50000, problems_info=[(1, "OneMax"), (2, "LeadingOnes"), (18, "LABS"), (19, "IsingRing")], **kwargs):
    for fid, name in problems_info:
        problem = ioh.get_problem(fid, instance=1, dimension=10, problem_class=ioh.ProblemClass.INTEGER)


def eval_and_record(problem, x, trace):
    """Call problem(x), append (evals, best_y) to trace, and return y."""
    y = problem(x)
    trace.append((problem.state.evaluations, problem.state.current_best.y))
    return y


class OnePlusOneEA:
    def __init__(self, budget=3000, mutation_rate=None, rng_seed=None):
        self.budget = budget; self.mr = mutation_rate
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        n = problem.meta_data.n_variables
        p = self.mr if self.mr is not None else (1 / max(n, 1))
        x = self.rng.randint(0,2,size=n)
        fx = eval_and_record(problem, x, trace)
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            y = x.copy()
            flips = self.rng.rand(n) < p
            if not flips.any():
                flips[self.rng.randint(n)] = True
            y[flips] = 1 - y[flips]
            fy = eval_and_record(problem, y, trace)
            if fy >= fx:
                x, fx = y, fy
        return np.array(trace, dtype=float)


class CatSo_only_restart:

    def __init__(self, budget: int = 50_000, mutation_rate: float = 0.1, restart_rate: float = 0.002,  budget_restart: int = 500, **kwargs): # mutation rate, random restart rate, random restart budget rate
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.budget_restart = budget_restart

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        trace = []  # <-- collect (evals, best_y) so the runner can plot

        n = problem.meta_data.n_variables
        p = self.mutation_rate if self.mutation_rate is not None else (1 / max(n, 1))

        # Initialize with random sample
        x = np.random.randint(0, 2, size=n)
        result = eval_and_record(problem, x, trace)

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            
            # Potential restart
            if np.random.random() < self.restart_rate:
                
                budget_restart = self.budget_restart
                candidate_restart = np.random.randint(0, 2, size=n)
                result_restart = eval_and_record(problem, candidate_restart, trace)

                while budget_restart > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found: # We do have to stay in budget
                    
                    budget_restart -= 1

                    # IMPORTANT: copy, then flip; and use the right length
                    new_candidate_restart = candidate_restart.copy()

                    for i in range(n):
                        if np.random.random() < p:
                            new_candidate_restart[i] = 1 - new_candidate_restart[i]  # Flip bit (0->1, 1->0)
                    
                    new_result_restart = eval_and_record(problem, new_candidate_restart, trace)

                    if new_result_restart >= result_restart:
                        candidate_restart = new_candidate_restart
                        result_restart = new_result_restart
            # Restart finished

            else : # no restart
                # Mutation operator : each bit is flipped with proba "mutation_rate"
                candidate = problem.state.current_best.x.copy()

                for i in range(n):
                    if np.random.random() < p:
                        candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

                _ = eval_and_record(problem, candidate, trace)
        return np.array(trace, dtype=float)  # <-- so run_reps/plot_* work
    


class CatSo_var_budget_restart:
    """
    Like CatSo_only_restart, but when a restart is triggered, the sub-budget
    is sampled from N(mean_budget_restart, var_budget_restart).
    """
    def __init__(self, budget: int = 50_000,
                 mutation_rate: float | None = None,
                 restart_rate: float = 0.002,
                 mean_budget_restart: float = 500,
                 var_budget_restart: float = 5,
                 rng_seed: int | None = None,
                 **kwargs):
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.mean_budget_restart = mean_budget_restart
        self.var_budget_restart = var_budget_restart
        self.rng = np.random.RandomState(rng_seed)

    def __call__(self, problem: ioh.problem.IntegerSingleObjective):
        trace = []
        n = problem.meta_data.n_variables
        p = (self.mutation_rate if self.mutation_rate is not None
             else 1 / max(n, 1))

        # start from a random point
        candidate = self.rng.randint(0, 2, size=n)
        _ = eval_and_record(problem, candidate, trace)

        while problem.state.evaluations < self.budget and not problem.state.optimum_found:

            if self.rng.rand() < self.restart_rate:
                # draw a positive integer sub-budget
                subB = int(max(1, np.floor(self.rng.normal(
                    loc=self.mean_budget_restart,
                    scale=max(1e-9, self.var_budget_restart)
                ))))
                # fresh random candidate and climb for subB steps
                cand = self.rng.randint(0, 2, size=n)
                f = eval_and_record(problem, cand, trace)

                while subB > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found:
                    subB -= 1
                    y = cand.copy()
                    flips = self.rng.rand(n) < p
                    if not flips.any():
                        flips[self.rng.randint(n)] = True
                    y[flips] = 1 - y[flips]
                    fy = eval_and_record(problem, y, trace)
                    if fy >= f:
                        cand, f = y, fy
                # no explicit merge needed: the logger keeps best-so-far

            else:
                # normal local step from best-so-far
                x = problem.state.current_best.x.copy()
                y = x.copy()
                flips = self.rng.rand(n) < p
                if not flips.any():
                    flips[self.rng.randint(n)] = True
                y[flips] = 1 - y[flips]
                _ = eval_and_record(problem, y, trace)

        return np.array(trace, dtype=float)



# uses your existing eval_and_record(problem, x, trace)

class CatSo_pop2_bias_rate:

    def __init__(self, budget: int = 50_000, mutation_rate: float = 0.1,
                 restart_rate: float = 0.002,  mean_budget_restart: float = 500,
                 var_budget_restart: float = 5, bias_rate: float = 0.9, **kwargs):
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.restart_rate = restart_rate
        self.mean_budget_restart = mean_budget_restart
        self.var_budget_restart = var_budget_restart
        self.bias_rate = bias_rate

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:

        trace = []  # <-- collect (evals, best_y) so your plotting works
        n = problem.meta_data.n_variables


        # set 1/n
        self.mutation_rate = 1 / problem.meta_data.n_variables

        # Initialize with random sample
        favorite = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        backup   = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        result_fav    = eval_and_record(problem, favorite, trace)
        result_backup = eval_and_record(problem, backup, trace)

        if result_backup == problem.state.current_best.y: # backup is better than fav, we swap
            copy = favorite
            favorite = backup
            backup = copy
            result_copy = result_fav
            result_fav = result_backup
            result_backup = result_copy

        # init candidate
        candidate = favorite

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:

            # Potential restart
            if np.random.random() < self.restart_rate:

                budget_restart = np.floor(
                    np.random.normal(loc=self.mean_budget_restart, scale=self.var_budget_restart)
                )

                candidate_restart = np.random.randint(0, 2, size=problem.meta_data.n_variables)
                result_restart = eval_and_record(problem, candidate_restart, trace)

                while budget_restart > 0 and problem.state.evaluations < self.budget and not problem.state.optimum_found:
                    budget_restart -= 1

                    new_candidate_restart = candidate_restart.copy()
                    for i in range(n):  
                        if np.random.random() < self.mutation_rate:
                            new_candidate_restart[i] = 1 - new_candidate_restart[i]

                    new_result_restart = eval_and_record(problem, new_candidate_restart, trace)

                    if new_result_restart >= result_restart:
                        candidate_restart = new_candidate_restart
                        result_restart = new_result_restart

                # insert restart result if good
                if result_restart >= result_backup:
                    if result_restart >= result_fav:
                        backup, result_backup = favorite, result_fav
                        favorite, result_fav = candidate_restart, result_restart
                    else:
                        backup, result_backup = candidate_restart, result_restart

            # No restart
            else:
                if np.random.random() < self.bias_rate:
                    candidate = favorite.copy()
                    for i in range(n):
                        if np.random.random() < self.mutation_rate:
                            candidate[i] = 1 - candidate[i]
                    new_result = eval_and_record(problem, candidate, trace)
                    if new_result >= result_fav:
                        favorite, result_fav = candidate, new_result
                else:
                    candidate = backup.copy()
                    for i in range(n):
                        if np.random.random() < self.mutation_rate:
                            candidate[i] = 1 - candidate[i]
                    new_result = eval_and_record(problem, candidate, trace)
                    if new_result >= result_backup:
                        if new_result >= result_fav:
                            backup, favorite = favorite, candidate
                            result_backup, result_fav = result_fav, new_result
                        else:
                            backup, result_backup = candidate, new_result

        return np.array(trace, dtype=float)  # <-- important for plotting



'''class CatherinotSotiriou:
    """
    Two-candidate biased (1+1) EA with stochastic restarts.

    - Initialize s1, s2 uniformly at random; keep s1 the best (favorite), s2 second-best.
    - Main loop:
        * With probability restart_rate: perform a restart run for a sampled sub-budget,
          then insert the restart best into (s1,s2) if it beats them.
        * Else: with probability bias_rate mutate s1, otherwise mutate s2;
          accept if not worse; keep (s1,s2) ordered by fitness.
    - Mutation: bit-wise with probability mut_rate, forcing at least one flip.
    - Returns a numpy array trace of (evaluations, best_so_far) pairs.
    """

    def __init__(self,
                 budget: int = 50_000,
                 mut_rate: float | None = None,   # default set to 1/n inside
                 bias_rate: float = 0.8,          # prob to explore s1 vs s2
                 restart_rate: float = 0.02,      # prob to trigger a restart step
                 restart_mean: int = 200,         # mean sub-budget for restart runs
                 restart_std: int = 80,           # std for restart sub-budget
                 force_one_flip: bool = True,
                 rng_seed: int | None = None):
        self.budget = budget
        self.mut_rate = mut_rate
        self.bias_rate = bias_rate
        self.restart_rate = restart_rate
        self.restart_mean = restart_mean
        self.restart_std = restart_std
        self.force_one_flip = force_one_flip
        self.rng = np.random.RandomState(rng_seed)

    # --- mutation helper ---
    def _mutate(self, x, p):
        n = len(x)
        y = x.copy()
        flips = self.rng.rand(n) < p
        if self.force_one_flip and not flips.any():
            flips[self.rng.randint(n)] = True
        y[flips] = 1 - y[flips]
        return y

    # --- one local-improvement run used during restart ---
    def _restart_run(self, problem, p, trace):
        """Run a small local search from a fresh random point for a sampled sub-budget."""
        n = problem.meta_data.n_variables
        # sample sub-budget (at least 1)
        subB = int(max(1, self.rng.normal(self.restart_mean, max(1, self.restart_std))))
        # start from a random candidate
        cand = self.rng.randint(0, 2, size=n)
        f = eval_and_record(problem, cand, trace)
        # hill-climb for subB evaluations (or until optimum)
        while subB > 0 and (not problem.state.optimum_found):
            subB -= 1
            y = self._mutate(cand, p)
            fy = eval_and_record(problem, y, trace)
            if fy >= f:
                cand, f = y, fy
        return cand, f

    def __call__(self, problem):
        trace = []

        n = problem.meta_data.n_variables
        p = self.mut_rate if self.mut_rate is not None else (1 / max(n, 1))

        # sample s1, s2 uniformly at random
        s1 = self.rng.randint(0, 2, size=n)
        f1 = eval_and_record(problem, s1, trace)

        s2 = self.rng.randint(0, 2, size=n)
        f2 = eval_and_record(problem, s2, trace)

        # keep s1 as best, s2 as second-best
        if f2 > f1:
            s1, s2 = s2, s1
            f1, f2 = f2, f1

        # main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:

            # decide: restart or normal step
            if self.rng.rand() < self.restart_rate:
                # --- restart branch 
                r_cand, r_fit = self._restart_run(problem, p, trace)
                # insert restart result into (s1, s2) if good enough
                if r_fit >= f1:
                    s2, f2 = s1, f1
                    s1, f1 = r_cand, r_fit
                elif r_fit >= f2:
                    s2, f2 = r_cand, r_fit
                # continue to next iteration
                continue

            # --- no restart: choose which incumbent to explore ---
            if self.rng.rand() < self.bias_rate:
                # explore the favorite s1
                cand = self._mutate(s1, p)
                f_new = eval_and_record(problem, cand, trace)
                if f_new >= f1:
                    s1, f1 = cand, f_new
            else:
                # explore the second-best s2
                cand = self._mutate(s2, p)
                f_new = eval_and_record(problem, cand, trace)
                if f_new >= f2:
                    s2, f2 = cand, f_new

            # maintain ordering: s1 must be the best
            if f2 > f1:
                s1, s2 = s2, s1
                f1, f2 = f2, f1

        return np.array(trace, dtype=float)
'''

        
'''
benchmark_submodular_problem_class(CatherinotSotiriou)

if __name__ == "__main__":
    # PBO sanity — OneMax
    prob1 = ioh.get_problem(1, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO)
    traces = run_reps(prob1, OnePlusOneEA, n_reps=3, budget=3000, mutation_rate=1/100)
    plot_mean_traces(traces, "OneMax — (1+1)EA baseline")

    # GRAPH — MaxCut sanities with your algorithm
    maxcut_fids = [k for k,v in ioh.ProblemClass.GRAPH.problems.items() if "MaxCut" in v]
    fid = maxcut_fids[0]
    prob2 = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
    traces2 = run_reps(prob2, CatherinotSotiriou, n_reps=3, budget=8000, bias_rate=0.8, restart_rate=0.02, restart_mean=200, restart_std=80)
    plot_mean_traces(traces2, f"MaxCut fid={fid} — CatherinotSotiriou")
'''



# =========================
# Helper: generic bit-flip
# =========================
def bitflip(self_rng, x, p, force_one=True):
    y = x.copy()
    flips = self_rng.rand(len(x)) < p
    if force_one and not flips.any():
        flips[self_rng.randint(len(x))] = True
    y[flips] = 1 - y[flips]
    return y

# =========================
# More algorithms
# =========================

class RandomSearch:
    def __init__(self, budget=3000, rng_seed=None):
        self.budget = budget
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        for _ in range(self.budget):
            x = self.rng.randint(0,2,size=problem.meta_data.n_variables)
            eval_and_record(problem, x, trace)
            if problem.state.optimum_found:
                break
        return np.array(trace, dtype=float)

class RLS:  # 1-bit randomized local search
    def __init__(self, budget=3000, rng_seed=None):
        self.budget = budget
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        n = problem.meta_data.n_variables
        x = self.rng.randint(0,2,size=n)
        fx = eval_and_record(problem, x, trace)
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            y = x.copy()
            i = self.rng.randint(n)
            y[i] = 1 - y[i]
            fy = eval_and_record(problem, y, trace)
            if fy >= fx:
                x, fx = y, fy
        return np.array(trace, dtype=float)

class OnePlusOneEA_Simple:
    def __init__(self, budget=3000, mutation_rate=None, rng_seed=None):
        self.budget = budget; self.mr = mutation_rate
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        n = problem.meta_data.n_variables
        p = self.mr if self.mr is not None else (1/max(n,1))
        x = self.rng.randint(0,2,size=n)
        fx = eval_and_record(problem, x, trace)
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            y = bitflip(self.rng, x, p)
            fy = eval_and_record(problem, y, trace)
            if fy >= fx: x, fx = y, fy
        return np.array(trace, dtype=float)

class OnePlusLambdaEA:
    """(1+λ) EA: from the parent, sample λ offspring; accept the best if ≥."""
    def __init__(self, budget=3000, mutation_rate=None, lam=4, rng_seed=None):
        self.budget = budget; self.mr = mutation_rate; self.lam = int(max(1,lam))
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        n = problem.meta_data.n_variables
        p = self.mr if self.mr is not None else (1/max(n,1))
        x = self.rng.randint(0,2,size=n)
        fx = eval_and_record(problem, x, trace)
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            best_y = None; best_f = -np.inf
            for _ in range(self.lam):
                y = bitflip(self.rng, x, p)
                fy = eval_and_record(problem, y, trace)
                if fy > best_f: best_f, best_y = fy, y
                if problem.state.optimum_found: break
            if best_f >= fx:
                x, fx = best_y, best_f
        return np.array(trace, dtype=float)

class MuPlusOneEA:
    """(μ+1) EA: keep μ individuals, mutate the current best; replace worst if child ≥ worst."""
    def __init__(self, budget=3000, mutation_rate=None, mu=5, rng_seed=None):
        self.budget = budget; self.mr = mutation_rate; self.mu = int(max(1,mu))
        self.rng = np.random.RandomState(rng_seed)
    def __call__(self, problem):
        trace = []
        n = problem.meta_data.n_variables
        p = self.mr if self.mr is not None else (1/max(n,1))
        pop = [self.rng.randint(0,2,size=n) for _ in range(self.mu)]
        fits = [eval_and_record(problem, ind, trace) for ind in pop]
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            best_idx = int(np.argmax(fits))
            parent = pop[best_idx]
            child = bitflip(self.rng, parent, p)
            fchild = eval_and_record(problem, child, trace)
            worst_idx = int(np.argmin(fits))
            if fchild >= fits[worst_idx]:
                pop[worst_idx] = child; fits[worst_idx] = fchild
        return np.array(trace, dtype=float)



# ============================================
# Problem suite and benchmarking infrastructure
# ============================================
def get_pbo_problem(fid, dim=100, instance=1):
    return ioh.get_problem(fid, instance=instance, dimension=dim, problem_class=ioh.ProblemClass.PBO)

def get_graph_maxcut():
    maxcut_fids = [k for k,v in ioh.ProblemClass.GRAPH.problems.items() if "MaxCut" in v]
    if not maxcut_fids:
        return None, None
    fid = maxcut_fids[0]
    return ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH), fid

def run_n_reps(problem, algo_cls, n_reps=5, **kwargs):
    traces = []
    successes = 0
    t2s = []
    finals = []
    for _ in range(n_reps):
        problem.reset()
        algo = algo_cls(**kwargs)
        tr = algo(problem)
        traces.append(tr)
        finals.append(problem.state.current_best.y)
        if problem.state.optimum_found:
            successes += 1
            t2s.append(problem.state.evaluations)
    return dict(traces=traces, successes=successes, times=t2s, finals=finals)

def plot_compare_algorithms(problem_name, algo_results, title_suffix=""):
    """algo_results: dict[name] -> list of traces"""
    plt.figure(figsize=(10,6))
    max_x = 0
    mean_curves = {}
    for name, traces in algo_results.items():
        if not traces: continue
        xmax = int(max(t[-1,0] for t in traces if t.size))
        max_x = max(max_x, xmax)
    grid = np.linspace(1, max_x, 200).astype(int)
    for name, traces in algo_results.items():
        interps = []
        for t in traces:
            xs, ys = t[:,0], t[:,1]
            interps.append(np.interp(grid, xs, ys, left=ys[0], right=ys[-1]))
        mean = np.mean(np.vstack(interps), axis=0)
        mean_curves[name] = mean
        plt.plot(grid, mean, lw=2, label=name)
    plt.xlabel("Evaluations"); plt.ylabel("Best-so-far")
    plt.title(f"{problem_name} — mean curves {title_suffix}")
    plt.grid(True); plt.legend(); plt.show()
    return mean_curves


def compare_algorithms_on_maxcoverage(fid=2100, budget=10_000, n_reps=3):
    """
    Compare CatSo_var_budget_restart (var=20) with classic algorithms
    on one fixed MaxCoverage instance. Uses your existing plot_compare_algorithms.
    """
    np.random.seed(42)  # to make runs deterministic

    problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
    pname = problem.meta_data.name

    # fixed parameters (your best combo)
    restart_rate = 10.0 / budget     # 0.001
    mean_restart = int(budget / 10)  # 100
    var_restart  = 10

    algos = {
        "(1+1)EA": lambda: OnePlusOneEA_Simple(budget=budget, mutation_rate=None),
        "RLS": lambda: RLS(budget=budget),
        "RandomSearch": lambda: RandomSearch(budget=budget),
        "Our algo": lambda: CatSo_var_budget_restart(
            budget=budget,
            mutation_rate=None,
            restart_rate=restart_rate,
            mean_budget_restart=mean_restart,
            var_budget_restart=var_restart
        ),
    }

    results = {}
    for name, make_algo in algos.items():
        traces = []
        for _ in range(n_reps):
            problem.reset()
            algo = make_algo()
            tr = algo(problem)
            traces.append(tr)
        results[name] = traces

    plot_compare_algorithms(
        pname, results,
        title_suffix=f"(fid={fid}, budget={budget}, var={var_restart})"
    )


    # run all algorithms
    results = {}
    for name, make_algo in algos.items():
        traces = run_reps(problem, make_algo().__class__, n_reps=n_reps, **make_algo().__dict__)
        results[name] = traces

    # plot with your existing helper
    plot_compare_algorithms(pname, results, title_suffix=f"(fid={fid}, budget={budget})")



def make_rate_grid(budget):
    return [10.0/budget, 1.0/budget, 1.0/(10.0*budget)]

def make_restart_budget_grid(budget):
    return [int(max(1, budget/1000)), int(max(1, budget/100)), int(max(1, budget/10))]

def run_param_grid_for_keyword(keyword, instance=1, budget=10000, n_reps=3):
    """Run CatSo_only_restart on ONE fid matched by `keyword` for all 9 param combos.
       Returns dict {(rate, b_restart): traces}.
    """
    picks = find_fids_by_keywords(keyword, limit_per_kw=1)
    if not picks.get(keyword):
        print(f"[skip] No fid matched '{keyword}'.")
        return None, None

    fid = picks[keyword][0]
    problem = get_graph_problem(fid, instance=instance)

    rates = make_rate_grid(budget)
    brestarts = make_restart_budget_grid(budget)

    results = {}   # (rate, b_restart) -> list of traces
    summary_rows = []  # for CSV

    for r in rates:
        for br in brestarts:
            problem.reset()
            traces = run_reps(
                problem, CatSo_only_restart,
                n_reps=n_reps, budget=budget,
                mutation_rate=None,        # 1/n default
                restart_rate=r,
                budget_restart=br
            )
            results[(r, br)] = traces

            finals = []
            succ = 0
            t2s = []
            for _ in range(n_reps):
                # state after last rep:
                finals.append(problem.state.current_best.y)
                if problem.state.optimum_found:
                    succ += 1
                    t2s.append(problem.state.evaluations)
            mean_final = float(np.mean(finals)) if finals else float("nan")
            mean_t2s = float(np.mean(t2s)) if t2s else float("nan")

            summary_rows.append([keyword, fid, instance, budget, r, br, succ, n_reps, f"{mean_final:.3f}", f"{mean_t2s}"])

    return (keyword, fid, problem.meta_data.name), results, summary_rows


def plot_var_sweep_for_keyword(keyword, budget=10_000, n_reps=3):
    """One figure for `keyword` (MaxCut or MaxCoverage) with three variance curves."""
    fixed_rr = 10.0 / budget              # 0.001 when budget=10000
    fixed_mean = int(budget / 10)         # 100 when budget=10000
    var_list = [fixed_mean/10, fixed_mean/20, fixed_mean/50, fixed_mean/100]  # 3 variances

    # pick one fid for the keyword
    picks = find_fids_by_keywords(keyword, limit_per_kw=1)
    if not picks.get(keyword):
        print(f"[skip] No fid matched '{keyword}'.")
        return
    fid = picks[keyword][0]
    problem = get_graph_problem(fid, instance=1)

    # run & collect mean curves
    all_traces = {}
    for v in var_list:
        traces = run_reps(
            problem, CatSo_var_budget_restart,
            n_reps=n_reps, budget=budget,
            mutation_rate=None,
            restart_rate=fixed_rr,
            mean_budget_restart=fixed_mean,
            var_budget_restart=v
        )
        all_traces[f"var={v:.2f}"] = traces

    # plot: unify grid and draw 3 curves
    plt.figure(figsize=(10,6))
    max_x = int(max(t[-1,0] for traces in all_traces.values() for t in traces if t.size))
    grid = np.linspace(1, max_x, 200).astype(int)

    for label, traces in all_traces.items():
        interps = []
        for t in traces:
            xs, ys = t[:,0], t[:,1]
            interps.append(np.interp(grid, xs, ys, left=ys[0], right=ys[-1]))
        mean_curve = np.mean(np.vstack(interps), axis=0)
        plt.plot(grid, mean_curve, lw=2, label=label)

    title = f"{problem.meta_data.name} (fid={fid}) — restart_rate={fixed_rr}, mean_restart_budget={fixed_mean}"
    plt.xlabel("Evaluations"); plt.ylabel("Best-so-far")
    plt.title(title); plt.grid(True); plt.legend(); plt.show()


def plot_var_sweep_maxcut_and_maxcoverage(budget=10_000, n_reps=3):
    plot_var_sweep_for_keyword("MaxCut", budget=budget, n_reps=n_reps)
    plot_var_sweep_for_keyword("MaxCoverage", budget=budget, n_reps=n_reps)
    




def plot_grid_per_rate(keyword, fid, prob_name, results):
    """Make 3 figures (one per restart_rate), each with 3 curves (budget_restart values)."""
    # collect unique sorted rates and brs
    rates = sorted({k[0] for k in results.keys()})
    for r in rates:
        # unify grid across all three curves at this r
        plt.figure(figsize=(10,6))
        max_x = 0
        traces_by_br = {}
        for (rr, br), tr_list in results.items():
            if rr != r: continue
            traces_by_br[br] = tr_list
            xmax = int(max(t[-1,0] for t in tr_list if t.size))
            max_x = max(max_x, xmax)
        grid = np.linspace(1, max_x, 200).astype(int)
        for br, tr_list in sorted(traces_by_br.items()):
            interps = []
            for t in tr_list:
                xs, ys = t[:,0], t[:,1]
                interps.append(np.interp(grid, xs, ys, left=ys[0], right=ys[-1]))
            mean = np.mean(np.vstack(interps), axis=0)
            plt.plot(grid, mean, lw=2, label=f"budget_restart={br}")
        title = f"{prob_name} (fid={fid}) — restart_rate={r}  [budget={int(max_x)}]"
        plt.xlabel("Evaluations"); plt.ylabel("Best-so-far")
        plt.title(title)
        plt.grid(True); plt.legend(); plt.show()

def run_all_keywords_grid(budget=10000, n_reps=3, instance=1, csv_path="grid_summary.csv"):
    keywords = ["MaxCut", "MaxCoverage", "MaxInfluence", "PackWhile"]
    all_rows = []
    for kw in keywords:
        info = run_param_grid_for_keyword(kw, instance=instance, budget=budget, n_reps=n_reps)
        if info is None:
            continue
        (keyword, fid, name), results, rows = info
        print(f"\n=== {keyword}: fid={fid} name={name} ===")
        plot_grid_per_rate(keyword, fid, name, results)
        all_rows.extend(rows)
    if all_rows:
        save_summary_csv(csv_path, all_rows,
                         header=("keyword","fid","instance","budget","restart_rate","budget_restart","successes","reps","mean_final","mean_t2s"))
        print(f"\nSaved {csv_path}")


def run_all_graph_instances_with_plots(budget=10_000, n_reps=3,
                                       filter_substrings=None,
                                       max_plots=None):
    """
    For every GRAPH fid (optionally filtered), run CatSo_only_restart
    with the fixed (restart_rate=10/budget, budget_restart=budget/10)
    and plot a single mean curve per fid.
    """
    fixed_rr = 10.0 / budget                  # 0.001 for budget=10000
    fixed_br = int(max(1, budget / 10))       # 100   for budget=10000

    shown = 0
    for fid, name in sorted(ioh.ProblemClass.GRAPH.problems.items()):
        if filter_substrings:
            low = name.lower()
            if not any(s.lower() in low for s in filter_substrings):
                continue

        try:
            problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        except Exception as e:
            print(f"[skip] fid={fid} {name}: {e}")
            continue

        traces = run_reps(
            problem, CatSo_only_restart,
            n_reps=n_reps, budget=budget,
            mutation_rate=None,
            restart_rate=fixed_rr,
            budget_restart=fixed_br
        )

        title = f"{name} (fid={fid}) — restart_rate={fixed_rr}, budget_restart={fixed_br}"
        plot_mean_traces(traces, title)

        shown += 1
        if max_plots is not None and shown >= max_plots:
            print(f"(stopping after {shown} fids due to max_plots)")
            break


'''def _bias_values_from_budget(budget):
    # 1 - (k / budget) for k in {10, 50, 100, 500, 1000}
    divisors = [10, 50, 100, 500, 1000]
    return [1.0 - (k / float(budget)) for k in divisors]'''


def plot_bias_sweep_for_fid(fid, budget=10_000, n_reps=3,
                            restart_rate=None, mean_br=None, var_br=20):
    """
    Sweep bias_rate for CatSo_pop2_bias_rate on a fixed GRAPH fid and plot all curves on one figure.
    Uses your existing plot_compare_algorithms.
    """
    problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
    pname = problem.meta_data.name

    rr  = restart_rate if restart_rate is not None else (10.0 / budget)   # your default best combo
    mbr = mean_br if mean_br is not None else int(budget / 10)            # your default best combo

    bias_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    results = {}
    for b in bias_list:
        label = f"bias={b:.6f}"
        traces = run_reps(
            problem,
            CatSo_pop2_bias_rate,
            n_reps=n_reps,
            budget=budget,
            mutation_rate=None,            # 1/n inside
            restart_rate=rr,
            mean_budget_restart=mbr,
            var_budget_restart=var_br,
            bias_rate=b
        )
        results[label] = traces

    plot_compare_algorithms(
        pname,
        results,
        title_suffix=f"(fid={fid}, budget={budget}, rr={rr}, mean_br={mbr}, var_br={var_br})"
    )


def compare_algorithms_on_graph_instance(fid, budget=10_000, n_reps=3,
                                         restart_rate=None, mean_br=None, var_br=20,
                                         bias_rate=0.9, problem_name_override=None):
    """
    Run (1+1)EA, RLS, RandomSearch, and CatSo_pop2_bias_rate (bias=0.9 by default)
    on a specific GRAPH problem fid, then plot mean curves together.

    - restart_rate defaults to 1/budget
    - mean_br (mean restart budget) defaults to budget/10
    - var_br controls the Normal restart-budget std-dev (keep small, e.g., 20)
    """
    problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
    pname = problem_name_override or problem.meta_data.name

    rr = (1.0 / budget) if restart_rate is None else float(restart_rate)
    mbr = int(budget / 10) if mean_br is None else int(mean_br)

    algo_results = {}

    # baselines
    algo_results["(1+1)EA"] = run_reps(problem, OnePlusOneEA_Simple,
                                       n_reps=n_reps, budget=budget, mutation_rate=None)
    algo_results["RLS"] = run_reps(problem, RLS,
                                   n_reps=n_reps, budget=budget)
    algo_results["RandomSearch"] = run_reps(problem, RandomSearch,
                                            n_reps=n_reps, budget=budget)

    # our bias algo (exact class you use in your code)
    algo_results["Our algo"] = run_reps(
        problem, CatSo_pop2_bias_rate, n_reps=n_reps,
        budget=budget,
        mutation_rate=None,            # uses 1/n internally anyway
        restart_rate=rr,
        mean_budget_restart=mbr,
        var_budget_restart=var_br,
        bias_rate=bias_rate
    )

    title = f"{pname} — mean curves (fid={fid}, budget={budget}, var={var_br})"
    plot_compare_algorithms(pname, algo_results, title_suffix=f"(fid={fid}, budget={budget}, var={var_br})")


def compare_many(fids, **kw):
    for fid in fids:
        try:
            name = ioh.ProblemClass.GRAPH.problems[fid]
        except KeyError:
            name = f"fid={fid}"
        print(f"\n--- {name} (fid={fid}) ---")
        compare_algorithms_on_graph_instance(fid=fid, problem_name_override=name, **kw)



# optional: save a small CSV 
import csv
def save_summary_csv(path, summary_rows, header=("problem","algorithm","successes","reps","mean_final","mean_t2s")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in summary_rows: w.writerow(r)

# ======================
# Example master routine
# ======================
def big_benchmark():
    # --- choose problems ---
    problems = []
    problems.append(("OneMax (fid=1, d=100)", get_pbo_problem(1, dim=100)))
    problems.append(("LeadingOnes (fid=2, d=100)", get_pbo_problem(2, dim=100)))
    problems.append(("LABS (fid=18, d=60)", get_pbo_problem(18, dim=60)))
    problems.append(("IsingRing (fid=19, d=100)", get_pbo_problem(19, dim=100)))
    gprob, gfid = get_graph_maxcut()
    if gprob is not None:
        problems.append((f"MaxCut (GRAPH fid={gfid})", gprob))

    # --- choose algorithms & params ---
    ALGOS = {
        "(1+1)EA 1/n": lambda budget: OnePlusOneEA_Simple(budget=budget, mutation_rate=None),
        "RLS (1-bit)": lambda budget: RLS(budget=budget),
        "(1+λ)EA λ=4": lambda budget: OnePlusLambdaEA(budget=budget, lam=4),
        "(μ+1)EA μ=5": lambda budget: MuPlusOneEA(budget=budget, mu=5),
        "RandomSearch": lambda budget: RandomSearch(budget=budget),
        "OurAlgo":       lambda budget: CatherinotSotiriou(budget=budget, bias_rate=0.8, restart_rate=0.02, restart_mean=200, restart_std=80),
    }

    BUDGET = 10000
    N_REPS = 5
    summary = []

    for pname, prob in problems:
        print(f"\n=== {pname} ===")
        # run all algos
        per_algo_traces = {}
        for aname, factory in ALGOS.items():
            result = run_n_reps(prob, algo_cls=lambda **kw: factory(BUDGET), n_reps=N_REPS)
            per_algo_traces[aname] = result["traces"]
            s = result["successes"]; reps = N_REPS
            mean_final = float(np.mean(result["finals"])) if result["finals"] else float("nan")
            mean_t2s = float(np.mean(result["times"])) if result["times"] else float("nan")
            print(f"{aname:15s}  success {s}/{reps}  mean final={mean_final:.3f}  mean t2s={mean_t2s}")
            summary.append([pname, aname, s, reps, f"{mean_final:.3f}", f"{mean_t2s}"])
        # plot comparison for this problem
        plot_compare_algorithms(pname, per_algo_traces)

    save_summary_csv("summary.csv", summary)
    print("\nSaved summary.csv (no pandas).")

def benchmark_all_graph_items(budget=10_000, n_reps=3,
                              restart_rate=0.01, budget_restart=200,
                              csv_path=None, limit=None):
    """
    Iterate over all GRAPH problems (fid -> name) and run CatSo_only_restart
    with the given restart_rate and budget_restart. Prints a short summary per fid.
    Optionally writes a CSV.
    """
    rows = []
    seen = 0
    for fid, name in ioh.ProblemClass.GRAPH.problems.items():
        if limit is not None and seen >= limit:
            break
        seen += 1

        try:
            problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        except Exception as e:
            print(f"[skip] fid={fid} could not be loaded: {e}")
            continue

        res = run_n_reps(
            problem,
            algo_cls=CatSo_pop2_bias_rate,
            n_reps=n_reps,
            budget=budget,
            mutation_rate=None,          # default 1/n
            restart_rate=restart_rate,   
            budget_restart=int(max(1, budget_restart))  
        )

        succ = res["successes"]
        mean_final = float(np.mean(res["finals"])) if res["finals"] else float("nan")
        mean_t2s   = float(np.mean(res["times"]))  if res["times"]  else float("nan")

        print(f"[fid {fid:4d}] {name:20s}  success {succ}/{n_reps}  "
              f"mean_final={mean_final:.3f}  mean_t2s={mean_t2s}")

        rows.append([name, fid, budget, restart_rate, budget_restart,
                     succ, n_reps, f"{mean_final:.3f}", f"{mean_t2s}"])

    if csv_path and rows:
        save_summary_csv(csv_path, rows,
            header=("problem_name","fid","budget","restart_rate","budget_restart",
                    "successes","reps","mean_final","mean_t2s"))
        print(f"\nSaved {csv_path}")


'''
def benchmark_all_graph_items_grid(budget=10_000, n_reps=3, csv_path="graph_all_grid.csv", limit=None):
    rates = [10.0/budget, 1.0/budget, 1.0/(10.0*budget)]
    brests = [int(max(1, budget/1000)), int(max(1, budget/100)), int(max(1, budget/10))]

    rows = []
    seen = 0
    for fid, name in ioh.ProblemClass.GRAPH.problems.items():
        if limit is not None and seen >= limit:
            break
        seen += 1
        try:
            problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        except Exception as e:
            print(f"[skip] fid={fid} could not be loaded: {e}")
            continue

        print(f"\n=== {name} (fid={fid}) — budget={budget}, reps={n_reps} ===")
        for r in rates:
            for br in brests:
                res = run_n_reps(
                    problem,
                    algo_cls=CatSo_only_restart,
                    n_reps=n_reps,
                    budget=budget,
                    mutation_rate=None,    # 1/n
                    restart_rate=r,        # <-- restart rate set here per combo
                    budget_restart=br      # <-- restart budget set here per combo
                )
                succ = res["successes"]
                mean_final = float(np.mean(res["finals"])) if res["finals"] else float("nan")
                mean_t2s   = float(np.mean(res["times"]))  if res["times"]  else float("nan")

                print(f"  r={r:.5f}  br={br:4d}  success {succ}/{n_reps}  "
                      f"mean_final={mean_final:.3f}  mean_t2s={mean_t2s}")

                rows.append([name, fid, budget, r, br, succ, n_reps, f"{mean_final:.3f}", f"{mean_t2s}"])

                problem.reset()

    if csv_path and rows:
        save_summary_csv(csv_path, rows,
            header=("problem_name","fid","budget","restart_rate","budget_restart",
                    "successes","reps","mean_final","mean_t2s"))
        print(f"\nSaved {csv_path}")
'''

# SANITY CHECK FOR GRAPHS

def quick_graph_sanity():
    picks = find_fids_by_keywords("MaxCut", "MaxCoverage", "MaxInfluence", "PackWhile", limit_per_kw=1)

    BUDGET = 6000
    N_REPS  = 3

    # choose a demo setting for our restart-only algo
    restart_rate = 0.02
    budget_restart = 200   # evaluations spent inside each restart run

    for kw, fid_list in picks.items():
        if not fid_list:
            print(f"[skip] No fid matched '{kw}' in this IOH build.")
            continue

        fid = fid_list[0]
        problem = get_graph_problem(fid, instance=1)
        print(f"\nRunning {kw}: fid={fid}, name={problem.meta_data.name}")

        # baseline (1+1)EA 1/n
        tr_base = run_reps(problem, OnePlusOneEA_Simple,
                           n_reps=N_REPS, budget=BUDGET, mutation_rate=None)  # 1/n default
        plot_mean_traces(tr_base, f"{kw} (fid={fid}) — (1+1)EA 1/n")

        # our restart-only algorithm
        problem.reset()
        tr_ours = run_reps(
            problem, CatSo_only_restart,
            n_reps=N_REPS, budget=BUDGET,
            mutation_rate=None,          # 1/n default inside runner
            restart_rate=restart_rate,
            budget_restart=budget_restart
        )
        plot_mean_traces(
            tr_ours,
            f"{kw} (fid={fid}) — Our algo\nrestart_rate={restart_rate}, budget_restart={budget_restart}"
        )

'''
if __name__ == "__main__":
    list_graph_problems()       # first time: see names
    quick_graph_sanity()        # then: quick tests + plots
'''

# =====================
# Parameter sweep demo
# =====================
def lambda_sweep_on_onemax():
    prob = get_pbo_problem(1, dim=100)
    BUDGET = 8000
    N_REPS = 3
    lam_values = [1, 2, 4, 8, 16]

    per_lam = {}
    for lam in lam_values:
        result = run_n_reps(
            prob,
            algo_cls=lambda **kw: OnePlusLambdaEA(budget=BUDGET, lam=lam),
            n_reps=N_REPS
        )
        per_lam[f"(1+λ)EA λ={lam}"] = result["traces"]
        mean_final = float(np.mean(result["finals"])) if result["finals"] else float("nan")
        print(f"λ={lam}: mean final={mean_final:.3f}, success {result['successes']}/{N_REPS}")
    plot_compare_algorithms("OneMax (d=100)", per_lam, title_suffix="— λ sweep")

# -------------
# Run examples
# -------------

if __name__ == "__main__":
    # 1) sanity check names once (comment out later)
    # list_graph_problems()

    # 2) quick sanity on a couple problems (small plots)
    # quick_graph_sanity()

    # 3) full 3×3 grid per problem type (plots + CSV)
    #run_all_keywords_grid(budget=100, n_reps=3, instance=1, csv_path="grid_summary.csv")

    # 4) benchmark all problem instances for only_restart algorithm
    BUDGET = 500
    RR     = 10.0 / BUDGET       # ← restart_rate (pick one of the three)
    BR     = int(BUDGET / 100)   # ← budget_restart (pick one of the three)

    '''benchmark_all_graph_items(
        budget=BUDGET,
        n_reps=2,
        restart_rate=RR,
        budget_restart=BR,
        csv_path="graph_all_single.csv",  # or None to skip csv
        limit=None                        # e.g. 5 while testing
    )'''

    # 5) plot all problem instances 
    '''run_all_graph_instances_with_plots(
        budget=10_000, n_reps=3,
        filter_substrings=["MaxCut","MaxCoverage","MaxInfluence","PackWhile"],
        max_plots=None
    )'''

    # 6) Variance sweeps (normal-distributed restart budget) for MaxCut & MaxCoverage
    #plot_var_sweep_maxcut_and_maxcoverage(budget=10_000, n_reps=3)
    #compare_algorithms_on_maxcoverage(fid=2204, budget=10_000, n_reps=3)

    # 7) Benchmark bias algo
    '''benchmark_all_graph_items(
        budget=BUDGET,
        n_reps=2,
        restart_rate=RR,
        budget_restart=BR,
        csv_path="benchmark_bias_09.csv",  # or None to skip csv
        limit=None                        # e.g. 5 while testing
    )'''

    # 8) Plot bias algo
    #plot_bias_sweep_for_fid(fid=2308, budget=5_000, n_reps=3, var_br=20)

    # 9) Compare to other algos
    #compare_algorithms_on_graph_instance(fid=2000, budget=10_000, n_reps=3, restart_rate=0.001, mean_br=1000, var_br=20, bias_rate=0.9)
    compare_many([2000, 2105, 2204, 2304],
             budget=10_000, n_reps=3,
             restart_rate=0.001, mean_br=1000, var_br=20, bias_rate=0.9)
