#import iohinspector
import ioh
import numpy as np
#import pandas as pd
#import polars as pl
#from tqdm import tqdm 
import matplotlib.pyplot as plt
#import os



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




class CatherinotSotiriou:
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
                # --- restart branch (second page of your notes) ---
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


