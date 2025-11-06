import iohinspector
import ioh
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os


id = 1

class BinaryEvolutionaryAlgorithm:
    """
    Simple (1+1) Evolutionary Algorithm for binary optimization with bit-flip mutation.
    """
    def __init__(self, budget: int = 50_000, mutation_rate: int = 0.1, plot: bool= False):
        self.budget = budget
        self.mutation_rate = mutation_rate
        self.plot = plot

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
        # init the plotting
        if self.plot :
            """
            fig, ax = plt.subplots(figsize=(16, 9))
            plot_x = [i for i in range(0,budget)]
            plot_y = []
            """
            global id
            id+=1
            logger = ioh.logger.Analyzer(
                    root=os.getcwd(),                  # Store data in the current working directory
                    folder_name=(f"BinEA_{problem.meta_data.name}_{self.mutation_rate}_{id}"),
                    algorithm_name="BinEA",
            )

            logger.add_experiment_attribute("mutation_rate", f"{self.mutation_rate}")
            
            problem.attach_logger(logger)


        # Initialize with a random binary solution
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)

        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            # Mutate: flip bits with given probability
            candidate = problem.state.current_best.x.copy()
            
            # Bit-flip mutation: flip each bit with probability mutation_rate
            for i in range(len(candidate)):
                if np.random.random() < self.mutation_rate:
                    candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)
            
            problem(candidate)


        # Plot :
        if self.plot :
            logger.close()
                
            manager = iohinspector.DataManager()
            manager.overview
            manager.add_folder(f"BinEA_{problem.meta_data.name}_{self.mutation_rate}_{id}")
            df = manager.load(True, True)
            
            # Plot results
            fig, ax = plt.subplots(figsize=(16, 9))
            dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
            ax.set_xlim(1, 5000)
            ax.set_title(f"Test of BinEA on {problem.meta_data.name} (mut {self.mutation_rate})")
        
            plt.show()



def benchmark_submodular_problem_class(algorithm, n_reps: int = 5, budget: int = 50000, problems="MaxCut", root_dir: str = ".", **kwargs):
    fids = [k if problems in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
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
            else : count_fail += 1


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


def benchmark_problem_class(algorithm, n_reps: int = 5, budget: int = 50000, problems="MaxCut", root_dir: str = ".", nb_plot: int = 3, **kwargs):
    fids = [k if problems in v else None for k, v in ioh.ProblemClass.GRAPH.problems.items()]
    fids = [fid for fid in fids if fid is not None]
    logger = ioh.logger.Analyzer(
        root=root_dir,                  
        folder_name=f"{problems}_{algorithm.__name__}",         
        algorithm_name=algorithm.__name__, 
    )

    for k, v in kwargs.items():
        logger.add_experiment_attribute(k, str(v))
    for fid in fids:
        problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.GRAPH)
        problem.attach_logger(logger)

        for _ in range(n_reps):
            algorithm(budget=budget, plot=(nb_plot > 0), **kwargs)(problem)
            problem.reset()
            nb_plot -= 1





def benchmark_compare_BinaryEA_PBO_pb(algorithm, n_reps: int = 5, budget: int = 50000, problems_info=[(1, "OneMax"), (2, "LeadingOnes"), (18, "LABS"), (19, "IsingRing")], **kwargs):
    for fid, name in problems_info:
        problem = ioh.get_problem(fid, instance=1, dimension=10, problem_class=ioh.ProblemClass.INTEGER)


class CatherinotSotiriou:
    
    # idea : alternate phases of exploration vs exploitation
    # exploration : random restarts, random un-restarts (the idea is that if the restart is not fruitful after a certain time, we go back to exploiting the previous best)
    # exploitation : typical bit-wise mutation operator

    def __init__(self, budget: int = 50_000, mutation_rate: int = 0.1, plot: bool= False, **kwargs): # mutation rate, random restart rate, random un-restart rate
        self.budget = budget
        self.mutation_rate = mutation_rate

    # whether we want to plot or not :
        self.plot = plot

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
        """
        Your algorithm implementation goes here.
        
        Args:
            problem: The binary optimization problem to solve
        """
        # Init the plotting
        if self.plot :
            """
            fig, ax = plt.subplots(figsize=(16, 9))
            plot_x = [i for i in range(0,budget)]
            plot_y = []
            """
            global id
            id+=1
            logger = ioh.logger.Analyzer(
                    root=os.getcwd(),                  # Store data in the current working directory
                    folder_name=(f"CatSot_{problem.meta_data.name}_{self.mutation_rate}_{id}"),
                    algorithm_name="CatSot",
            )

            logger.add_experiment_attribute("mutation_rate", f"{self.mutation_rate}")
            
            problem.attach_logger(logger)




        # Initialize with random sample
        x = np.random.randint(0, 2, size=problem.meta_data.n_variables)
        problem(x)

        # Main loop
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            # Mutation operator : each bit is flipped with proba "mutation_rate"
            candidate = problem.state.current_best.x.copy()

            for i in range(len(candidate)):
                if np.random.random() < self.mutation_rate:
                    candidate[i] = 1 - candidate[i]  # Flip bit (0->1, 1->0)

            problem(candidate)
        


        # Plot :
        if self.plot :
            logger.close()
                
            manager = iohinspector.DataManager()
            manager.overview
            manager.add_folder(f"CatSot_{problem.meta_data.name}_{self.mutation_rate}_{id}")
            df = manager.load(True, True)
            
            # Plot results
            fig, ax = plt.subplots(figsize=(16, 9))
            dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
            ax.set_xlim(1, 5000)
            ax.set_title(f"Test of CatSot on {problem.meta_data.name} (mut {self.mutation_rate})")
        
            plt.show()


        # print(f"algo was called, init sampled {x}, finished w {problem.state.current_best.x}")


"""
OneMax = ioh.get_problem(1, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO)
pb = ioh.get_problem(1, instance=3, dimension=20, problem_class=ioh.ProblemClass.PBO)
CatherinotSotiriou(plot=True)(pb)
pb.reset()
BinaryEvolutionaryAlgorithm(plot=True)(pb)
"""

benchmark_problem_class(CatherinotSotiriou)



#benchmark_submodular_problem_class(CatherinotSotiriou)