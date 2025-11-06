#import iohinspector
import ioh
import numpy as np
#import pandas as pd
import polars as pl
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os


# OneMax : Problem 1
OneMax = ioh.get_problem(1, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO) 
# LeadingOnes : Problem 2
LeadingOnes = ioh.get_problem(2, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO) 
# LABS : Problem 18
LABS = ioh.get_problem(18, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO) 
# IsingRing : Problem 19
IsingRing = ioh.get_problem(19, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO) 



def test1() :
    # Create some example problems
    problems = [
        ioh.get_problem(1, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO),  # OneMax
        ioh.get_problem(2, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO),  # LeadingOnes  
        ioh.get_problem(18, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO), # LABS
    ]

    print("Example evaluations for different PBO problems (dimension=5):")
    print("=" * 60)

    # Generate some example binary vectors
    test_vectors = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1], 
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ]

    for i, problem in enumerate(problems):
        print(f"\n{problem.meta_data.name} (ID {problem.meta_data.problem_id}):")
        print("-" * 40)
        for vec in test_vectors:
            result = problem(vec)
            print(f"  f({vec}) = {result:.3f}")
        problem.reset()  # Reset for next use



def test2() : # meta-data, rd vectors, best sol found so far
    # problem = ioh.get_problem(1, instance=1, dimension=10, problem_class=ioh.ProblemClass.PBO)
    print("\n\n\nMeta-data of one max, n=10 :")
    print(f"Number of variables: {OneMax.meta_data.n_variables}")
    print(f"Lower bounds: {OneMax.bounds.lb}")
    print(f"Upper bounds: {OneMax.bounds.ub}")


    # Generate a random binary vector within the problem bounds
    print("\n\n\nRd vector OneMax test")
    x0 = np.random.randint(OneMax.bounds.lb[0], OneMax.bounds.ub[0] + 1, size=OneMax.meta_data.n_variables)
    x1 = np.random.randint(OneMax.bounds.lb[0], OneMax.bounds.ub[0] + 1, size=OneMax.meta_data.n_variables)

    # Evaluation happens like a 'normal' objective function would
    print(f"Random binary vector: {x0}")
    result0 = OneMax(x0)
    print(f"Function value x0: {result0}")
    print(f"Random binary vector: {x1}")
    result1 = OneMax(x1)
    print(f"Function value x1: {result1}")

    print("\ninternal state OneMax : ")
    print(OneMax.state)

    # bestvect = OneMax.state
    # not directly a vect
    # bestresult = OneMax(bestvect)
    # print(f"Best: {bestresult}")







    






class RandomSearch:
    # 'Simple random search algorithm for binary optimization'
    def __init__(self, budget: int):
        self.budget: int = budget
        
    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
        # 'Evaluate the problem n times with a randomly generated binary solution'
        for _ in range(self.budget):
            # Generate random binary vector (0s and 1s)
            x = np.random.randint(0, 2, size=problem.meta_data.n_variables)            
            problem(x)


# If we want to perform multiple runs with the same objective function, after every run, the problem has to be reset, 
# such that the internal state reflects the current run.
def run_experiment(problem, algorithm, n_runs=5, verbose=True):
    for run in tqdm(range(n_runs)):
        
        # Run the algorithm on the problem
        algorithm(problem)

        # print the best found for this run
        if verbose:
            print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}")

        # Reset the problem
        problem.reset()


def run_RDS() :
    run_experiment(OneMax, RandomSearch(100))


def test_logger() :
    logger = ioh.logger.Analyzer(
            root=os.getcwd(),                  # Store data in the current working directory
            folder_name="PBO_RS_Test",         # in a folder named: 'PBO_RS_Test'
            algorithm_name="RandomSearch_PBO", # meta-data for the algorithm used to generate these results
    )

    # this automatically creates a folder 'PBO_RS_Test' in the current working directory
    # if the folder already exists, it will given an additional number to make the name unique
    logger


    for fid in [1, 2, 18, 19]:  # OneMax, LeadingOnes, LABS, IsingRing
        problem = ioh.get_problem(fid, instance=1, dimension=20, problem_class=ioh.ProblemClass.PBO)
        problem.attach_logger(logger)
        run_experiment(problem, RandomSearch(5000), n_runs=15, verbose=False)
    # After the experiment is done, we can close the logger
    logger.close()

    manager = iohinspector.DataManager()
    manager.add_folder("PBO_RS_Test-2")
    manager.overview
    df = manager.load(True, True)

    #########################################################################################################################################################################################################################################################################################################################################################################  PLOT
    fig, ax = plt.subplots(figsize=(16, 9))
    # Plot results for OneMax (function_id = 1)
    dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id") == 1), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
    ax.set_xlim(1, 5000)
    ax.set_title("Fixed Budget Performance on OneMax Problem")
    plt.show()

#test_logger()


def test5() :
    problems_info = [
        (1, "OneMax", "Maximize number of 1s"),
        (2, "LeadingOnes", "Maximize leading 1s"), 
        (18, "LABS", "Low autocorrelation"),
        (19, "IsingRing", "Ising model on ring")
    ]

    print("PBO Problem Characteristics (dimension=10):")
    print("=" * 60)

    for fid, name, description in problems_info:
        problem = ioh.get_problem(fid, instance=1, dimension=10, problem_class=ioh.ProblemClass.INTEGER)
        
        print(f"\n{name} (ID {fid}): {description}")
        print("-" * 40)
        
        # Test some specific patterns
        test_cases = [
            ("All zeros", [0] * 10),
            ("All ones", [1] * 10),
            ("Alternating", [i % 2 for i in range(10)]),
            ("First half", [1] * 5 + [0] * 5)
        ]
        
        for case_name, vector in test_cases:
            result = problem(vector)
            print(f"  {case_name:12} {vector} -> {result:.3f}")
        
        problem.reset()

    print("\nThese different problem characteristics will help us understand")
    print("how different algorithm strategies perform on various landscapes.")







class BinaryEvolutionaryAlgorithm:
    """
    Simple (1+1) Evolutionary Algorithm for binary optimization with bit-flip mutation.
    """
    def __init__(self, budget: int, mutation_rate: float):
        self.budget = budget
        self.mutation_rate = mutation_rate

    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> None:
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
            print(f"problem.state.current_best.x = {problem.state.current_best.x}\n")
            print(f"problem.state.current_best.y = {problem.state.current_best.y}\n")



def test_LO():
    problem = ioh.get_problem(2, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO)
    #BinaryEvolutionaryAlgorithm(20,0.1)(problem)
    print("\nProblem is LeadingOnes")
    print("-" * 40)
    result = problem([1,0,0,0,0])
    print(f"Test on [1,0,0,0,0] : result = {result}")
    result = problem([1,0,1,0,0])
    print(f"Test on [1,0,1,0,0] : result = {result}")
    print(f"problem.state.current_best.x = {problem.state.current_best.x}\n")

    problem.reset()


test_LO()

def test3() : #diff instances of OneMax
    problems = [
        ioh.get_problem(1, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO),  # OneMax
        ioh.get_problem(1, instance=2, dimension=5, problem_class=ioh.ProblemClass.PBO),  # OneMax with different instance  
        ioh.get_problem(1, instance=3, dimension=5, problem_class=ioh.ProblemClass.PBO), # OneMax with different instance
    ]

    print("Example evaluations for different PBO problems (dimension=5):")
    print("=" * 60)

    # Generate some example binary vectors
    """
    test_vectors = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1], 
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1]
    ]
    """

    fig, ax = plt.subplots(figsize=(16, 9))

    #legend_set = False


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
        
        """
        for vec in test_vectors:
            result = problem(vec)
            print(f"  f({vec}) = {result:.3f}")
        """ 
        
        
        # Plot results for OneMax (function_id = 1)
        ax.set_title(f"Test on instance {problem.meta_data.instance} of {problem.meta_data.name} (ID {problem.meta_data.problem_id})")
        ax.set_xlim(1, 5000)

        """
        if legend_set :
            dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
        else :
            legend_set = True
            dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id").eq(problem.meta_data.problem_id)), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
        """
        problem.reset()  # Reset for next use
        
    #ax.get_legend().remove()
    plt.show()

#test3()

#print(ioh.ProblemClass.GRAPH.problems)


def test4():
    problems = [
        ioh.get_problem(1, instance=1, dimension=5, problem_class=ioh.ProblemClass.PBO),  # OneMax
        ioh.get_problem(1, instance=2, dimension=5, problem_class=ioh.ProblemClass.PBO),  # OneMax with different instance  
        ioh.get_problem(1, instance=3, dimension=5, problem_class=ioh.ProblemClass.PBO), # OneMax with different instance
    ]

    #print("Example evaluations for different PBO problems (dimension=5):")
    print("=" * 60)
    fig, ax = plt.subplots(figsize=(16, 9))

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
        ax.set_title(f"Test on three instances of {problem.meta_data.name} (ID {problem.meta_data.problem_id})")

        problem.reset()  # Reset for next use

    plt.show() # show them all at once


# test4()

def test_BinEA() :
    for idx, mutation_rate in enumerate([0.01, 0.1, 0.5]):  # Different bit-flip probabilities
        logger = ioh.logger.Analyzer(
            root=os.getcwd(), 
            folder_name=f"PBO_EA{idx}",     
            algorithm_name=f"BinaryEA-{mutation_rate}",    # Include mutation rate in algorithm name
        )
        logger.add_experiment_attribute("mutation_rate", f"{mutation_rate}") # Add as experiment attribute
        
        for fid in [1, 2, 18, 19]:  # OneMax, LeadingOnes, LABS, IsingRing
            problem = ioh.get_problem(fid, instance=1, dimension=20, problem_class=ioh.ProblemClass.INTEGER)
            problem.attach_logger(logger)
            run_experiment(problem, BinaryEvolutionaryAlgorithm(2000, mutation_rate), n_runs=15, verbose=False)
        logger.close()

    # We can reuse the DataManager from before if we want to have random search in the comparison
    manager = iohinspector.DataManager()
    manager.overview
    for idx in range(3):
        manager.add_folder(f"PBO_EA{idx}")
    df = manager.load(True, True)


    # df.filter(pl.col("function_id") == 1)


    fig, ax = plt.subplots(figsize=(16, 9))
    # Plot results for OneMax (function_id = 1)
    dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id") == 1), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
    ax.set_xlim(1, 5000)
    ax.grid()
    ax.set_yscale('linear')
    ax.set_title("Fixed Budget Performance on OneMax Problem")




    fig, ax = plt.subplots(figsize=(16, 9))
    dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id") == 18), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
    ax.set_xlim(1, 5000)
    ax.grid()
    ax.set_yscale('linear')
    ax.set_title("Fixed Budget Performance on LABS Problem")

    plt.show()


#test_BinEA()


def benchmark_problem_class(algorithm, n_reps: int = 5, budget: int = 50000, problems="MaxCut", root_dir: str = ".", **kwargs):
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
            algorithm(budget=budget, **kwargs)(problem)
            problem.reset()
    

    manager = iohinspector.DataManager()
    manager.overview
    for idx in fids:
        manager.add_folder(f"PBO_pb{idx}")
    df = manager.load(True, True)

    fig, ax = plt.subplots(figsize=(16, 9))
    # Plot results for each id in fids
    for idx in fids :
        dt_plot = iohinspector.plot.single_function_fixedbudget(df.filter(pl.col("function_id") == idx), ax=ax, maximization=True, measures=['mean']) #We need to specify maximization=True 
    ax.set_xlim(1, 5000)
    ax.grid()
    ax.set_yscale('linear')
    ax.set_title("Fixed Budget Performance on "^problems^" instances")

    plt.show()



# benchmark_problem_class(BinaryEvolutionaryAlgorithm, n_reps=5, budget=500, problems="MaxCut", mutation_rate=0.05)







"""
if ([0 0 1 1 1 0 0 1 1 0 0 1 0 1 1 0 0 1 1 0 1 0 0 1 0 0 1 0 1 1 1 0 1 1 1 0 0 1 1 0 1 1 0 1 1 1 1 1 1 0 1 0 0 1 0 0 1 1 0 1 0 1 0 1 0 0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0 0 1 0 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 0 1 1 1 1 0 1 1 1 0 0 0 1 0 0 1 1 1 1 0 1 1 0 0 0 0 1 1 0 1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 1 1 1 0 1 0 0 1 0 1 0 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 1 1 1 1 0 0 0 0 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 0 1 0 0 1 0 0 0 0 1 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 0 1 0 1 0 0 1 0 1 1 1 0 0 0 1 1 0 1 1 1 0 1 1 0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 0 1 1 0 1 1 1 0 1 0 1 0 0 0 1 0 1 0 0 1 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0 1 1 1 0 0 1 0 1 1 1 0 0 1 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 1 1 0 0 0 0 1 0 0 1 1 1 0 1 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 1 0 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 1 1 0 1 1 1 1 0 1 1 0 0 0 1 0 1 1 0 1 0 0 0 1 1 1 1 1 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 0 1 1 1 1 1 0 0 1 1 0 1 1 0 1 0 0 0 1 0 1 1 1 1 0 1 0 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 0 1 0 1 1 1 1 0 1 0 0 0 1 1 0 0] == [0 1 0 1 0 0 1 1 1 1 0 1 1 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 0 1 1 0 1 1 0 1 0 0 0 0 1 0 0 0 1 1 1 1 1 0 1 0 1 0 0 0 1 1 0 1 1 1 0 1 1 0 1 1 1 1 0 1 0 1 0 0 1 1 1 0 1 0 0 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 0 0 0 1 1 0 0 1 1 0 0 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1 0 0 0 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 1 1 1 0 0 0 1 0 1 0 1 1 1 1 0 0 0 1 0 1 0 1 0 0 1 1 1 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 0 1  0 1 1 0 0 1 0 0 1 1 1 1 1 1 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 1 1 1 1 0 0 0 0 1 0 1 1 0 1 1 0 0 0 0 0 1 1 0 1 0 1 1 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 0 1 0 0 0 1 1 0 0 1 1 1 1 0 0 1 0 1 0 1 1 0 0 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 1 0 1 0 1 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 1 1 0 1 0 0 0 1 0 1 1 0 1 0 0 0 1 1 0 0 1 0 1 0 1 1 1 1 0 1 0 1 1 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 0 1 0 0 1 1 1 0 0 1 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 0 0 1 1 0 0 1]):
    print("true")
else : print("false")
not the same !!! -> no seed
"""



print("\nhello world\n")