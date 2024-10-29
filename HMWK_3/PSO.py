import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Parameters:
    def __init__(self, swarm_size, Gmax, c1, c2, w):
        self.swarm_size = swarm_size
        self.Gmax = Gmax
        self.c1 = c1
        self.c2 = c2
        self.w = w

class Particle:
    def __init__(self, objective_function, nvar, bounds):
        self.obj_func = objective_function
        self.x = np.empty(nvar)
        self.velocity = np.empty(nvar)
        self.bounds = bounds
        self.objective_value = None

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = np.array(x, copy=True)

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = np.array(velocity, copy=True)

    def evaluate_objective_function(self):
        self.objective_value = self.obj_func(self.x)

    def initialize_location(self, value=None):
        if value is None:
            xmin, xmax = self.bounds
            self.x = xmin + np.random.rand(len(self.x)) * (xmax - xmin)
            self.evaluate_objective_function()
        else:
            self.x = np.full(len(self.x), np.inf)
            self.objective_value = value

class ParticleFactory:
    def __init__(self, obj_func, nvar, bounds):
        self.obj_func = obj_func
        self.nvar = nvar
        self.bounds = bounds

    def create_particle(self):
        return Particle(self.obj_func, self.nvar, self.bounds)

class Swarm:
    def __init__(self, swarm_size, particle_factory):
        self.swarm_size = swarm_size
        self.particle_factory = particle_factory
        self.swarm = np.empty(swarm_size, dtype=object)

    def add_particle_at(self, index, particle):
        self.swarm[index] = particle

    def initialize_swarm(self):
        for i in range(self.swarm_size):
            particle = self.particle_factory.create_particle()
            particle.initialize_location()
            self.add_particle_at(i, particle)

class PSO:
    def __init__(self, params, obj_func, nvar, bounds):
        self.params = params
        self.obj_func = obj_func
        self.particle_factory = ParticleFactory(obj_func, nvar, bounds)
        self.swarm = Swarm(params.swarm_size, self.particle_factory)
        self.lbest = Swarm(params.swarm_size, self.particle_factory)
        self.c1 = self.params.c1
        self.c2 = self.params.c2
        self.w = self.params.w

    def run(self, execution, report):
        self.swarm.initialize_swarm()
        self.lbest.initialize_swarm()
        gbest = self.particle_factory.create_particle()
        gbest.initialize_location(np.inf)
        t = 0
        while t < self.params.Gmax:
            for i in range(self.swarm.swarm_size):
                particle = self.swarm.swarm[i]
                particle_lbest = self.lbest.swarm[i]
                
                if particle.objective_value < particle_lbest.objective_value:
                    particle_lbest.set_x(particle.get_x())
                    particle_lbest.evaluate_objective_function()
                
                if particle_lbest.objective_value < gbest.objective_value:
                    gbest.set_x(particle_lbest.get_x())
                    gbest.evaluate_objective_function()
            
            r1 = np.random.rand(len(gbest.get_x()))
            r2 = np.random.rand(len(gbest.get_x()))

            for i in range(self.swarm.swarm_size):
                particle = self.swarm.swarm[i]
                lbest = self.lbest.swarm[i]
                for j in range(len(gbest.get_x())):
                    current_velocity = particle.velocity[j]
                    current_position = particle.x[j]
                    personal_best = lbest.get_x()[j]
                    global_best = gbest.get_x()[j]

                    new_velocity = (
                        self.w * current_velocity +
                        self.c1 * r1[j] * (personal_best - current_position) +
                        self.c2 * r2[j] * (global_best - current_position)
                    )

                    particle.velocity[j] = new_velocity
                    particle.x[j] = current_position + new_velocity
                
                particle.evaluate_objective_function()
            report.add_best_individual_at_generation(t, execution, gbest)
            t += 1
        return gbest

class Report:
    def __init__(self, executions, generations):
        self.generations = generations
        self.best_individuals = []

    def add_best_individual_at_generation(self, generation, execution, individual):
        self.best_individuals.append((generation, execution, individual.objective_value))

    def save_to_csv(self, output_file):
        df = pd.DataFrame(self.best_individuals, columns=["Generation", "Execution", "Best Fitness"])
        df.to_csv(output_file, index=False)

    def plot_convergence(self, output_folder, func_name):
        for execution in range(len(self.best_individuals) // self.generations):
            execution_data = [record[2] for record in self.best_individuals if record[1] == execution]
            plt.plot(execution_data, label=f'Execution {execution + 1}')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(f'Convergence Plot for {func_name}')
        plt.legend()
        plt.savefig(f"{output_folder}/{func_name.replace(' ', '_')}_convergence.png")
        plt.clf()

class Experiment:
    def __init__(self, params, problem, number_executions):
        objective_func, bounds, func_name, nvar = problem
        self.pso = PSO(params, objective_func, nvar, bounds)
        self.number_executions = number_executions
        self.report = Report(number_executions, params.Gmax)
        self.func_name = func_name

    def execute(self, output_folder):
        for execution in range(self.number_executions):
            best_particle = self.pso.run(execution, self.report)
            print(f"Execution {execution + 1} - {self.func_name} - Best fitness= {best_particle.objective_value}")
        self.report.save_to_csv(f"{output_folder}/{self.func_name.replace(' ', '_')}_results.csv")
        self.report.plot_convergence(output_folder, self.func_name)

if __name__ == "__main__":
    problems = [
        (
            lambda x: 10 * len(x) + sum([(x_i**2 - 10 * np.cos(2 * np.pi * x_i)) for x_i in x]),
            np.array([-5.12, 5.12]).astype(float),
            "Rastrigin (n=2)",
            2
        ),
        (
            lambda x: sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)]),
            np.array([-2.5, 2.5]).astype(float),
            "Rosenbrock (n=2)",
            2
        )
    ]

    swarm_size = 30       # Number of particles
    Gmax = 100            # Generations
    c1 = 1.5              # Cognitive parameter
    c2 = 1.5              # Social parameter
    w = 0.5               # Inertia weight
    number_executions = 5
    output_folder = "experiment_results"

    params = Parameters(swarm_size, Gmax, c1, c2, w)

    for problem in problems:
        experiment = Experiment(params, problem, number_executions)
        experiment.execute(output_folder)


