def differential_evolution_binomial_crossover(n_x, p_cr):
    j_star = random.randint(1, n_x)
    J = {j_star}

    for j in range(1, n_x + 1):
        if random.uniform(0, 1) < p_cr and j != j_star:
            J.add(j)

    return J

def differential_evolution_mutation(population, mu,F,lb,ub):
    random_numbers = random.sample(range(1, mu + 1 ), 2)
    i2, i3 = random_numbers[0], random_numbers[1]
    ui= np.array(population[0])+F*(np.array(population[i2-1])-np.array(population[i3-1]))
    ui_adj=adjust_values(ui,lb,ub)
    return ui_adj

def stochastic_ranking_selection(population, fitness_func, constraints, Pf):
    fit = []
    penalties = []
    # Caches to store evaluated fitness and penalties
    fitness_cache = {}
    penalty_cache = {}
    # Evaluate each individual sequentially

    for ind in population:
        # Evaluate fitness and penalty for each individual
        fit_ind, penalty_sq = evaluate_individual(ind, fitness_func, constraints, fitness_cache, penalty_cache)
        fit.append(fit_ind)

        penalties.append(penalty_sq)
    # Stochastic ranking selection
    #print(fit)
    #print("initial penalty: ", penalties)
    for i in range(len(population)):
        swapped = False
        for j in range(len(population)-1):
            u = np.random.uniform(0, 1)
            # Stochastic selection condition
            if u < Pf or (penalties[j] == penalties[j + 1] and penalties[j] == 0):
                # Sort by fitness if penalties are equal or zero, or with probability Pf
                if fit[j] > fit[j + 1]:
                    #print(f"Swapping by fitnes: penalties[{j}]={penalties[j]} == penalties[{j + 1}]={penalties[j + 1]} or {u}<{Pf}")

                    #print(f"Swapping by fitness: fit[{j}]={fit[j]} > fit[{j + 1}]={fit[j + 1]}")

                    # Swap in population and in fit and penalties lists
                    population[j], population[j + 1] = population[j + 1], population[j]
                    fit[j], fit[j + 1] = fit[j + 1], fit[j]
                    penalties[j], penalties[j + 1] = penalties[j + 1], penalties[j]
                    swapped = True
            else:
                # Sort by penalty if the above condition is not met
                if penalties[j] > penalties[j + 1]:
                    #print(f"Swapping by penalty: penalties[{j}]={penalties[j]} > penalties[{j + 1}]={penalties[j + 1]}")

                    # Swap in population and in fit and penalties lists
                    population[j], population[j + 1] = population[j + 1], population[j]
                    fit[j], fit[j + 1] = fit[j + 1], fit[j]
                    penalties[j], penalties[j + 1] = penalties[j + 1], penalties[j]
                    swapped = True
        # Break loop if no swaps occurred in this iteration
        if not swapped:
            break

    # Verify and debug the order
    #print("Ordered penalties:", penalties)
    #print("Ordered fitness:", fit)
    #print("sorted penalty: ", penalties)

    # Return the first 100 individuals after sorting
    return population[:100], population[0]

def evaluate_individual(ind, fitness_func, constrains, fitness_cache, penalty_cache):
    ind_tuple = tuple(ind)  # Convert individual to tuple for caching

    # Evaluate fitness with caching
    """
    if ind_tuple not in fitness_cache:
        fit_ind = fitness_func(ind)
        fitness_cache[ind_tuple] = fit_ind
    else:
        fit_ind = fitness_cache[ind_tuple]"""
    fit_ind = fitness_func(ind)
    #print(ind)

    #print(fit_ind)
    # Evaluate constraints with caching
    """
    if ind_tuple not in penalty_cache:
        constrains_results = constrains(ind)
        penalty = [0 if res < 0 else res for res in constrains_results]
        penalty_sq = sum([pen**3 for pen in penalty])
        #print(penalty_sq)
        penalty_cache[ind_tuple] = penalty_sq
    else:
        penalty_sq = penalty_cache[ind_tuple]
    #print(penalty_sq)
    #print(fit_ind)"""
    constrains_results = constrains(ind)
    penalty = [0 if res < 0 else res for res in constrains_results]
    penalty_sq = sum([pen**3 for pen in penalty])

    return fit_ind, penalty_sq

def differential_evolution_real_encoding_stochastic_ranking(f, mu, lb, ub, type_opt,constraints):
    """
    Performs a differential evolution algorithm with real ecoding

    :param sym function f:  fitness function
    :param int mu: number of individuals in the population
    :param float lb= lower bound
    :param float ub= upper bound

    """
    new_gen=initialization(mu,lb,ub)
    gen_num=1
    dictionary = {}
    generation_best={}
    #Stopping criteria: distancia euclidiana promedio entre los pares de valores
    while(gen_num<=50):
        #print(new_gen)
        pc=0.9
        parents=new_gen
        new_gen,best_ind=stochastic_ranking_selection(parents,f,constraints,0.1)
        generation_best[gen_num] = best_ind

        for i in range(0,mu):
            xi=parents[i] #ith individual
            #mutation
            u1=differential_evolution_mutation(new_gen,mu,3,lb,ub)


            #crossover
            J=differential_evolution_binomial_crossover(len(u1),pc)#2 is the number of decision variables
            x_prime=[0]*len(xi)
            for j in range(1,len(xi)):
                if (j in J):
                    x_prime[j-1]=u1[j-1]
                else:
                    x_prime[j-1]=xi[j-1]
            x_prime_adj=adjust_values(x_prime,lb,ub)
            new_gen.append(x_prime_adj)

        #print(average_distance(new_gen))
        gen_num+=1
    #convergence_graph(best_fitnes_value_conv,gen_num-1)
    return generation_best

def violation_degree_check(constraints, point):
    """
    Evaluate the degree of violation of constraints at a given point.

    :param constraints: A list of constraint functions (lambda or numpy expressions)
    :param point: A NumPy array or list of decision variable values
    :return: The total penalty calculated based on constraint violations
    """
    constrains_results = []

    # Evaluate each constraint using the current point
    constrains_results=constraints(point)
    # Calculate penalties
    penalties = np.array([0 if res < 0 else res for res in constrains_results])
    penalty_sq = np.sum(np.power(penalties, 3))
    violation_degree = penalty_sq

    return violation_degree

def isfeasible_check(constraints, point):
    """
    Check if the last k individuals in the list of points are feasible.

    :param constraints: List of constraint functions (sympy expressions)
    :param points: List of points in the search space (list of tuples)
    :param k: Number of last individuals to check
    :return: True if all last k individuals are feasible, False otherwise
    """


    results=constraints(point)
    results_bool=[True if cons<=0 else False for cons in results]

    return results_bool

def initialization(mu, lb, ub):
    """
    Initialize the population

    :param int mu: number of individuals in the population
    :param list lb: lower bounds for each variable
    :param list ub: upper bounds for each variable
    :return: A list of individuals in the population
    :rtype: list
    """
    # Create the population array
    p = []

    for _ in range(mu):
        # Generate a random individual within the specified bounds
        y = np.random.uniform(lb, ub)  # Generate random numbers from uniform distribution
        p.append(y)

    return p