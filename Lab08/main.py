import numpy as np


def evolutionary_algorithm_task():
    chromosome_size = 10
    population_size = 10
    generations = 0
    mutation_probability = 0.6

    population = np.random.choice((0, 1), size=(population_size, chromosome_size))

    while not any(np.sum(population, axis=1) == chromosome_size):
        population = update_population(population, mutation_probability)
        generations += 1

    print(generations)


def update_population(population, mutation_probability):
    sorted_indices = np.argsort(population.sum(axis=1))[::-1]
    population = population[sorted_indices]

    best, second_best = population[:2]

    cross_indexes = np.random.choice(len(best))

    child1 = np.concatenate((best[:cross_indexes], second_best[cross_indexes:]))
    child2 = np.concatenate((second_best[:cross_indexes], best[cross_indexes:]))

    child1 = mutate(child1, mutation_probability)
    child2 = mutate(child2, mutation_probability)

    population[-1] = child1
    population[-2] = child2

    return population


def mutate(child, mutation_probability):
    if np.random.rand() < mutation_probability:
        index = np.random.choice(len(child))
        child[index] = 1 - child[index]
    return child


def reverse_roulette_wheel_selection(array):
    probabilities = 1 / array
    return probabilities / np.sum(probabilities)


def genetic_algorithm_task():
    population_size = 10
    mutation_probability = 0.1
    generations = 0

    population = generate_initial_population(population_size)

    while True:
        differences = calculate_differences(population)
        if np.isin(0, differences):
            index = np.argmin(differences)
            list_a = [int("".join(map(str, number)), 2) for number in population[:, :4]]
            list_b = [int("".join(map(str, number)), 2) for number in population[:, 4:]]
            print(list_a[index])
            print(list_b[index])
            break

        percentages = reverse_roulette_wheel_selection(differences)
        new_population_indexes = np.random.choice(range(len(population)), size=population_size, p=percentages)
        population = population[new_population_indexes]

        population = update_population_genetic(population, mutation_probability)

        generations += 1

    print(generations)


backpack_data = np.array(
    [
        [3, 266],
        [13, 442],
        [10, 671],
        [9, 526],
        [7, 388],
        [1, 245],
        [8, 145],
        [8, 145],
        [2, 126],
        [9, 322],
    ]
)


def generate_initial_population(population_size):
    return np.array(
        [
            list(np.binary_repr(np.random.randint(0, 16), width=4) + np.binary_repr(np.random.randint(0, 16), width=4))
            for _ in range(population_size)
        ],
        dtype=np.uint8,
    )


def calculate_differences(population):
    list_a = [int("".join(map(str, number)), 2) for number in population[:, :4]]
    list_b = [int("".join(map(str, number)), 2) for number in population[:, 4:]]
    return np.abs(33 - (2 * np.power(list_a, 2) + list_b))


def update_population_genetic(population, mutation_probability):
    new_population = []
    for _ in range(len(population) // 2):
        parents_indexes = np.random.choice(range(len(population)), size=2, replace=False)
        cross_indexes = np.random.choice(range(8))
        parent_1, parent_2 = population[parents_indexes]
        child = np.concatenate((parent_1[:cross_indexes], parent_2[cross_indexes:]))
        child = mutate(child, mutation_probability)
        population[np.random.choice(parents_indexes)] = child
        new_population.append(child)

    return np.concatenate([population, np.array(new_population)])


def calculate_backpack_fitness(backpack):
    result = backpack * backpack_data[:, 0]
    result = np.sum(result, axis=1)
    result[result > 35] = 0
    return result


def backpack_problem_task():
    mutation_probability = 0.05
    population_size = 8
    generations = 0

    population = np.random.choice((0, 1), size=(population_size, 10))

    while True:
        fitness = calculate_backpack_fitness(population)

        values = population * backpack_data[:, 1]
        if np.any((np.sum(values, axis=1) > 2100) & (fitness > 0)):
            break

        percentages = fitness / np.sum(fitness)
        new_population_indexes = np.random.choice(range(len(population)), size=population_size // 4, p=percentages)

        population = update_population_backpack(population, new_population_indexes, mutation_probability)

        generations += 1

    print(f"Generation: {generations}")
    print_best_solution_backpack(population)


def update_population_backpack(population, new_population_indices, mutation_probability):
    parents = population[new_population_indices]
    new_population = [parents[0], parents[1]]

    for _ in range(int(len(population) * 0.75)):
        cut_index = np.random.choice(range(len(population)))
        child = np.concatenate((parents[0][:cut_index], parents[1][cut_index:]))
        child = mutate(child, mutation_probability)
        new_population.append(child)

    return np.array(new_population)


def print_best_solution_backpack(population):
    values = np.sum(population * backpack_data[:, 1], axis=1)
    index = np.argmax(values)
    weight = np.sum(population[index] * backpack_data[:, 0])

    if weight < 35:
        print(f"Backpack: {population[index]}")
        print(f"Value: {values[index]}")
        print(f"Weight: {weight}")
    else:
        print("No solution found.")


def calculate_distance(pos_1, pos_2):
    squared_distances = (pos_1 - pos_2) ** 2
    sum_of_squares = np.sum(squared_distances)
    euclides_dist = np.sqrt(sum_of_squares)
    return euclides_dist

salesman_data = np.array(
    [
        [119, 38],
        [37, 38],
        [197, 55],
        [85, 165],
        [12, 50],
        [100, 53],
        [81, 142],
        [121, 137],
        [85, 145],
        [80, 197],
        [91, 176],
        [106, 55],
        [123, 57],
        [40, 81],
        [78, 125],
        [190, 46],
        [187, 40],
        [37, 107],
        [17, 11],
        [67, 56],
        [78, 133],
        [87, 23],
        [184, 197],
        [111, 12],
        [66, 178],
    ]
)

def reverse_roulette(array):
    probabilities = 1 / array
    return probabilities / np.sum(probabilities)


def solve_traveling_salesman_problem():
    population_size = 100
    num_cities = len(salesman_data)
    elite_threshold = 0.2
    mutation_probability = 0.01
    generations = 0
    max_generations = 1000

    city_distances = []
    for city1 in salesman_data:
        row_distances = [calculate_distance(city1, city2) for city2 in salesman_data]
        city_distances.append(row_distances)
    city_distances = np.array(city_distances)

    initial_population = []
    for _ in range(population_size):
        individual = np.arange(num_cities)
        np.random.shuffle(individual)
        initial_population.append(individual)
    initial_population = np.array(initial_population)

    while True:
        roads = np.array([calculate_road(route, city_distances) for route in initial_population])

        if np.any(roads < 1200):
            break

        sel_probab = reverse_roulette(roads)

        selected_indexes = np.random.choice(
            range(population_size), size=int(population_size * elite_threshold), p=sel_probab
        )

        selected_routes = initial_population[selected_indexes]

        new_population = [*selected_routes]

        for _ in range(int(population_size * (1 - elite_threshold))):
            paretns_indexes = np.random.choice(
                range(len(selected_routes)), size=2, replace=False
            )
            parent_1, parent_2 = selected_routes[paretns_indexes]

            cross_indexes = np.sort(np.random.choice(num_cities, size=2, replace=False))
            lower, higher = cross_indexes

            child = np.full(num_cities, -2)
            child[lower:higher] = parent_1[lower:higher]

            index = 0
            for i in range(num_cities):
                if child[i] == -2:
                    while parent_2[index] in child:
                        index += 1
                    child[i] = parent_2[index]
                    index += 1

            if np.random.rand() < mutation_probability:
                mutation_index = np.random.choice(num_cities, size=2, replace=False)
                child[mutation_index[0]], child[mutation_index[1]] = (
                    child[mutation_index[1]],
                    child[mutation_index[0]],
                )

            new_population.append(child)

        initial_population = np.array(new_population).astype(np.int16)

        generations += 1

    print(f"Generations: {generations}")
    print_best_solution_salesman(initial_population, city_distances)


def calculate_road(cities, city_distances):
    return np.sum(city_distances[cities[:-1], cities[1:]])


def print_best_solution_salesman(population, city_distances):
    roads = np.array([calculate_road(pop, city_distances) for pop in population])
    index = np.argmin(roads)
    print(population[index])
    print(roads[index])


task_functions = {
    "1": evolutionary_algorithm_task,
    "2": genetic_algorithm_task,
    "3": backpack_problem_task,
    "4": solve_traveling_salesman_problem,
}

while True:
    choice = input("Which task: ")
    if choice in task_functions:
        task_functions[choice]()
        break
    else:
        print("Invalid task choice.")
