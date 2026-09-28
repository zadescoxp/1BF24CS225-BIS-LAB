import numpy as np
import pandas as pd
import random


# ============================================================
# 1. PARAMETERS
# ============================================================

CAPITAL = 100_000

NUM_STOCKS = 100
TARGET_STOCKS = 10

POPULATION_SIZE = 100
GENERATIONS = 300

MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80

RISK_FREE_RATE = 0.06       # 6% annual
RISK_PENALTY = 1.0

MIN_WEIGHT = 0.02           # Minimum 2% if stock is selected
MAX_WEIGHT = 0.20           # Maximum 20% in one stock


# ============================================================
# 2. STOCK DATA
# ============================================================

# Replace this with your actual historical price data.
#
# rows    = trading days
# columns = stocks
#
# Example:
#
# prices = pd.read_csv("stock_prices.csv", index_col=0)
#
# For demonstration, we generate fake prices.

np.random.seed(42)

stocks = [f"Stock_{i+1}" for i in range(NUM_STOCKS)]

prices = pd.DataFrame(
    np.random.lognormal(
        mean=0.0005,
        sigma=0.02,
        size=(1000, NUM_STOCKS)
    ),
    columns=stocks
).cumprod()


# ============================================================
# 3. CALCULATE RETURNS
# ============================================================

returns = prices.pct_change().dropna()

# Annualized expected return
expected_returns = returns.mean() * 252

# Covariance matrix
covariance_matrix = returns.cov() * 252


# ============================================================
# 4. PORTFOLIO EVALUATION
# ============================================================

def portfolio_metrics(weights):

    expected_return = np.dot(
        weights,
        expected_returns.values
    )

    variance = np.dot(
        weights.T,
        np.dot(covariance_matrix.values, weights)
    )

    volatility = np.sqrt(variance)

    sharpe_ratio = (
        expected_return - RISK_FREE_RATE
    ) / volatility if volatility > 0 else 0

    return expected_return, volatility, sharpe_ratio


# ============================================================
# 5. CREATE A RANDOM PORTFOLIO
# ============================================================

def create_individual():

    # Start with no stocks
    weights = np.zeros(NUM_STOCKS)

    # Randomly select exactly TARGET_STOCKS
    selected = np.random.choice(
        NUM_STOCKS,
        TARGET_STOCKS,
        replace=False
    )

    # Generate random weights
    random_weights = np.random.uniform(
        MIN_WEIGHT,
        MAX_WEIGHT,
        TARGET_STOCKS
    )

    # Normalize so total = 100%
    random_weights /= random_weights.sum()

    # Assign weights
    weights[selected] = random_weights

    return weights


# ============================================================
# 6. CREATE INITIAL POPULATION
# ============================================================

def create_population():

    return [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]


# ============================================================
# 7. FITNESS FUNCTION
# ============================================================

def fitness(individual):

    # Number of stocks selected
    number_selected = np.count_nonzero(individual)

    # Must contain exactly TARGET_STOCKS
    if number_selected != TARGET_STOCKS:
        return -999999

    # Weights must sum to 1
    if not np.isclose(individual.sum(), 1):
        return -999999

    # Weight constraints
    selected_weights = individual[individual > 0]

    if np.any(selected_weights < MIN_WEIGHT):
        return -999999

    if np.any(selected_weights > MAX_WEIGHT):
        return -999999

    # Calculate portfolio metrics
    expected_return, volatility, sharpe = \
        portfolio_metrics(individual)

    # Risk-adjusted fitness
    fitness_value = (
        expected_return
        - RISK_PENALTY * volatility
    )

    return fitness_value


# ============================================================
# 8. SELECTION
# ============================================================

def selection(population):

    # Tournament selection
    tournament_size = 3

    selected = []

    for _ in range(POPULATION_SIZE):

        participants = random.sample(
            population,
            tournament_size
        )

        winner = max(
            participants,
            key=fitness
        )

        selected.append(winner.copy())

    return selected


# ============================================================
# 9. CROSSOVER
# ============================================================

def crossover(parent1, parent2):

    if random.random() > CROSSOVER_RATE:

        return parent1.copy(), parent2.copy()

    # Random crossover mask
    mask = np.random.rand(NUM_STOCKS) < 0.5

    child1 = np.where(
        mask,
        parent1,
        parent2
    )

    child2 = np.where(
        mask,
        parent2,
        parent1
    )

    # Repair both children
    child1 = repair(child1)
    child2 = repair(child2)

    return child1, child2


# ============================================================
# 10. REPAIR PORTFOLIO
# ============================================================

def repair(weights):

    # Keep positive weights
    selected = np.where(weights > 0)[0]

    # If too many stocks selected
    if len(selected) > TARGET_STOCKS:

        # Keep the largest weights
        largest = np.argsort(
            weights
        )[-TARGET_STOCKS:]

        new_weights = np.zeros(NUM_STOCKS)

        new_weights[largest] = weights[largest]

        weights = new_weights

    # If too few stocks selected
    elif len(selected) < TARGET_STOCKS:

        available = np.where(
            weights == 0
        )[0]

        needed = TARGET_STOCKS - len(selected)

        additional = np.random.choice(
            available,
            needed,
            replace=False
        )

        weights[additional] = np.random.uniform(
            MIN_WEIGHT,
            MAX_WEIGHT,
            needed
        )

    # Normalize
    weights = np.maximum(weights, 0)

    if weights.sum() > 0:
        weights /= weights.sum()

    return weights


# ============================================================
# 11. MUTATION
# ============================================================

def mutation(individual):

    individual = individual.copy()

    if random.random() < MUTATION_RATE:

        selected = np.where(
            individual > 0
        )[0]

        not_selected = np.where(
            individual == 0
        )[0]

        # Randomly replace one stock
        remove_stock = random.choice(selected)
        add_stock = random.choice(not_selected)

        individual[remove_stock] = 0

        individual[add_stock] = random.uniform(
            MIN_WEIGHT,
            MAX_WEIGHT
        )

        # Repair portfolio
        individual = repair(individual)

    return individual


# ============================================================
# 12. GENETIC ALGORITHM
# ============================================================

def genetic_algorithm():

    population = create_population()

    best_portfolio = None
    best_fitness = -np.inf

    for generation in range(GENERATIONS):

        # ------------------------------------
        # Evaluate population
        # ------------------------------------

        fitness_scores = [
            fitness(individual)
            for individual in population
        ]

        # Find best individual
        best_index = np.argmax(
            fitness_scores
        )

        generation_best = population[
            best_index
        ]

        generation_fitness = fitness_scores[
            best_index
        ]

        # Update global best
        if generation_fitness > best_fitness:

            best_fitness = generation_fitness

            best_portfolio = \
                generation_best.copy()

        # ------------------------------------
        # Selection
        # ------------------------------------

        selected = selection(population)

        # ------------------------------------
        # Crossover + Mutation
        # ------------------------------------

        new_population = []

        for i in range(
            0,
            POPULATION_SIZE,
            2
        ):

            parent1 = selected[i]

            parent2 = selected[
                (i + 1) % POPULATION_SIZE
            ]

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population[
            :POPULATION_SIZE
        ]

        # ------------------------------------
        # Progress
        # ------------------------------------

        if generation % 10 == 0:

            expected_return, volatility, sharpe = \
                portfolio_metrics(best_portfolio)

            print(
                f"Generation {generation:3d} | "
                f"Return: {expected_return:.2%} | "
                f"Risk: {volatility:.2%} | "
                f"Sharpe: {sharpe:.3f}"
            )

    return best_portfolio


# ============================================================
# 13. RUN THE ALGORITHM
# ============================================================

best_weights = genetic_algorithm()


# ============================================================
# 14. DISPLAY FINAL PORTFOLIO
# ============================================================

expected_return, volatility, sharpe = \
    portfolio_metrics(best_weights)

print("\n" + "=" * 60)
print("OPTIMIZED PORTFOLIO")
print("=" * 60)

print(f"Capital:          ₹{CAPITAL:,.0f}")
print(f"Expected Return:  {expected_return:.2%}")
print(f"Volatility:       {volatility:.2%}")
print(f"Sharpe Ratio:     {sharpe:.3f}")

print("\nStock Allocation:")
print("-" * 60)


portfolio = []

for i, weight in enumerate(best_weights):

    if weight > 0:

        amount = CAPITAL * weight

        portfolio.append({
            "Stock": stocks[i],
            "Weight": weight,
            "Investment": amount
        })


portfolio_df = pd.DataFrame(portfolio)

portfolio_df["Weight"] = (
    portfolio_df["Weight"] * 100
).round(2)

portfolio_df["Investment"] = (
    portfolio_df["Investment"]
).round(2)

print(portfolio_df.to_string(index=False))

print("\nTotal Investment:")
print(
    f"₹{portfolio_df['Investment'].sum():,.2f}"
)
