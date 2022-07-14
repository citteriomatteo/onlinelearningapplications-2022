import numpy as np

# Smoothing parameter for the click probability of the third product
LAMBDA = 0.7

# Number of days to test
NUM_OF_DAYS = 1000

# Window size of days for Sliding Window (proportional to the square root of NUM_OF_DAYS)
WINDOW_SIZE = int(np.sqrt(NUM_OF_DAYS))

# Number of user interactions per day
DAILY_INTERACTIONS = 500

NUM_PLOT_ITERATION = 20

# Total amount of products in the problem
NUM_PRODUCTS = 5

