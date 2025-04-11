import numpy as np
from fuzzy import FuzzySelector, trapezoidal_membership
import matplotlib.pyplot as plt
def main():
    # Input range
    x = np.linspace(0, 10, 500)

    # Parameters for trapezoidal function
    params = np.array([2, 4, 6, 8])

    # Instantiate fuzzy selector with trapezoidal strategy
    selector = FuzzySelector(trapezoidal_membership)

    # Compute membership values
    membership = selector.computeMembership(x, params)

    # Save results for visualization
    np.savez("membership_output.npz", x=x, membership=membership, params=params)
    
    print("Membership values computed and saved to 'membership_output.npz'.")
    plt.plot(x, membership)
    plt.show()
if __name__ == "__main__":
    main()
