import numpy as np

def bootstrap_metrics(list: list[float], num_iterations: int=1000):
    """
    Perform bootstrapping with replacement to estimate mean and standard deviation of WER/CER.
    
    Parameters:
    wer_list (list): List of word error rates
    num_iterations (int): Number of bootstrap iterations
    
    Returns:
    dict: Contains mean, standard deviation, and confidence interval
    """
    # Convert input list to numpy array
    np_list = np.array(list)
    n = len(np_list)
    
    # Initialize array to store bootstrap means
    bootstrap_means = np.zeros(num_iterations)
    
    # Perform bootstrapping
    for i in range(num_iterations):
        # Sample with replacement
        bootstrap_sample = np.random.choice(np_list, size=n, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    
    # Calculate statistics
    mean_estimate = np.mean(bootstrap_means)
    std_estimate = np.std(bootstrap_means, ddof=1)
    
    # Calculate confidence interval
    alpha = 1 - 0.95 # 95% confidence level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {
        'mean': mean_estimate,
        'std': std_estimate,
        'confidence_interval': (ci_lower, ci_upper),
    }