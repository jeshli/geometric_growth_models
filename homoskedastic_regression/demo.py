
from regression_with_weight_variance import BurnRateAnalyzer

# Usage
if __name__ == "__main__":
    time_points = range(1, 11)
    burn_data = [17, 18, 32, 20, 20, 23, 28, 29, 52, 135]

    analyzer = BurnRateAnalyzer(time_points, burn_data)
    analyzer.plot_log_space()
    analyzer.plot_original_space()
    analyzer.print_results(12)