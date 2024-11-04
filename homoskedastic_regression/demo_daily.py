
from regression_with_weight_variance import BurnRateAnalyzer
import pandas as pd

# Usage
if __name__ == "__main__":

    DF = pd.concat([pd.read_csv('../data/daily/Dates.csv', delimiter='|'),
                    pd.read_csv('../data/daily/CyclesBurned.csv', delimiter='|')], axis=1)
    DF['Cycles Burned'] = DF['Cycles Burned'].str.replace(',', '')
    DF['Cycles Burned'] = pd.to_numeric(DF['Cycles Burned'].values)
    DF['Date'] = pd.to_datetime(DF['Date'])
    # DF = DF[DF['Date'] > '2022-02-01']  # filter out an outlier spike
    DF.reset_index(inplace=True, drop=True)

    burn_data = DF['Cycles Burned'].values
    Time_Data = DF['Date']
    time_points = ((Time_Data - Time_Data[0]).dt.days / 365).values

    analyzer = BurnRateAnalyzer(time_points, burn_data)
    analyzer.plot_log_space()
    analyzer.plot_original_space(type='line')

    analyzer.print_results(1)