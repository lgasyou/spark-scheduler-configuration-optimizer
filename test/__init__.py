import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('../results/logs/jobruntime.csv')
    end_time = data['simulate_end_time']
    start_time = data['simulate_start_time']
    time_costs = end_time - start_time
    print('Time Costs: \n', time_costs)
    sum_time_cost = time_costs.sum()
    print('Sum Time Cost:', sum_time_cost, 'ms')
