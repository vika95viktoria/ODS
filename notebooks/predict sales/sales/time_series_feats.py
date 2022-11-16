import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def get_avg_metric_for_block(grouped, gr_local_cols, grouping, lag, date_block_num):
    lower_bound = date_block_num - lag
    upper_bound = date_block_num - 1
    grouped_local = grouped[(grouped.date_block_num >= lower_bound) & (grouped.date_block_num <= upper_bound)]
    gr_local = grouped_local.groupby(gr_local_cols).agg({f'{grouping}_avg_item_cnt': ['mean']})
  #  gr_local = grouped_local.groupby(gr_local_cols).agg({f'{grouping}_avg_item_cnt': ['mean', 'max', 'min']})
    gr_local.columns = [f'{grouping}_avg_item_cnt_lag_{lag}']
        #[f'{grouping}_avg_item_cnt_lag_{lag}', f'{grouping}_max_item_cnt_lag_{lag}', f'{grouping}_min_item_cnt_lag_{lag}']
    gr_local.reset_index(inplace=True)
    gr_local['date_block_num'] = date_block_num
    gr_local = gr_local.sort_values(by=gr_local_cols)
    return gr_local.to_numpy()


def get_avg_metric(grouped, group_cols, grouping, lag, date_block):
    date_blocks = grouped['date_block_num'].unique()
    gr_local_cols = [x for x in group_cols if x != 'date_block_num']
    agg_data = []
    if date_block is not None:
        return get_avg_metric_for_block(grouped, gr_local_cols, grouping, lag, date_block)
    for i in range(12, max(date_blocks) + 1):
        arr = get_avg_metric_for_block(grouped, gr_local_cols, grouping, lag, i)
        agg_data.extend(arr)
    return agg_data


def get_lag_metric(df, group_cols, date_block, lags):
    group = df.groupby(group_cols).agg({'item_cnt_month': ['mean']})
    gr_local_cols = [x for x in group_cols if x != 'date_block_num']
    avg_col_index = len(gr_local_cols)
    grouping = '_'.join(gr_local_cols)
    group.columns = [f'{grouping}_avg_item_cnt']
    group.reset_index(inplace=True)
    final_result = []
    for lag in lags:
        lag_res = get_avg_metric(group, group_cols, grouping, lag, date_block)
        if len(final_result) == 0:
            final_result = lag_res
        else:
          #  new_avg_data = np.array([x[avg_col_index:(avg_col_index + 3)] for x in lag_res])
            new_avg_data = np.array([x[avg_col_index:(avg_col_index + 1)] for x in lag_res])
            final_result = np.concatenate((final_result, new_avg_data), axis=1)
    lag_columns = [col for lag in lags[1:]
                   for col in [f'{grouping}_avg_item_cnt_lag_{lag}'
                             # , f'{grouping}_max_item_cnt_lag_{lag}',
                             #   f'{grouping}_min_item_cnt_lag_{lag}'
                               ]]
    columns = gr_local_cols + [f'{grouping}_avg_item_cnt_lag_{lags[0]}',
                               # f'{grouping}_max_item_cnt_lag_{lags[0]}',
                               # f'{grouping}_min_item_cnt_lag_{lags[0]}',
                               'date_block_num'] + lag_columns
    return pd.DataFrame(final_result, columns=columns)


def calculate_and_add_lag(df, group_cols, date_block=None):
    group_by_cols = get_lag_metric(df, group_cols, date_block, lags=[1, 2, 3, 6, 9, 12])
    column_dict = df.dtypes.to_dict()
    for c in group_cols:
        if c in column_dict:
            group_by_cols[c] = group_by_cols[c].astype(column_dict[c])
    return group_by_cols, group_cols

full_df = pd.read_csv('full.csv')
full_df = calculate_and_add_lag(full_df, ['date_block_num',  'item_id'])