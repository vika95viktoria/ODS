import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import re
import calendar
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta


def get_shop_type(regex, shop_name):
    matched = regex.search(shop_name)
    if matched:
        return matched.group()
    return 'МАГАЗИН'


def get_item_categories():
    return pd.read_csv('item_categories.csv')


def get_items(item_categories):
    items = pd.read_csv('items.csv.zip')
    item_with_categories_dict = items.merge(item_categories, on=['item_category_id'])
    item_with_categories_dict['category'] = item_with_categories_dict['item_category_name'].apply(
        lambda x: x.split(' - ')[0])
    item_with_categories_dict['item_name_correct'] = items['item_name'].apply(lambda x: x.lower())
    item_with_categories_dict['sub_category'] = item_with_categories_dict['item_category_name'].apply(
        lambda x: x.split(' - ')[0] if len(x.split(' - ')) < 2 else x.split(' - ')[1])
    item_with_categories_dict.drop(columns=['item_name', 'item_category_name'])
    return item_with_categories_dict


def get_shops():
    shops = pd.read_csv('shops.csv')
    shop_regex = re.compile(r'ТЦ|ТРЦ|ТРК|ТК')
    shops['shop_type'] = shops['shop_name'].apply(lambda x: get_shop_type(shop_regex, x))
    shops['city'] = shops['shop_name'].apply(lambda x: x.split(' ')[0])
    return shops


def prepare_full_dataset(df, items, shops, default_date=None):
    shop_id_map = {0: 57, 1: 58, 10: 11}
    df['shop_id'] = df['shop_id'].apply(lambda x: shop_id_map[x] if x in shop_id_map else x)
    df_with_shop = pd.merge(df, shops, on='shop_id')
    full_df = pd.merge(df_with_shop, items, on='item_id')
    if default_date:
        full_df['date'] = default_date
    full_df['date'] = full_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
    full_df['month'] = full_df['date'].apply(lambda x: x.month)
    full_df['year'] = full_df['date'].apply(lambda x: x.year)
    return full_df


def enrich_dataset(df, items, shops):
    df_with_shop = pd.merge(df, shops, on='shop_id')
    full_df = pd.merge(df_with_shop, items, on='item_id')
    full_df.drop(columns=['shop_name', 'item_name', 'item_category_name', 'item_name_correct'], inplace=True)
    return full_df



def prepare_zero_dataset(df, items, shops, item_categories):
    df_with_shop = pd.merge(df, shops, on='shop_id')
    df_with_item_info = pd.merge(df_with_shop, items, on='item_id')
    full_df = pd.merge(df_with_item_info, item_categories, on='item_category_id')
    full_df['category'] = full_df['item_category_name'].apply(lambda x: x.split(' - ')[0])
    full_df['sub_category'] = full_df['item_category_name'].apply(
        lambda x: x.split(' - ')[0] if len(x.split(' - ')) < 2 else x.split(' - ')[1])
    full_df['city'] = full_df['shop_name'].apply(lambda x: x.split(' ')[0])
    to_drop = [x for x in full_df.columns if x not in ['shop_id', 'shop_type', 'item_id', 'month', 'year',
                                                       'category', 'sub_category', 'city', 'item_category_id']]
    full_df.drop(columns=to_drop, inplace=True)
    return full_df


def get_duplicate_dict(items, test_df):
    count_item = items['item_name_correct'].value_counts().reset_index()
    duplicate_items = count_item[count_item.item_name_correct > 1]['index'].values
    duplicates = items[items.item_name_correct.isin(duplicate_items)]
    dupl_ids = duplicates['item_id'].values
    test_used_ids_from_duplicates = test_df[test_df.item_id.isin(dupl_ids)]['item_id'].unique()
    dupl_id_groups = duplicates.groupby('item_name_correct')['item_id'].apply(list).reset_index()['item_id'].values
    dup_item_dict = {}
    for gr in dupl_id_groups:
        fst_item = gr[0]
        if fst_item in test_used_ids_from_duplicates:
            dup_item_dict[gr[1]] = fst_item
        else:
            dup_item_dict[fst_item] = gr[1]
    return dup_item_dict


def get_first_day_of_month(year, month):
    return datetime.datetime(year=year, month=month, day=1).strftime("%Y-%m-%d")


def get_last_day_of_month(year, month):
    day = calendar.monthrange(year, month)[1]
    return datetime.datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")


def get_date_range(start_year, start_month, end_year, end_month):
    result = []

    end = datetime.date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
    current = datetime.date(start_year, start_month, 1)

    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return np.array(result)


def generate_all_ids_dataset(start_year, start_month, end_year, end_month, item_ids, shop_ids):
    all_dates = get_date_range(start_year, start_month, end_year, end_month)
    res = np.array(np.meshgrid(item_ids, shop_ids, all_dates)).T.reshape(-1, 3)
    parsed_cols = [[x.year, x.month, x.strftime('%d.%m.%Y')] for x in res[:, 2]]
    res_arr = np.hstack((res[:, :2], parsed_cols))
    item_id_dataset = pd.DataFrame(res_arr, columns=['item_id', 'shop_id', 'year', 'month', 'date'])
    item_id_dataset['month'] = item_id_dataset['month'].astype(np.int64)
    item_id_dataset['year'] = item_id_dataset['year'].astype(np.int64)
    return item_id_dataset

# shops = get_shops()
# item_categories = get_item_categories()
# items = get_items(item_categories)
# # # train_df = prepare_full_dataset(pd.read_csv('sales_train.csv.zip'), items, shops, item_categories)
# # # test_df = prepare_full_dataset(pd.read_csv('test.csv.zip'), items, shops, item_categories, default_date='01.11.2015')
# shop_ids = shops[~shops.shop_id.isin([0, 1, 10])]['shop_id'].values
# start = time.time()
# res = generate_all_ids_dataset(2013, 1, 2015, 10, items['item_id'].values, shop_ids)
# end = time.time()
# print(end - start)

