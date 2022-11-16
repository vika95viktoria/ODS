import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import datetime
import re
import calendar
from dateutil.relativedelta import relativedelta
from itertools import product


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
    item_with_categories_dict['item_name_correct'] = item_with_categories_dict['item_name'].apply(lambda x: x.lower())
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


def generate_all_ids_dataset(item_ids, shop_ids):
    matrix = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(34):
        matrix.append(np.array(list(product([i], shop_ids, item_ids)), dtype='int16'))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols, inplace=True)
    return matrix