import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None, exclude_popular=None, filter_weeks_no_sold=None):
    # Let's remove the most popular products (they will be bought anyway)
    if exclude_popular is not None:
        popularity = data.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top = popularity.sort_values('n_sold', ascending=False).head(exclude_popular).item_id.tolist()
        data.loc[data['item_id'].isin(top), 'item_id'] = 999999

    # Leave popular products
    if take_n_popular is not None:
        popularity = data.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
        data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # We will remove products that have not been sold in the last n weeks
    if filter_weeks_no_sold is not None:
        data.loc[data['week_no'] > filter_weeks_no_sold, 'item_id'] = 999999

    # Let's remove the categories that are not interesting for recommendations (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # We will remove goods that are too cheap (we will not make money on them). 1 purchase from mailing lists costs 60 rubles.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Let's remove too expensive goods
    data = data[data['price'] < 50]

    # Let's take the top in popularity
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Let's create a fictitious item_id (if the user bought goods from the top 5000, then he "bought" such a product)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(user_id, recommednations):
    pass