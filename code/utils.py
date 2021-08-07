def prefilter_items(data, take_n_popular=None, exclude_popular=None, filter_weeks_no_sold=None):
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

    # We will remove goods that are too cheap (we will not make money on them). 1 purchase from mailing lists costs 60 rubles.

    # Let's remove too expensive goods

    # Other filters

    return data


def postfilter_items(user_id, recommednations):
    pass