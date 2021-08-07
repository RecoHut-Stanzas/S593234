import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """
    Input
    -----
    user_item_matrix: pd.DataFrame
    """

    def __init__(self, data, weighting=True):

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Prepares auxiliary dictionaries"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Trains a model that recommends products among the items purchased by the user"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=128, regularization=0.05, iterations=15, num_threads=4):
        """ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """We recommend products similar to the top-N products purchased by the user"""

        recs = self.own_recommender.recommend(userid=self.userid_to_id[user],
                                              user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                              N=N,
                                              filter_already_liked_items=False,
                                              filter_items=None,
                                              recalculate_user=False)
        res = []
        for rec in recs:
            recs = self.model.similar_items(rec[0], N)

            res.append(self.id_to_itemid[recs[-1][0]])
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """We recommend top-N products among those purchased by similar users"""

        res = []
        users = self.model.similar_users(self.userid_to_id[user], N + 1)
        for user in users[1:]:
            recs = self.own_recommender.recommend(userid=user[0],
                                                  user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                  N=1,
                                                  filter_already_liked_items=False,
                                                  filter_items=None,
                                                  recalculate_user=False)
            res.append(self.id_to_itemid[recs[-1][0]])

        return res