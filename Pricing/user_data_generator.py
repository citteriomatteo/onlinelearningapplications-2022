import json
import json
import numpy as np


class StandardDataGenerator:

    def __init__(self, filename):
        self._filename = filename
        self._data = None
        with open(self._filename) as f:
            self._data = json.load(f)
        # features
        self._features = self._data['features']
        # classes
        self._classes = {}
        # demand curve
        self._conversion_rates = self._data['conversion_rates']
        # prices
        #self._prices = self._data['prices']
        self._prices = np.array(self._data['prices'])

        for key in self._data['classes']:
            self._classes[key] = {}
            self._classes[key]['features'] = self._data['classes'][key]['features']
            self._classes[key]['fraction'] = self._data['classes'][key]['fraction']

        # alpha ratios
        self._alpha_ratios = self._data['alpha_ratios']
        # number of daily users
        self._num_daily_users = self._data['num_daily_users']
        # number of product sold
        self._num_product_sold = self._data['num_product_sold']
        self._secondaries = self._data['secondaries']

    def get_source(self) -> str:
        return self._filename

    def get_all(self):
        return self._data

    def get_features(self):
        return self._features

    def get_classes(self):
        return self._classes

    def get_alpha_ratios(self):
        return self._alpha_ratios

    def get_num_daily_users(self):
        return self._num_daily_users

    def get_num_product_sold(self):
        return self._num_product_sold

    def get_conversion_rates(self):
        return self._conversion_rates

    def get_prices(self):
        return self._prices

    def get_secondaries(self):
        return self._secondaries
