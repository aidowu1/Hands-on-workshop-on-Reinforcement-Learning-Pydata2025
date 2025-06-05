from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from typing import List, Dict, Any, Tuple
import os

import src.main.configs.global_configs as configs


class DataProvider:
    """
    Component used to provide the S & P 500 dataset
    :param s_and_p_raw_data_path: File path of raw data
    :param s_and_p_scaled_data_path: File path of scaled data
    """
    def __init__(
            self,
            s_and_p_raw_data_path: str = configs.S_P_RAW_DATA_PATH,
            s_and_p_scaled_data_path: str = configs.S_P_SCALED_DATA_PATH
    ):
        """
        Constructor
        """
        self._s_and_p_raw_data_path = s_and_p_raw_data_path
        self._s_and_p_scaled_data_path = s_and_p_scaled_data_path
        self._closing_price_raw_df = None
        self._closing_price_raw = None
        self._closing_price_scaled = None
        self._closing_price_train = None
        self._closing_price_test = None
        self._data_scaler = None
        self.features = None

    def _featureEngineerData(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineer the S & P price data
        :param df: Input dataframe
        :return: Feature engineered data
        """
        df[configs.FEATURE_SMA_10] = ta.trend.sma_indicator(df[configs.DATA_BAR_TYPE], window=10)
        df[configs.FEATURE_RSI] = ta.momentum.RSIIndicator(df[configs.DATA_BAR_TYPE], window=14).rsi()
        df[configs.FEATURE_MACD] = ta.trend.macd_diff(df[configs.DATA_BAR_TYPE])
        df.dropna(inplace=True)
        return df

    def getData(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the S & P closing price data
        :return: Datasets
        """
        if os.path.exists(self._s_and_p_raw_data_path):
            print \
                (f"{self._s_and_p_raw_data_path} already exists in local file system (cache), ingesting the file locally")
            self._closing_price_raw_df = pd.read_csv(self._s_and_p_raw_data_path, index_col=None)
        else:
            print \
                (f"{self._s_and_p_raw_data_path} does not exists in local file system (cache), so ingesting the file from Yahoo Finance remote endpoint..")
            self._closing_price_raw_df = yf.download(
                configs.S_AND_P_YAHOO_TICKER,
                start=configs.DATA_START_DATE,
                end=configs.DATA_END_DATE,
                multi_level_index=False)
            self._closing_price_raw_df.to_csv(self._s_and_p_raw_data_path, index=False)
        if os.path.exists(self._s_and_p_scaled_data_path):
            print \
                (f"{self._s_and_p_scaled_data_path} already exists in local file system (cache), ingesting the file locally")
            closing_price_scaled_df =  pd.read_csv(self._s_and_p_scaled_data_path, index_col=None)
            self._closing_price_with_features_scaled = closing_price_scaled_df[configs.S_AND_P_DATA_COLUMNS].values
        else:
            print \
                (f"{self._s_and_p_scaled_data_path} does not exists in local file system (cache), so will recompute the data scaling..")
            close_prices_df = self._closing_price_raw_df[[configs.DATA_BAR_TYPE]]
            close_prices_with_features_df = self._featureEngineerData(close_prices_df)

            close_prices_with_features = close_prices_with_features_df.values
            self._data_scaler = MinMaxScaler()
            self._closing_price_with_features_scaled = self._data_scaler.fit_transform(close_prices_with_features)
            closing_price_scaled_df = pd.DataFrame(self._closing_price_with_features_scaled, columns=[configs.S_AND_P_DATA_COLUMNS])
            closing_price_scaled_df.to_csv(self._s_and_p_scaled_data_path, index=False)

        self._partitionDataset()

        return  self._closing_price_raw_df, self._closing_price_with_features_scaled, self._closing_price_train, self._closing_price_test

    def _partitionDataset(self, slit_fraction: float =configs.TRAIN_SPLIT_FACTOR):
        """
        Partitions data into training and test splits
        :param slit_fraction: Split fraction
        """
        prices = self._closing_price_with_features_scaled
        split = int(len(prices) * slit_fraction)
        self._closing_price_train, self._closing_price_test = prices[:split], prices[split:]