import yfinance as yf
import pandas as pd
import numpy as np

class DataModule:
    """
    Data Module for handling stock market data
    주식 시장 데이터를 처리하는 데이터 모듈
    
    This module is responsible for:
    - Downloading historical stock data from Yahoo Finance
    - Preprocessing the data (cleaning, normalization)
    - Splitting data into training and validation sets
    
    이 모듈의 주요 기능:
    - Yahoo Finance에서 주식 히스토리 데이터 다운로드
    - 데이터 전처리 (클리닝, 정규화)
    - 학습용/검증용 데이터 분할
    """
    
    def __init__(self, tickers, period="6mo", interval="1d", train_split=0.9):
        """
        Initialize the Data Module
        데이터 모듈 초기화
        
        Args:
            tickers (list): List of stock ticker symbols
                          주식 종목 코드 리스트
            period (str): Data period to download (e.g., "6mo" for 6 months)
                        다운로드할 데이터 기간 (예: "6mo"는 6개월)
            interval (str): Data interval (e.g., "1d" for daily)
                         데이터 간격 (예: "1d"는 일별)
            train_split (float): Ratio for train/validation split (0.9 = 90% training)
                              학습/검증 데이터 분할 비율 (0.9 = 90% 학습용)
        """
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.train_split = train_split
        
    def download_data(self):
        """
        Download and preprocess stock data
        주식 데이터 다운로드 및 전처리
        
        Returns:
            pd.DataFrame: Preprocessed stock price data
                        전처리된 주식 가격 데이터
        """
        # Download adjusted close prices for all tickers
        # 모든 종목의 수정 종가 다운로드
        data = yf.download(self.tickers, period=self.period, interval=self.interval)["Adj Close"]
        
        # Sort by date and remove any rows with missing values
        # 날짜순으로 정렬하고 결측치가 있는 행 제거
        data = data.sort_index().dropna()
        
        return data
    
    def split_data(self, data):
        """
        Split data into training and validation sets
        데이터를 학습용과 검증용으로 분할
        
        Args:
            data (pd.DataFrame): Input stock price data
                              입력 주식 가격 데이터
                              
        Returns:
            tuple: (training_data, validation_data)
                  (학습용 데이터, 검증용 데이터)
        """
        # Calculate split index
        # 분할 지점 계산
        split_idx = int(len(data) * self.train_split)
        
        # Split and keep the date index
        # 데이터 분할 및 날짜 인덱스 유지
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        return train_data, val_data

    def get_data(self):
        """
        Main method to get processed and split data
        처리되고 분할된 데이터를 가져오는 메인 메서드
        
        Returns:
            tuple: (training_data, validation_data)
                  (학습용 데이터, 검증용 데이터)
        """
        # Download and preprocess data
        # 데이터 다운로드 및 전처리
        data = self.download_data()
        
        # Split into training and validation sets
        # 학습용과 검증용으로 분할
        train_data, val_data = self.split_data(data)
        
        return train_data, val_data
