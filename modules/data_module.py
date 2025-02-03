import yfinance as yf
import pandas as pd

def get_stock_data(tickers, period="1y", interval="1d"):
    """
    주식 데이터를 다운로드하고 전처리하는 함수
    
    Args:
        tickers (list): 주식 심볼 리스트
        period (str): 데이터 수집 기간
        interval (str): 데이터 간격
        
    Returns:
        tuple: (학습 데이터, 검증 데이터)
    """
    # 주식 데이터 다운로드
    data = yf.download(tickers, period=period, interval=interval)["Adj Close"]
    data = data.sort_index().dropna()
    
    # 학습/검증 데이터 분할 (80:20)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    val_data = data.iloc[split_idx:].reset_index(drop=True)
    
    return train_data, val_data

# 기본 주식 심볼 리스트
DEFAULT_TICKERS = [
    "AAPL",      # Apple
    "DLB",       # Dolby Laboratories
    "DIS",       # Disney
    "MSFT",      # Microsoft
    "META",      # Meta Platforms
    "BRK-A",     # Berkshire Hathaway A
    "AVGO",      # Broadcom
    "AMZN",      # Amazon
    "IONQ",      # IonQ
    "NVDA",      # Nvidia
    "OKLO",      # Oklo
    "LUNR",      # Intuitive Machines
    "JPM",       # JP Morgan
    "CAKE",      # Cheesecake Factory
    "KO",        # Coca Cola
    "TSLA",      # Tesla
    "TTWO",      # TTWO
    "HON",       # Honeywell
    "ARM"        # Arm Holdings
] 