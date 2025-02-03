import numpy as np
from scipy.stats import norm
import pandas as pd

class ImprovedStockTradingEnv:
    def __init__(self, price_data, initial_capital=10000, transaction_cost=0.01, window=5, 
                 risk_penalty=50, trade_penalty=5, validation_mode=False, prediction_penalty=50):
        """
        강화학습 환경 초기화
        
        Args:
            price_data (pd.DataFrame): 주가 데이터
            initial_capital (float): 초기 자본금
            transaction_cost (float): 거래 수수료
            window (int): 기술적 지표 계산을 위한 윈도우 크기
            risk_penalty (float): 리스크 패널티 가중치
            trade_penalty (float): 거래 패널티 가중치
            validation_mode (bool): 검증 모드 여부
            prediction_penalty (float): 예측 오차 패널티 가중치
        """
        self.price_data = price_data.copy().reset_index(drop=True)
        self.num_days = len(self.price_data)
        self.num_stocks = self.price_data.shape[1]
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window = window
        self.risk_penalty = risk_penalty
        self.trade_penalty = trade_penalty
        self.validation_mode = validation_mode
        self.prediction_penalty = prediction_penalty
        
        # 총 자산: 인덱스 0은 현금, 1~num_stocks는 주식
        self.num_assets = self.num_stocks + 1
        # 비율 계산을 위한 기준 가격
        self.initial_prices = self.price_data.iloc[0].values
        # 이산 행동 공간 생성
        self.actions = self._create_action_mapping()
        self.action_space = list(self.actions.keys())
        self.action_size = len(self.actions)
        self.portfolio_history = []
        self.max_portfolio_value = initial_capital
        self.last_bs_ratio = None
        self.reset()
    
    def _create_action_mapping(self):
        """
        행동 공간 매핑 생성
        0: 아무것도 하지 않음
        1~n: 현금에서 주식으로 이전 (다양한 비율)
        n+1~m: 주식에서 현금으로 이전 (다양한 비율)
        """
        actions = {0: None}
        action_id = 1
        fractions = [0.25, 0.5, 0.75, 1.0]
        
        # 현금에서 각 주식으로
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (0, stock, frac)
                action_id += 1
        
        # 각 주식에서 현금으로
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (stock, 0, frac)
                action_id += 1
        
        return actions
    
    def reset(self):
        """환경 초기화"""
        self.current_day = 0
        self.portfolio = np.zeros(self.num_assets)
        self.portfolio[0] = self.initial_capital
        self.portfolio_history = [self.get_total_value(self.current_day)]
        self.max_portfolio_value = self.initial_capital
        state = self._get_state()
        return state
    
    def _get_state(self):
        """
        현재 상태 벡터 반환
        - 포트폴리오 비율
        - 각 주식별 5가지 특성:
            1. 가격 비율
            2. 이동평균 비율
            3. 연간화된 변동성
            4. BS 비율
            5. 평균 로그 수익률
        """
        total_value = self.get_total_value(self.current_day)
        portfolio_frac = self.portfolio / total_value
        
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
            
        price_ratio = current_prices / self.initial_prices
        
        ma_ratios = []
        volatilities = []
        bs_ratios = []
        avg_returns = []
        
        # Black-Scholes 계산을 위한 상수
        T_const = 30 / 365  # 30일
        r = 0.01  # 무위험 이자율
        
        for i in range(self.num_stocks):
            start_idx = max(0, self.current_day - self.window + 1)
            window_prices = self.price_data.iloc[start_idx:self.current_day+1, i].values
            ma = np.mean(window_prices)
            ma_ratio = ma / self.initial_prices[i]
            ma_ratios.append(ma_ratio)
            
            if len(window_prices) > 1:
                log_returns = np.diff(np.log(window_prices))
                vol_daily = np.std(log_returns)
                vol_annual = vol_daily * np.sqrt(252)
                avg_ret = np.mean(log_returns)
            else:
                vol_annual = 0.0
                avg_ret = 0.0
                
            volatilities.append(vol_annual)
            avg_returns.append(avg_ret)
            
            # Black-Scholes 비율 계산
            S = current_prices[i]
            K = S
            if vol_annual > 0:
                d1 = (np.log(S/K) + (r + 0.5 * vol_annual**2) * T_const) / (vol_annual * np.sqrt(T_const))
                d2 = d1 - vol_annual * np.sqrt(T_const)
                call_price = S * norm.cdf(d1) - K * np.exp(-r * T_const) * norm.cdf(d2)
            else:
                call_price = S
            bs_ratio = call_price / S
            bs_ratios.append(bs_ratio)
        
        self.last_bs_ratio = np.array(bs_ratios)
        technical_features = np.concatenate([price_ratio, np.array(ma_ratios),
                                          np.array(volatilities), np.array(bs_ratios),
                                          np.array(avg_returns)])
        state = np.concatenate([portfolio_frac, technical_features])
        return state
    
    def get_total_value(self, day_index):
        """총 포트폴리오 가치 계산"""
        total = self.portfolio[0]
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        total += np.sum(self.portfolio[1:])
        return total
    
    def step(self, action):
        """
        환경 진행
        1. 현재 가격으로 거래 실행
        2. 다음 날로 이동
        3. 보상 계산
        """
        done = False
        current_prices = self.price_data.iloc[self.current_day].values
        value_before = self.get_total_value(self.current_day)
        
        # 거래 실행
        if action != 0:
            trade = self.actions[action]
            src, dst, frac = trade
            available = self.portfolio[src]
            transfer_amt = frac * available
            self.portfolio[src] -= transfer_amt
            self.portfolio[dst] += transfer_amt * (1 - self.transaction_cost)
            trade_flag = True
        else:
            trade_flag = False
        
        old_bs = self.last_bs_ratio.copy() if self.last_bs_ratio is not None else None
        
        # 다음 날로 이동
        self.current_day += 1
        if self.current_day >= self.num_days:
            done = True
            next_state = self._get_state()
            final_value = self.get_total_value(self.num_days - 1)
            reward = final_value - value_before
            self.portfolio_history.append(final_value)
            info = {'portfolio_value': final_value}
            return next_state, reward, done, info
        
        # 포트폴리오 가치 업데이트
        next_prices = self.price_data.iloc[self.current_day].values
        self.portfolio[1:] = self.portfolio[1:] * (next_prices / current_prices)
        
        value_after = self.get_total_value(self.current_day)
        self.max_portfolio_value = max(self.max_portfolio_value, value_after)
        drawdown = (self.max_portfolio_value - value_after) / self.max_portfolio_value
        
        # 보상 계산
        reward = (value_after - value_before) - self.risk_penalty * drawdown
        if trade_flag:
            reward -= self.trade_penalty
        
        next_state = self._get_state()
        
        # 검증 모드에서 추가 패널티
        if self.validation_mode and old_bs is not None:
            start_bs = self.num_assets + 2 * self.num_stocks
            new_bs = next_state[start_bs : start_bs + self.num_stocks]
            bs_error = np.mean(np.abs(new_bs - old_bs))
            reward -= self.prediction_penalty * bs_error
        
        self.portfolio_history.append(value_after)
        info = {'portfolio_value': value_after}
        return next_state, reward, done, info 