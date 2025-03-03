import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import pytz
import configparser

# ロギング設定
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MT5TradingBot:
    def __init__(self, config_file='config.ini'):
        self.config = self._load_config(config_file)
        self.connected = False
        self.account_info = None
        self.symbols = self.config['Trading']['symbols'].split(',')
        self.timeframe = getattr(mt5, self.config['Trading']['timeframe'])
        self.lot_size = float(self.config['Trading']['lot_size'])
        self.max_positions = int(self.config['Risk']['max_positions'])
        self.stop_loss_pips = int(self.config['Risk']['stop_loss_pips'])
        self.take_profit_pips = int(self.config['Risk']['take_profit_pips'])
        self.strategy = self.config['Strategy']['name']
        
    def _load_config(self, config_file):
        """設定ファイルを読み込む"""
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
        
    def connect(self):
        """MT5に接続"""
        if not mt5.initialize():
            logging.error(f"MT5の初期化に失敗しました。エラー: {mt5.last_error()}")
            return False
            
        # ログイン情報
        login = int(self.config['MT5']['login'])
        password = self.config['MT5']['password']
        server = self.config['MT5']['server']
        
        if not mt5.login(login, password, server):
            logging.error(f"MT5へのログインに失敗しました。エラー: {mt5.last_error()}")
            mt5.shutdown()
            return False
            
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logging.error("アカウント情報を取得できませんでした")
            mt5.shutdown()
            return False
            
        logging.info(f"MT5に正常に接続しました。アカウント: {self.account_info.login}")
        self.connected = True
        return True
        
    def disconnect(self):
        """MT5から切断"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("MT5から切断しました")
            
    def get_historical_data(self, symbol, bars=100):
        """指定されたシンボルの過去データを取得"""
        if not self.connected:
            logging.error("MT5に接続されていません")
            return None
            
        # 現在の時間からUTCで取得
        utc_now = datetime.now(pytz.UTC)
        rates = mt5.copy_rates_from(symbol, self.timeframe, utc_now, bars)
        
        if rates is None or len(rates) == 0:
            logging.error(f"{symbol}の過去データを取得できませんでした")
            return None
            
        # DataFrameに変換
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def calculate_signals(self, df):
        """トレードシグナルを計算
        この関数は選択した戦略に基づいてカスタマイズできます
        """
        if self.strategy == 'moving_average_crossover':
            # 移動平均線クロスオーバー戦略
            fast_ma = int(self.config['Strategy']['fast_ma'])
            slow_ma = int(self.config['Strategy']['slow_ma'])
            
            df['ma_fast'] = df['close'].rolling(window=fast_ma).mean()
            df['ma_slow'] = df['close'].rolling(window=slow_ma).mean()
            
            # シグナル生成: 速いMAが遅いMAを上回ったら買い、下回ったら売り
            df['signal'] = 0
            df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1  # 買いシグナル
            df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1  # 売りシグナル
            
            # 前の行と比較してシグナルが変化した場合のみアクションを起こす
            df['action'] = df['signal'].diff()
            
        elif self.strategy == 'rsi':
            # RSI戦略
            rsi_period = int(self.config['Strategy']['rsi_period'])
            rsi_overbought = int(self.config['Strategy']['rsi_overbought'])
            rsi_oversold = int(self.config['Strategy']['rsi_oversold'])
            
            # RSI計算
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # シグナル生成
            df['signal'] = 0
            df.loc[df['rsi'] < rsi_oversold, 'signal'] = 1  # 買いシグナル
            df.loc[df['rsi'] > rsi_overbought, 'signal'] = -1  # 売りシグナル
            
            # 前の行と比較してシグナルが変化した場合のみアクションを起こす
            df['action'] = df['signal'].diff()
            
        else:
            logging.error(f"未サポートの戦略: {self.strategy}")
            df['action'] = 0
            
        return df
        
    def get_point(self, symbol):
        """シンボルの1ポイントの価値を取得"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"シンボル情報を取得できませんでした: {symbol}")
            return None
        return symbol_info.point
        
    def place_order(self, symbol, order_type, price=0.0):
        """注文を出す"""
        if not self.connected:
            logging.error("MT5に接続されていません")
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"シンボル情報を取得できませんでした: {symbol}")
            return None
            
        point = symbol_info.point
        
        # 注文リクエスト作成
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl": 0.0,  # 後で設定
            "tp": 0.0,  # 後で設定
            "deviation": 20,
            "magic": 123456,
            "comment": f"Python Bot {self.strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 買いか売りかに応じてSLとTPを設定
        if order_type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(symbol).ask
            sl = price - self.stop_loss_pips * point
            tp = price + self.take_profit_pips * point
        else:
            price = mt5.symbol_info_tick(symbol).bid
            sl = price + self.stop_loss_pips * point
            tp = price - self.take_profit_pips * point
            
        request["price"] = price
        request["sl"] = sl
        request["tp"] = tp
        
        # 注文送信
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"注文が失敗しました: {result.retcode}. {result.comment}")
            return None
            
        logging.info(f"注文が成功しました: {symbol}, {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'}, Lot: {self.lot_size}, Price: {price}, SL: {sl}, TP: {tp}")
        return result
        
    def close_all_positions(self):
        """すべてのポジションを閉じる"""
        if not self.connected:
            logging.error("MT5に接続されていません")
            return False
            
        positions = mt5.positions_get()
        if positions is None:
            logging.info("閉じるポジションがありません")
            return True
            
        for position in positions:
            # 注文タイプ（買いか売り）を反転
            order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            # 決済リクエスト
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 20,
                "magic": 123456,
                "comment": "Python Bot Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 注文送信
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"ポジション決済に失敗しました: {result.retcode}. {result.comment}")
                return False
                
            logging.info(f"ポジション決済に成功しました: {position.symbol}, Ticket: {position.ticket}")
            
        return True
        
    def count_positions(self):
        """現在のポジション数を取得"""
        if not self.connected:
            logging.error("MT5に接続されていません")
            return 0
            
        positions = mt5.positions_get()
        return len(positions) if positions else 0
        
    def run(self):
        """メイン実行ループ"""
        if not self.connected and not self.connect():
            logging.error("MT5に接続できないため、ボットを実行できません")
            return
            
        try:
            while True:
                for symbol in self.symbols:
                    # 過去データ取得
                    df = self.get_historical_data(symbol)
                    if df is None:
                        continue
                        
                    # シグナル計算
                    df = self.calculate_signals(df)
                    
                    # 最新の行のシグナルに基づいて行動
                    latest = df.iloc[-1]
                    
                    # ポジション数をチェック
                    positions_count = self.count_positions()
                    
                    if latest['action'] > 0 and positions_count < self.max_positions:  # 新しい買いシグナル
                        self.place_order(symbol, mt5.ORDER_TYPE_BUY)
                    elif latest['action'] < 0 and positions_count < self.max_positions:  # 新しい売りシグナル
                        self.place_order(symbol, mt5.ORDER_TYPE_SELL)
                        
                    # アカウント情報を表示
                    account = mt5.account_info()
                    logging.info(f"残高: {account.balance}, 有効証拠金: {account.equity}, マージン: {account.margin}")
                    
                # 設定された間隔で実行
                time_interval = int(self.config['Trading']['check_interval_seconds'])
                time.sleep(time_interval)
                
        except KeyboardInterrupt:
            logging.info("ユーザーによる中断")
        except Exception as e:
            logging.error(f"エラーが発生しました: {str(e)}")
        finally:
            # 終了時にはポジションを閉じるかどうか
            if self.config.getboolean('Trading', 'close_positions_on_exit', fallback=False):
                self.close_all_positions()
            self.disconnect()

if __name__ == "__main__":
    bot = MT5TradingBot()
    bot.run()
