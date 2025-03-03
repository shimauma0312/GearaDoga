import os
import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import schedule
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import configparser
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import yfinance as yf

# ロギング設定
def setup_logger():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"GEARA-DOGA_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger = logging.getLogger("GEARA-DOGA")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 設定ファイル読み込み
def load_config():
    config = configparser.ConfigParser()
    config_file = "config.ini"
    
    if not os.path.exists(config_file):
        config['MT5'] = {
            'login': '12345',
            'password': 'password',
            'server': 'MetaQuotes-Demo',
            'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
        }
        
        config['TRADING'] = {
            'symbol': 'USDCHF',  # デフォルトはスイスフラン
            'lot_size': '0.01',
            'max_positions': '5',
            'risk_percent': '2',
            'sl_pips': '50',
            'tp_pips': '100',
            'timeframe': 'H1',
            'data_period': '500'
        }
        
        config['ML'] = {
            'retrain_hours': '24',
            'prediction_threshold': '0.7',
            'features': 'open,high,low,close,volume,rsi,macd,bollinger',
            'use_sentiment': 'True'
        }
        
        with open(config_file, 'w') as f:
            config.write(f)
    else:
        config.read(config_file)
        
    return config

class GEARA_DOGA_FXBot:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("GEARA-DOGA FX自動売買システムを初期化中...")
        
        self.config = load_config()
        self.symbol = self.config['TRADING']['symbol']
        self.lot_size = float(self.config['TRADING']['lot_size'])
        self.max_positions = int(self.config['TRADING']['max_positions'])
        self.risk_percent = float(self.config['TRADING']['risk_percent'])
        self.sl_pips = int(self.config['TRADING']['sl_pips'])
        self.tp_pips = int(self.config['TRADING']['tp_pips'])
        
        self.timeframe_dict = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        self.timeframe = self.timeframe_dict[self.config['TRADING']['timeframe']]
        self.data_period = int(self.config['TRADING']['data_period'])
        
        self.model_path = "models/prediction_model.joblib"
        self.scaler_path = "models/scaler.joblib"
        
        # MT5の初期設定
        self.mt5_initialized = False
        
        # 機械学習モデル
        self.model = None
        self.scaler = None
        
        # T5モデルの初期化
        self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        self.t5_model = TFT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        
        # モデルのディレクトリ確認
        if not os.path.exists("models"):
            os.makedirs("models")
            
        # 前回の予測保存用
        self.last_prediction = None
    
    def initialize_mt5(self):
        try:
            # MT5に接続
            if not mt5.initialize(
                login=int(self.config['MT5']['login']),
                password=self.config['MT5']['password'],
                server=self.config['MT5']['server'],
                path=self.config['MT5']['path']
            ):
                raise Exception(f"MT5初期化失敗: {mt5.last_error()}")
            
            self.mt5_initialized = True
            self.logger.info(f"MT5の初期化に成功しました。バージョン: {mt5.version()}")
            return True
        except Exception as e:
            self.logger.error(f"MT5初期化エラー: {str(e)}")
            return False
    
    def shutdown_mt5(self):
        if self.mt5_initialized:
            mt5.shutdown()
            self.mt5_initialized = False
            self.logger.info("MT5接続を終了しました")
    
    def fetch_market_data(self):
        try:
            if not self.mt5_initialized and not self.initialize_mt5():
                return None
            
            # レート情報の取得
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"シンボル {self.symbol} は利用できません")
                return None
            
            if not symbol_info.visible:
                self.logger.info(f"シンボル {self.symbol} の表示設定")
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"シンボル {self.symbol} の選択失敗")
                    return None
            
            # ローソク足データの取得
            data = mt5.copy_rates_from_pos(
                self.symbol, 
                self.timeframe, 
                0, 
                self.data_period
            )
            
            if data is None or len(data) == 0:
                self.logger.error("マーケットデータの取得に失敗しました")
                return None
            
            # DataFrameに変換
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # テクニカル指標の追加
            df = self.add_technical_indicators(df)
            
            self.logger.info(f"{self.symbol}のマーケットデータを取得: {len(df)}行")
            return df
        
        except Exception as e:
            self.logger.error(f"マーケットデータ取得エラー: {str(e)}")
            return None
    
    def add_technical_indicators(self, df):
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ボリンジャーバンド
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bollinger_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']
        
        # ATR (Average True Range)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 移動平均
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # 前日比変化率
        df['pct_change'] = df['close'].pct_change()
        
        # 欠損値処理
        df.dropna(inplace=True)
        
        return df
    
    def fetch_news_data(self):
        """ニュースデータを取得してセンチメント分析に使用"""
        try:
            # Yahoo Financeからニュースを取得
            currency_pair = self.symbol[:3] + "=" + self.symbol[3:]
            ticker = yf.Ticker(currency_pair)
            news = ticker.news
            
            if not news:
                self.logger.info(f"{self.symbol}に関するニュースが見つかりませんでした")
                return None
            
            # ニュースのタイトルを抽出
            news_titles = [item['title'] for item in news[:5]]  # 最新5件のみ
            
            self.logger.info(f"{len(news_titles)}件のニュースを取得しました")
            return news_titles
        
        except Exception as e:
            self.logger.error(f"ニュースデータ取得エラー: {str(e)}")
            return None
    
    def analyze_sentiment(self, news_titles):
        """MT5を使用してニュースのセンチメント分析を行う"""
        if not news_titles:
            return 0.0  # ニュースがない場合は中立
        
        try:
            sentiments = []
            
            for title in news_titles:
                # T5モデルを使用してセンチメント分析
                input_text = f"sentiment analysis: {title}"
                input_ids = self.tokenizer(input_text, return_tensors="tf").input_ids
                
                outputs = self.t5_model.generate(
                    input_ids,
                    max_length=8,
                    num_beams=4,
                    early_stopping=True
                )
                
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 結果を数値に変換
                if "positive" in result.lower():
                    sentiments.append(1.0)
                elif "negative" in result.lower():
                    sentiments.append(-1.0)
                else:
                    sentiments.append(0.0)
            
            # 平均センチメントスコア
            avg_sentiment = sum(sentiments) / len(sentiments)
            self.logger.info(f"ニュース分析結果: センチメントスコア = {avg_sentiment:.2f}")
            
            return avg_sentiment
        
        except Exception as e:
            self.logger.error(f"センチメント分析エラー: {str(e)}")
            return 0.0
    
    def train_model(self, force=False):
        """機械学習モデルの学習"""
        # 前回の学習から設定時間が経過していない場合はスキップ
        model_retrain_hours = int(self.config['ML']['retrain_hours'])
        model_exists = os.path.exists(self.model_path)
        
        if model_exists and not force:
            model_time = os.path.getmtime(self.model_path)
            model_datetime = datetime.fromtimestamp(model_time)
            hours_passed = (datetime.now() - model_datetime).total_seconds() / 3600
            
            if hours_passed < model_retrain_hours:
                self.logger.info(f"前回の学習から{hours_passed:.1f}時間が経過しています。{model_retrain_hours}時間経過後に再学習します。")
                # モデルの読み込み
                if self.model is None:
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                return
        
        self.logger.info("機械学習モデルの学習を開始します")
        
        # データの取得
        df = self.fetch_market_data()
        if df is None or len(df) < 100:  # 十分なデータがない場合
            self.logger.error("学習に必要なデータが不足しています")
            return
        
        try:
            # 特徴量と目標変数の準備
            feature_list = self.config['ML']['features'].split(',')
            
            # 利用可能な特徴量のみを使用
            available_features = [f for f in feature_list if f in df.columns]
            
            X = df[available_features].values
            
            # 目標: 次の期間の価格変動
            y = df['close'].pct_change().shift(-1).fillna(0).values
            
            # トレーニングとテストデータに分割
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # スケーリング
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # モデルのトレーニング
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # モデルの評価
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.logger.info(f"モデル学習完了: トレーニングスコア={train_score:.4f}, テストスコア={test_score:.4f}")
            
            # モデルの保存
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info(f"モデルを保存しました: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"モデル学習エラー: {str(e)}")
    
    def predict_market(self):
        """市場の動きを予測"""
        if self.model is None:
            self.logger.info("モデルが読み込まれていません。学習を実行します。")
            self.train_model()
            
            if self.model is None:
                self.logger.error("モデルの準備ができていません。予測を中止します。")
                return None
        
        try:
            # 最新データの取得
            df = self.fetch_market_data()
            if df is None or len(df) < 10:
                self.logger.error("予測に十分なデータがありません")
                return None
            
            # 特徴量の準備
            feature_list = self.config['ML']['features'].split(',')
            available_features = [f for f in feature_list if f in df.columns]
            
            latest_data = df[available_features].iloc[-1].values.reshape(1, -1)
            latest_data_scaled = self.scaler.transform(latest_data)
            
            # 予測
            prediction = self.model.predict(latest_data_scaled)[0]
            
            # ニュースセンチメントの考慮
            if self.config.getboolean('ML', 'use_sentiment'):
                news_titles = self.fetch_news_data()
                sentiment_score = self.analyze_sentiment(news_titles)
                
                # センチメントスコアを予測に組み込む (重みは0.3)
                prediction = prediction * 0.7 + sentiment_score * 0.3
            
            self.last_prediction = prediction
            self.logger.info(f"予測結果: {prediction:.6f} (正: 上昇予測, 負: 下降予測)")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"予測エラー: {str(e)}")
            return None
    
    def calculate_position_size(self):
        """リスク管理に基づくポジションサイズの計算"""
        try:
            if not self.mt5_initialized and not self.initialize_mt5():
                return self.lot_size
            
            # アカウント情報
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("アカウント情報の取得に失敗しました")
                return self.lot_size
            
            balance = account_info.balance
            
            # シンボル情報
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"シンボル情報の取得に失敗しました: {self.symbol}")
                return self.lot_size
            
            # 通貨ペアの現在の価格
            current_price = mt5.symbol_info_tick(self.symbol).ask
            
            # リスク額の計算 (口座残高の一定割合)
            risk_amount = balance * (self.risk_percent / 100)
            
            # ストップロス幅 (pips)
            sl_price_diff = self.sl_pips * symbol_info.point * 10
            
            # 必要なロットサイズの計算
            if sl_price_diff > 0:
                # 1ロットのサイズは通常100,000単位
                contract_size = symbol_info.trade_contract_size
                lot_size = risk_amount / (sl_price_diff * contract_size)
                
                # ロットサイズの調整
                lot_step = symbol_info.volume_step
                lot_size = round(lot_size / lot_step) * lot_step
                
                # 最小・最大ロットサイズの制限を適用
                lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
                
                self.logger.info(f"計算されたロットサイズ: {lot_size}, リスク額: {risk_amount:.2f}")
                return lot_size
            else:
                return self.lot_size
                
        except Exception as e:
            self.logger.error(f"ポジションサイズ計算エラー: {str(e)}")
            return self.lot_size
    
    def execute_trade(self):
        """予測に基づいて取引を実行"""
        prediction = self.predict_market()
        if prediction is None:
            self.logger.error("予測結果がないため取引を中止します")
            return
        
        try:
            if not self.mt5_initialized and not self.initialize_mt5():
                return
            
            # 現在のポジション数の確認
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                positions = []
            
            current_positions = len(positions)
            
            # 最大ポジション数のチェック
            if current_positions >= self.max_positions:
                self.logger.info(f"現在のポジション数 ({current_positions}) が最大値 ({self.max_positions}) に達しています")
                return
            
            # シンボル情報
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"シンボル情報の取得に失敗しました: {self.symbol}")
                return
            
            # 予測閾値
            threshold = float(self.config['ML']['prediction_threshold'])
            
            # 取引サイズの計算
            lot_size = self.calculate_position_size()
            
            # 現在の価格
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.logger.error(f"価格情報の取得に失敗しました: {self.symbol}")
                return
            
            if prediction > threshold:
                # 買いシグナル
                self.logger.info(f"買いシグナル検出: 予測値 {prediction:.6f} > 閾値 {threshold}")
                
                # 注文パラメータの設定
                price = tick.ask
                sl = price - (self.sl_pips * symbol_info.point * 10)
                tp = price + (self.tp_pips * symbol_info.point * 10)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "GEARA-DOGA-BUY",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # 注文送信
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"買い注文を実行しました: {lot_size}ロット、価格 {price:.5f}")
                else:
                    self.logger.error(f"買い注文が失敗しました: {result.retcode}, {result.comment}")
            
            elif prediction < -threshold:
                # 売りシグナル
                self.logger.info(f"売りシグナル検出: 予測値 {prediction:.6f} < 閾値 {-threshold}")
                
                # 注文パラメータの設定
                price = tick.bid
                sl = price + (self.sl_pips * symbol_info.point * 10)
                tp = price - (self.tp_pips * symbol_info.point * 10)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "GEARA-DOGA-SELL",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # 注文送信
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"売り注文を実行しました: {lot_size}ロット、価格 {price:.5f}")
                else:
                    self.logger.error(f"売り注文が失敗しました: {result.retcode}, {result.comment}")
            
            else:
                self.logger.info(f"シグナルなし: 予測値 {prediction:.6f}, 閾値 {threshold}")
        
        except Exception as e:
            self.logger.error(f"取引実行エラー: {str(e)}")
    
    def check_positions(self):
        """現在のポジションを確認し、状況を報告"""
        try:
            if not self.mt5_initialized and not self.initialize_mt5():
                return
            
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                self.logger.info(f"現在の{self.symbol}ポジション: なし")
                return
            
            total_profit = 0
            position_count = len(positions)
            
            if position_count > 0:
                position_info = []
                for position in positions:
                    position_type = "買い" if position.type == mt5.POSITION_TYPE_BUY else "売り"
                    profit = position.profit
                    total_profit += profit
                    position_info.append(f"#{position.ticket}: {position_type}, {position.volume}ロット, 損益: {profit:.2f}")
                
                positions_str = "\n- ".join(position_info)
                self.logger.info(f"現在の{self.symbol}ポジション ({position_count}件):\n- {positions_str}\n合計損益: {total_profit:.2f}")
            else:
                self.logger.info(f"現在の{self.symbol}ポジション: なし")
            
        except Exception as e:
            self.logger.error(f"ポジション確認エラー: {str(e)}")
    
    def run(self):
        """メインの実行ループ"""
        self.logger.info("GEARA-DOGA FX自動売買システムを開始します")
        
        if not self.initialize_mt5():
            self.logger.error("MT5の初期化に失敗したため、システムを終了します")
            return
        
        # 初回の学習
        self.train_model()
        
        # スケジュール設定
        # モデルの再学習を定期的に実行 (デフォルト: 24時間ごと)
        retrain_hours = int(self.config['ML']['retrain_hours'])
        schedule.every(retrain_hours).hours.do(self.train_model)
        
        # 市場データの分析と取引を定期的に実行
        timeframe_map = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440
        }
        
        # 分析と取引の間隔を設定
        interval_minutes = timeframe_map.get(self.timeframe, 60)
        schedule.every(interval_minutes).minutes.do(self.execute_trade)
        
        # ポジション確認を15分ごとに実行
        schedule.every(15).minutes.do(self.check_positions)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("ユーザーによってシステムが停止されました")
        except Exception as e:
            self.logger.error(f"実行エラー: {str(e)}")
        finally:
            self.shutdown_mt5()
            self.logger.info("GEARA-DOGA FX自動売買システムを終了します")

if __name__ == "__main__":
    logger = setup_logger()
    try:
        bot = GEARA_DOGA_FXBot(logger)
        bot.run()
    except Exception as e:
        logger.critical(f"クリティカルエラー: {str(e)}")
