import os
import time
import datetime
import logging
import numpy as np
import pandas as pd
import pickle
import schedule
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from MetaTrader5 import *
from MetaTrader5 import shutdown
import MetaTrader5 as mt5


# ロギング設定
def setup_logger():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = f"{log_dir}/trading_{datetime.datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


logger = setup_logger()

# 環境変数の読み込み
load_dotenv()
ACCOUNT = int(os.getenv("MT5_ACCOUNT"))
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")
SYMBOL = os.getenv("TRADING_SYMBOL", "USDJPY")
TIMEFRAME = eval(os.getenv("TIMEFRAME", "TIMEFRAME_M5"))
LOT_SIZE = float(os.getenv("LOT_SIZE", "0.01"))
STOP_LOSS_PIPS = int(os.getenv("STOP_LOSS_PIPS", "20"))
TAKE_PROFIT_PIPS = int(os.getenv("TAKE_PROFIT_PIPS", "30"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/trading_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
TRAINING_PERIODS = int(os.getenv("TRAINING_PERIODS", "25000"))
RETRAIN_DAYS = int(os.getenv("RETRAIN_DAYS", "7"))


# MT5への接続
def connect_to_mt5():
    logger.info("MT5に接続を試みています...")

    # MetaTrader 5の初期化
    if not mt5.initialize():
        logger.error(f"MT5の初期化に失敗しました: {last_error()}")
        return False

    # トレーディングアカウントへのログイン
    if not login(ACCOUNT, password=PASSWORD, server=SERVER):
        logger.error(f"MT5へのログインに失敗しました: {last_error()}")
        shutdown()
        return False

    # 接続情報とアカウント情報の表示
    logger.info(f"MT5に接続しました - {account_info().name}")
    logger.info(f"取引サーバー: {account_info().server}")
    logger.info(f"残高: {account_info().balance} {account_info().currency}")
    logger.info(f"取引商品: {SYMBOL}, タイムフレーム: {TIMEFRAME}")

    return True


# 特徴量の作成
def create_features(data):
    logger.info(f"特徴量作成開始: {len(data)}行")

    # データのコピーを作成して断片化を防止
    data = data.copy()

    # 特徴量計算用の辞書 - 最後に一度にデータフレームに追加
    features = {}

    # 基本的な価格データ
    features['return'] = data['close'].pct_change()
    features['range'] = data['high'] - data['low']
    features['body'] = abs(data['open'] - data['close'])
    # ゼロ除算を防ぐ
    with np.errstate(divide='ignore', invalid='ignore'):
        features['body_to_range'] = features['body'] / features['range']

    # 移動平均と関連指標 (一部の期間だけ使用)
    for window in [10, 20, 50]:
        features[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        features[f'ma_diff_{window}'] = data['close'] - features[f'ma_{window}']
        # ゼロ除算を防ぐ
        with np.errstate(divide='ignore', invalid='ignore'):
            features[f'ma_ratio_{window}'] = data['close'] / features[f'ma_{window}']

        # ボリンジャーバンド
        features[f'std_{window}'] = data['close'].rolling(window=window).std()
        features[f'upper_band_{window}'] = features[f'ma_{window}'] + 2 * features[f'std_{window}']
        features[f'lower_band_{window}'] = features[f'ma_{window}'] - 2 * features[f'std_{window}']

        # ゼロ除算を防ぐ
        band_width = features[f'upper_band_{window}'] - features[f'lower_band_{window}']
        with np.errstate(divide='ignore', invalid='ignore'):
            features[f'bb_position_{window}'] = (data['close'] - features[f'lower_band_{window}']) / band_width
            features[f'bb_width_{window}'] = band_width / features[f'ma_{window}']

    # RSI (14期間)
    window = 14
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

    # ゼロ除算を防ぐ
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    features['ema_12'] = data['close'].ewm(span=12).mean()
    features['ema_26'] = data['close'].ewm(span=26).mean()
    features['macd'] = features['ema_12'] - features['ema_26']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # ATRの計算
    features['atr_14'] = calc_atr(data, 14)

    # 過去の価格変動 (一部だけ使用)
    for i in range(1, 3):
        features[f'close_lag_{i}'] = data['close'].shift(i)
        features[f'return_lag_{i}'] = features['return'].shift(i)

    # 時間帯特徴
    time_data = pd.to_datetime(data['time'])
    features['hour'] = time_data.dt.hour
    features['day_of_week'] = time_data.dt.dayofweek

    # セッション情報
    features['tokyo_session'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
    features['london_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
    features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)

    # ボリューム関連特徴
    if 'tick_volume' in data.columns:
        features['volume_ma_20'] = data['tick_volume'].rolling(window=20).mean()
        # ゼロ除算を防ぐ
        with np.errstate(divide='ignore', invalid='ignore'):
            features['volume_ratio_20'] = data['tick_volume'] / features['volume_ma_20']

    # 特徴量をデータフレームに一度に追加
    features_df = pd.DataFrame(features)

    # 元のデータと特徴量を結合
    result = pd.concat([data.reset_index(), features_df], axis=1)

    # 無限値を NaN に置き換え
    result = result.replace([np.inf, -np.inf], np.nan)

    # 欠損値の処理 - すべて削除するのではなく、主要な列だけで判断
    # または、必要に応じて欠損値を埋めることも検討
    essential_columns = ['close', 'return', 'ma_20', 'rsi', 'macd']
    result = result.dropna(subset=essential_columns)

    # 残りの欠損値を0で埋める
    result = result.fillna(0)

    logger.info(f"特徴量作成完了: {len(result)}行, {len(result.columns)}列")

    return result

# ATRの計算
def calc_atr(data, window):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(window=window).mean()


# 目標変数の作成（将来の値動きの予測）
def create_target(data, pips_threshold=5, look_ahead=3):
    # 将来の価格変動を予測（n期間後に価格がpips以上上昇するか）
    future_return = data['close'].shift(-look_ahead) - data['close']
    point_value = 0.01 if SYMBOL.endswith('JPY') else 0.0001  # 通貨ペアによって異なる
    pips_threshold_value = pips_threshold * point_value

    # 上昇(1)、下降(-1)、横ばい(0)の3クラス分類
    data['target'] = 0
    data.loc[future_return > pips_threshold_value, 'target'] = 1
    data.loc[future_return < -pips_threshold_value, 'target'] = -1

    return data

# データの取得
def get_market_data(symbol, timeframe, num_bars=20000):
    logger.info(f"{symbol}の過去{num_bars}本のデータを取得しています...")
    rates = copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    if rates is None or len(rates) == 0:
        logger.error(f"データの取得に失敗しました: {last_error()}")
        return None

    # データフレームに変換
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    logger.info(f"データを取得しました: {len(df)}行")
    return df


# モデルの学習
def train_model():
    logger.info("機械学習モデルのトレーニングを開始します...")

    # データの取得
    df = get_market_data(SYMBOL, TIMEFRAME, TRAINING_PERIODS)
    if df is None:
        return None, None

    # 特徴量と目標変数の作成
    df = create_features(df.reset_index())
    df = create_target(df)

    # トレーニングデータの準備
    X = df.drop(['target', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'], axis=1,
                errors='ignore')
    y = df['target']

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # モデルの学習
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    logger.info(f"特徴量: {X.columns.tolist()}")

    model.fit(X_scaled, y)

    logger.info(f"モデルのトレーニングが完了しました")

    # 特徴量の重要度
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("特徴量の重要度 (上位10):")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")

    # モデルと標準化のための情報を保存
    save_model(model, scaler)

    return model, scaler


# モデルの保存
def save_model(model, scaler):
    # モデルディレクトリがない場合は作成
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # モデルとスケーラーの保存
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"モデルを保存しました: {MODEL_PATH}")


# モデルの読み込み
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.info("既存のモデルが見つかりません。新しいモデルをトレーニングします。")
        return train_model()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        logger.info(f"モデルを読み込みました: {MODEL_PATH}")
        return model, scaler

    except Exception as e:
        logger.error(f"モデルの読み込みに失敗しました: {e}")
        logger.info("新しいモデルをトレーニングします。")
        return train_model()


# 予測の実行
def predict_market_direction(model, scaler):
    # 最新のデータを取得（予測用）
    df = get_market_data(SYMBOL, TIMEFRAME, 400)  # 400本分のデータを取得
    if df is None:
        return None

    # 特徴量の作成
    df = create_features(df.reset_index())

    # 予測に必要な特徴量のみを抽出
    X = df.drop(['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'], axis=1,
                errors='ignore')
    X = X.iloc[-1:].dropna(axis=1)  # 最新の1行を使用し、欠損値のある列は削除

    # scikit-learnのバージョン対応（feature_names_in_がない場合の処理）
    try:
        model_features = model.feature_names_in_
    except AttributeError:
        # 古いscikit-learnバージョン対応
        if hasattr(model, 'feature_importances_'):
            # トレーニング時と同じ特徴量の順序を維持する必要がある
            logger.warning("古いscikit-learnバージョンを検出しました。特徴量の順序に注意してください。")
            # 学習時の特徴量名がわからないため、Xの列をそのまま使用
            model_features = X.columns
        else:
            logger.error("モデルが正しくトレーニングされていません。")
            return None

    # 不足している特徴量を0で埋める
    missing_features = set(model_features) - set(X.columns)
    if missing_features:
        logger.warning(f"予測に必要な特徴量が不足しています: {missing_features}")
        # 不足している特徴量を0で埋める
        for feature in missing_features:
            X[feature] = 0

    # 予測に不要な特徴量を削除
    X = X[model_features]

    # 標準化
    X_scaled = scaler.transform(X)

    # 予測
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    # 予測結果のログ
    class_indices = {c: i for i, c in enumerate(model.classes_)}

    if prediction == 1:
        direction = "買い"
        probability = probabilities[class_indices.get(1, 0)]
    elif prediction == -1:
        direction = "売り"
        probability = probabilities[class_indices.get(-1, 0)]
    else:
        direction = "横ばい"
        probability = probabilities[class_indices.get(0, 0)]

    logger.info(f"予測結果: {direction} (確率: {probability:.4f})")

    return prediction, probability

# 取引実行
def execute_trade(prediction, probability, confidence_threshold=0.3):
    if prediction == 0 or probability < confidence_threshold:
        logger.info(f"取引条件を満たしていません（予測: {prediction}, 確率: {probability:.4f}, 閾値: {confidence_threshold}）")
        return

    # シンボル情報の取得
    symbol_info = symbol_info_tick(SYMBOL)
    if symbol_info is None:
        logger.error(f"シンボル情報の取得に失敗しました: {last_error()}")
        return

    # 現在のポジション数を確認
    positions = positions_get(symbol=SYMBOL)
    if positions is None:
        logger.error(f"ポジション情報の取得に失敗しました: {last_error()}")
        return

    # 既存ポジションが多すぎる場合は取引しない
    if len(positions) >= 5:
        logger.warning(f"既に{len(positions)}ポジションが開いています。新規取引を見送ります。")
        return

    # 価格情報
    current_price = symbol_info.ask if prediction == 1 else symbol_info.bid

    # ストップロスとテイクプロフィットの計算
    symbol_info_obj = mt5.symbol_info(SYMBOL)
    if symbol_info_obj is None:
        logger.error(f"シンボル情報の取得に失敗しました: {mt5.last_error()}")
        return
    
    point = symbol_info_obj.point
    sl_distance = STOP_LOSS_PIPS * (10 * point)
    tp_distance = TAKE_PROFIT_PIPS * (10 * point)

    if prediction == 1:  # 買い
        sl = current_price - sl_distance
        tp = current_price + tp_distance
        trade_type = ORDER_TYPE_BUY
    else:  # 売り
        sl = current_price + sl_distance
        tp = current_price - tp_distance
        trade_type = ORDER_TYPE_SELL

    # シンボルの充填モードを取得 - 充填モード定数を修正
    # MT5の充填モードを確認（異なるブローカーでサポートされる充填モードが異なる）
    logger.info(f"シンボル {SYMBOL} の充填モード: {symbol_info_obj.filling_mode}")
    
    # 充填モードを設定
    # 一般的に使用される充填モード
    filling_type = ORDER_FILLING_IOC  # 指定した量で注文を実行、または現在の市場価格でキャンセル
    
    # 注文送信
    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": trade_type,
        "price": current_price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": f"ML Scalper {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type_time": ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    logger.info(f"取引を実行します: {SYMBOL}, {'買い' if trade_type == ORDER_TYPE_BUY else '売り'}, "
                f"ロット: {LOT_SIZE}, 価格: {current_price}, SL: {sl}, TP: {tp}")

    # 注文送信
    result = order_send(request)
    if result.retcode != TRADE_RETCODE_DONE:
        logger.error(f"注文送信エラー: {result.retcode}, {result.comment}")
    else:
        logger.info(f"注文が成功しました: チケット番号 {result.order}")


# 定期的な再学習のスケジュール
def schedule_retraining():
    def retrain_job():
        logger.info("定期的な再学習を開始します...")
        train_model()

    # 毎週日曜日の深夜2時に再学習
    schedule.every().sunday.at("02:00").do(retrain_job)
    logger.info(f"モデルの再学習をスケジュールしました (間隔: {RETRAIN_DAYS}日)")


# メイン取引ループ
def trading_loop(model, scaler):
    try:
        # 現在の市場状態を予測
        prediction_result = predict_market_direction(model, scaler)
        if prediction_result is None:
            logger.error("予測の実行に失敗しました")
            return

        prediction, probability = prediction_result

        # 信頼度に基づいて取引を実行
        execute_trade(prediction, probability)

    except Exception as e:
        logger.error(f"取引ループでエラーが発生しました: {e}")


# メイン関数
def main():
    logger.info("スキャルピング自動売買ツールを起動しています...")

    # MT5に接続
    if not connect_to_mt5():
        logger.error("MT5への接続に失敗しました。プログラムを終了します。")
        return

    # モデルの読み込みまたは学習
    model, scaler = load_model()
    if model is None or scaler is None:
        logger.error("モデルの準備に失敗しました。プログラムを終了します。")
        shutdown()
        return

    # 再学習のスケジュール設定
    schedule_retraining()

    # メインループ
    logger.info("自動取引を開始します...")

    try:
        while True:
            # スケジュールされたタスクの実行
            schedule.run_pending()

            # 市場が開いている場合のみ取引を実行
            if not terminal_info().trade_allowed:
                logger.info("市場は現在取引不可の状態です。次の確認まで待機します。")
            else:
                # 取引ロジックの実行
                trading_loop(model, scaler)

            # 待機（5分間隔）
            time.sleep(300)

    except KeyboardInterrupt:
        logger.info("ユーザーによりプログラムが停止されました。")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
    finally:
        logger.info("MT5との接続を終了します。")
        shutdown()
        logger.info("プログラムを終了します。")


if __name__ == "__main__":
    main()
