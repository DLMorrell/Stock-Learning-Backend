from Tools.trading_bot import trading_bot
from Tools.get_alpaca_market_data import market_data





if __name__ == "__main__":
    stock = 'SPY'

    md = market_data()

    print(md.getPrice('1Min', stock, 5))