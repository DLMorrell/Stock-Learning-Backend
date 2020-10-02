import alpaca_trade_api as tradeapi
import os
from time import sleep
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class market_data():
    def __init__(self):
        self.alpaca = tradeapi.REST(os.getenv('API'), os.getenv('SECRET'), os.getenv('ENDPOINT'), 'v2')
    
    # barDuration = 'minute' | '1Min' | '5Min' | '15Min' | 'day' | '1D'
    # symbol = stock symbol your getting the price for.
    # limit = number of elements
    def getPrice(self, barDuration, symbol, limit ):
        price = self.alpaca.get_barset(symbol, barDuration, limit = limit)
        symbol_price = price[symbol]

        return symbol_price
    
    # Number of shares owned
    def getShares(self, symbol):
        Position = self.alpaca.get_position(symbol)
        return Position 

    # Wait for market to open
    def waitForMarketToOpen(self):
        while True:
            time = self.alpaca.get_clock()
            sleep(10)#10 seconds
            if time.is_open:
                print("****** Market is Open ******")
                break
            print("-- Waiting for market to open --")

