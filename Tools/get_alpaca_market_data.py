import alpaca_trade_api as tradeapi
import os
from time import sleep
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

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


    def getPrice(self, barDuration, symbol, limit, endDate ):
        price = self.alpaca.get_barset(symbol, barDuration, limit = limit, end = endDate)
        symbol_price = price[symbol]
        return symbol_price

    def getAllData(self, stock):
        startDate = "2015-01-04T00:00:00Z"#Beginning of time if possible
        endDate = "2020-10-02T00:00:00Z"#Change to todays date
        allData = []
        while datetime.strptime(startDate, '%Y-%m-%dT%H:%M:%SZ') <= datetime.strptime(endDate, '%Y-%m-%dT%H:%M:%SZ'):
            sampleData = self.getPrice('1Min', stock, 1000, endDate)
            endDate = str(sampleData[0].t)[:len(str(sampleData[0].t))-6].replace(' ', 'T')+'Z'
            for bar in sampleData:
                allData.insert(0, f'{bar.t}, {bar.o}, {bar.h}, {bar.l}, {bar.c}, {bar.v}')

        return allData
    
    
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

