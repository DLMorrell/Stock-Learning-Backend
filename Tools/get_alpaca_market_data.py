import alpaca_trade_api as tradeapi
import os
from time import sleep
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import pytest
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

    def getAllData(self, stock, startDate=None, endDate=None):
        print(f'LOG: Getting all {stock} data...')

        if endDate == None:
            today_raw = datetime.now().isoformat()
            endDate = str(today_raw)[:len(str(today_raw))-7].replace(' ', 'T')+'Z'

        if startDate == None:
            startDate = "2015-01-04T00:00:00Z"


        allData = []
        while datetime.strptime(startDate, '%Y-%m-%dT%H:%M:%SZ') <= datetime.strptime(endDate, '%Y-%m-%dT%H:%M:%SZ'):
            sampleData = self.getPrice('1Min', stock, 1000, endDate)
            endDate = str(sampleData[0].t)[:len(str(sampleData[0].t))-6].replace(' ', 'T')+'Z'
            for bar in sampleData:
                allData.insert(0, f'{bar.t}, {bar.o}, {bar.h}, {bar.l}, {bar.c}, {bar.v}')

        print(f'LOG: Done')
        return allData
    
    def saveAllToCsv(self, stock):
        file_name = f'{stock}_All_Data.csv'
        allData = self.getAllData(stock)
        col_names = 'Time, Open, High, Low, Close, Vol'
        print(f'LOG: Saving data...')
        with open(file_name, mode='w', newline='\n') as csv_file:
            csv_file.write(col_names + '\n')
            for row in allData:
                csv_file.write(row + '\n')
            csv_file.close()
        print(f'LOG: Done')
    
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

