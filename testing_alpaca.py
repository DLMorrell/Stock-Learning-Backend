from Tools.trading_bot import trading_bot
from Tools.get_alpaca_market_data import market_data
import csv
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    stock = 'SPY'
    file_name = f'{stock}_All_Data.csv'
    md = market_data()

    today_raw = datetime.now().isoformat()
    today = str(today_raw)[:len(str(today_raw))-7].replace(' ', 'T')+'Z'

    print(f'today: {today}') 



    
    #print(df.head())
    #df.to_csv(, index=False, )