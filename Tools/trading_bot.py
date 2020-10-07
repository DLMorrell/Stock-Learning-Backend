import alpaca_trade_api as tradeapi
import os
from time import sleep
from dotenv import load_dotenv

class trading_bot():
    def __init__(self):
        self.alpaca = tradeapi.REST(os.getenv('API'), os.getenv('SECRET'), os.getenv('ENDPOINT'), 'v2')
        self.account = self.alpaca.get_account()

    #Completes transactions such as buying and selling
    def submitOrder(self, quantity, company, side): # Ex. 1, "FB", "buy"
        time = self.alpaca.get_clock()
        if quantity > 0:
          try:
            self.alpaca.submit_order(
                symbol= company,
                qty= quantity,
                side= side,
                type= 'market',
                time_in_force= 'day',
            )
            print(f'Market order of |  {quantity} {company} {side}  | completed.')
            return True
          except:
            print(f'Order of | {quantity} {company}  {side} | did not go through.')
            return False
        else:
          print(f'Quantity is <=0, order of | {quantity} {company} {side} | not sent.');
          return True

    # Sell everything
    def sellAllCompanyStocks(self, company):
        spyPosition = None
        try:
            currentData = self.alpaca.get_position('SPY')
            spyPosition = currentData["qty"]
        except:
            spyPosition = 0
        self.submitOrder(spyPosition,company,"sell")
        print(f'***** All of {company} stocks have been sold *****')
    
    #Sells all shares
    def closeAllPositions(self):
        self.alpaca.close_all_positions()
        print("***** All positions have been closed *****")
    
    def cancelAllPendingOrders(self):
        self.alpaca.cancel_all_orders()
        print("***** All pending orders have been canceled *****")