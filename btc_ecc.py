import ccxt
import time

class BtcEcc(object):
    @staticmethod
    def get_exchanges():
        return ccxt.exchanges

    @staticmethod
    def get_exchange(exchange_id, apiKey, secret):
        #exchange_id = 'huobi'
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': apiKey,
            'secret': secret,
            'timeout': 30000,
            'enableRateLimit': True,
        })
        return exchange

    @staticmethod
    def get_order_books(exchange, symbol, limit):
        return exchange.fetch_order_book(symbol, limit)

    @staticmethod
    def get_best_bid(order_books):
        bid_item = order_books['bids'][0]
        return bid_item[0], bid_item[1]

    @staticmethod 
    def get_best_ask(order_books):
        ask_item = order_books['asks'][0]
        return ask_item[0], ask_item[1]

    @staticmethod
    def get_tickers(exchange, symbol):
        '''
        获取指定币对的价格信息
        '''
        for i in range(5):
            ticker = exchange.fetch_ticker(symbol)
            print('price: {0}!'.format(ticker))
            time.sleep(0.1)


def test1():
    bte = 'BTC/USDT'
    etb = 'USDT/BTC'
    symbol = bte
    exchange = BtcEcc.get_exchange('huobipro', '356cda29-253918bb-573a5afa-eeeae', 'a80d3cfd-4db6c763-0b73daee-cba31')
    print(exchange)
    markets = exchange.load_markets()
    # print('type:{0} {1}!'.format(type(markets), markets))
    limit = 10
    #obs = exchange.fetch_order_book('ETH/BTC', limit)
    order_books = BtcEcc.get_order_books(exchange, symbol, limit)
    bid_price, bid_amount = BtcEcc.get_best_bid(order_books)
    ask_price, ask_amount = BtcEcc.get_best_ask(order_books)
    print('bids:{0} --- {1}!'.format(bid_price, bid_amount))
    print('asks:{0} --- {1}!'.format(ask_price, ask_amount))
    print('delta:{0}!'.format(bid_price-ask_price))
    '''
    if exchange.has['fetchTicker']:
        print(exchange.fetch_ticker('ETH/BTC'))
    if exchange.has['fetchTrades']:
        print(exchange.fetch_trades('ETH/BTC'))
    '''
    #print('balance:{0}!'.format(exchange.fetch_balance()))
    # sell BTC and buy USDT
    #sell_btc_order = exchange.create_limit_sell_order(symbol, 0.001, ask_price)
    #print('sell order: {0}!'.format(sell_btc_order))
    buy_btc_order = exchange.create_limit_buy_order(symbol, 0.001, bid_price)
    print('buy_order: {0}!'.format(buy_btc_order))

def test2():
    bte = 'BTC/USDT'
    symbol = bte
    exchange = BtcEcc.get_exchange('huobipro', '356cda29-253918bb-573a5afa-eeeae', 'a80d3cfd-4db6c763-0b73daee-cba31')
    BtcEcc.get_tickers(exchange, symbol)


def main():
    #test1()
    test2()
    with open('a1.txt', 'a', encoding='utf8') as fd:
        fd.write('msg5:Hello world\r\n')
    print('bye!')

if '__main__' == __name__:
    main()
