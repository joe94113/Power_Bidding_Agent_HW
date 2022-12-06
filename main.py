#################################################################################################################################
# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return
# You should not modify this part.
#################################################################################################################################

if __name__ == "__main__":
    from datetime import datetime, timedelta
    from trade import Trade
    import pandas as pd

    args = config()

    df_bid = pd.read_csv(args.bidresult)
    df_con = pd.read_csv(args.consumption)
    df_gen = pd.read_csv(args.generation)

    con_data = df_con.loc[:, 'consumption'].tolist()
    gen_data = df_gen.loc[:, 'generation'].tolist()

    trade = Trade()

    model_con, model_gen = trade.get_model()
    pre_con = model_con.predict([con_data])[0]
    pre_gen = model_gen.predict([gen_data])[0]

    diff = pre_con - pre_gen

    data = []
    current_time = datetime.strptime(df_con.loc[167, 'time'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=1)
    for need in diff:
        # print(round(need))
        if round(need) > 0:
            data.append([datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S'), 'buy', trade.buy_price, round(need)])
        elif round(need) < 0:
            data.append([datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S'), 'sell', trade.sell_price, abs(round(need))])
        current_time = current_time + timedelta(hours=1)
    output(args.output, data)
