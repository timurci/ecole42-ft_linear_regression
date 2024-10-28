from argparse import ArgumentParser
import json
from linear import LinearRegression
from pandas import read_csv, DataFrame, melt
from numpy import set_printoptions
import seaborn as sns
from matplotlib import pyplot as plt


def _parse_command_line() -> tuple[DataFrame, LinearRegression]:
    parser = ArgumentParser(
        prog="analyzer",
        description="score and visualize the linear model to a given data")

    parser.add_argument("data_file",
                        help="path of the csv file containing raw data")
    parser.add_argument("weights_file",
                        help="path of the json file containing weights")

    args = parser.parse_args()

    data = read_csv(args.data_file)

    model = LinearRegression()
    with open(args.weights_file) as f:
        w = json.load(f)
        model.set_weights(w)

    return (data, model)


def _plot_data(model, X, y, column_inx, column_name):
    table = DataFrame({
        "prediction": model.predict(X),
        "observation": y,
        column_name: X[:, column_inx]
    })

    table = melt(table,
                 id_vars=[column_name],
                 value_vars=["prediction", "observation"])

    pred = table.loc[table["variable"] == "prediction"]

    pred_min = pred.loc[pred[column_name] == pred[column_name].min()]
    pred_max = pred.loc[pred[column_name] == pred[column_name].max()]

    X_minmax = [pred_min[column_name], pred_max[column_name]]
    y_minmax = [pred_min["value"], pred_max["value"]]

    sns.set_theme()
    sns.relplot(
        data=table,
        x=column_name,
        y="value",
        hue="variable")

    plt.plot(X_minmax, y_minmax)
    plt.show()


def main():
    try:
        data, model = _parse_command_line()
    except Exception as e:
        print(type(e).__name__ + ":", e)
        exit(1)

    print("====================================================")
    print("The data and the model have been successfuly loaded.")
    print("====================================================")
    print("Observation Statistics", data["price"].describe(), sep="\n")
    print("====================================================")

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    mse = model.mse(X, y)

    set_printoptions(precision=2, suppress=True)
    print("{:15}".format("Weights"), ":", model)
    print("{:15}".format("MSE"), ":", "{:.2f}".format(mse))
    print("{:15}".format("SQRT MSE"), ":", "{:.2f}".format(mse ** .5))

    _plot_data(model, X, y, 0, "mileage")


if __name__ == "__main__":
    main()
