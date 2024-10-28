from argparse import ArgumentParser, Namespace
from pandas import read_csv, DataFrame  # , melt
from numpy import set_printoptions
import json
# import seaborn as sns
# from matplotlib import pyplot as plt
from linear import LinearRegression


def _parse_command_line() -> tuple[DataFrame, Namespace]:
    parser = ArgumentParser(
        prog="regressor",
        description="run gradient descent algorithm",
        epilog="target variable is expected to be in last column")

    parser.add_argument("file_path",
                        help="path of the csv file to be processed")
    parser.add_argument("-a", "--alpha", type=float, default=.1,
                        help="learning rate")
    parser.add_argument("-i", "--iteration", type=int, default=1000,
                        help="number of iterations in gradient descent")
    parser.add_argument("-n", "--no-scaling", action="store_true",
                        help="disable standard scaling in gradient descent")

    args = parser.parse_args()
    data = read_csv(args.file_path)
    return (data, args)


def main():
    try:
        data, args = _parse_command_line()
    except Exception as e:
        print(type(e).__name__ + ":", e)
        exit(1)

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    model = LinearRegression()

    model.fit(X, y,
              args.alpha,
              args.iteration,
              (not args.no_scaling))

    set_printoptions(precision=4, suppress=True)
    print("weights", model)

    with open("weights.json", 'w') as f:
        w = model.weights()

        if w is None:
            raise AssertionError("the model is not trained")

        json.dump(w.tolist(), f)
        print("The results are written into", "\"" + f.name + "\"")

    #  =====================================

    mse = model.mse(X, y)
    print("PREDICTION", model.predict(X))
    print("MSE", mse)
    print("SQRT_MSE", mse ** .5)

    result = DataFrame({
        "pred": model.predict(X),
        "obs": y,
        "features": X[:, 0]
    })

    print(result)
    # melted = melt(result, id_vars=["features"], value_vars=["pred", "obs"])

    # sns.set_theme()
    # sns.relplot(
    #     data=melted,
    #     x="features",
    #     y="value",
    #     hue="variable"
    # )
    # plt.show()


if __name__ == "__main__":
    main()
