from argparse import ArgumentParser
import json
from numpy import array
from linear import LinearRegression


def _parse_command_line() -> LinearRegression:
    parser = ArgumentParser(
        prog="estimator",
        description="estimate value for a given mileage")

    parser.add_argument("file_path",
                        help="path of the json file containing weights")

    args = parser.parse_args()

    model = LinearRegression()
    with open(args.file_path) as f:
        w = json.load(f)
        model.set_weights(w)

    return model


def main():
    try:
        model = _parse_command_line()
    except Exception as e:
        print(type(e).__name__ + ":", e)
        exit(1)

    print("====================================================")
    print("The model has been successfuly loaded.")
    print("[Type \"exit\" to terminate the program.]")
    print("====================================================")
    print("Please enter the mileage to see the estimated price,")

    while True:
        user_input = input("mileage: ")
        if user_input.lower() == "exit":
            break

        mileage = float(user_input)
        price = model.predict(array([mileage]))

        print("price:", price)


if __name__ == "__main__":
    main()
