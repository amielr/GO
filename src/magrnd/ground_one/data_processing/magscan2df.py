import pandas as pd


def convert_magscan_to_df(scan):
    df = pd.DataFrame({"x": scan.x, "y": scan.y, "B": scan.b, "height": scan.a,
                       "time": scan.time})

    if hasattr(scan, "b_before_subtraction") and len(scan.b_before_subtraction) > 0:
        df["original B"] = scan.b_before_subtraction

    return df
