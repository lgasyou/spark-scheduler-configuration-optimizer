import pandas as pd


def list2excel(arr: list, filename: str):
    writer = pd.ExcelWriter(filename)
    pd.DataFrame(arr).to_excel(writer)
    writer.save()
