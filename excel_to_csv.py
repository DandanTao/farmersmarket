import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Excel File to convert")
parser.add_argument('excel_file', type=str, help="Excel file to convert")

args = parser.parse_args()
excel_file = args.excel_file

try:
    xls = pd.read_excel(excel_file)
    xls.dropna()
    filename = excel_file[:excel_file.rfind(".")]
    xls.to_csv(filename+".csv", index=False)
except Exception as inst:
    print(type(inst))
    print(inst)
