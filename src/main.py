from data import load_results, process_results
def main():
    df_raw = load_results()
    df = process_results(df_raw)


if __name__ == "__main__":
    main()
