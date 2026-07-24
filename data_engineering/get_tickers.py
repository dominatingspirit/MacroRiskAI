import re
import time
import pandas as pd
from googlesearch import search

# -------------------------------
# Read your companies.csv
# -------------------------------

companies = pd.read_csv("companies.csv")

output = []

# -------------------------------
# Find ticker
# -------------------------------

for _, row in companies.iterrows():

    company = row["Company"]
    sector = row["Sector"]

    print(f"Searching: {company}")

    ticker = ""

    try:

        query = f"site:screener.in {company}"

        results = list(search(query, num_results=5))

        screener_url = None

        for url in results:

            if "screener.in/company/" in url:

                screener_url = url
                break

        if screener_url:

            match = re.search(r"/company/([^/]+)/", screener_url)

            if match:

                ticker = match.group(1)

        print("Ticker:", ticker)

    except Exception as e:

        print(e)

    output.append({

        "Company": company,
        "Ticker": ticker,
        "Sector": sector

    })

    time.sleep(1)

# -------------------------------
# Save
# -------------------------------

pd.DataFrame(output).to_csv(

    "companies_with_tickers.csv",

    index=False

)

print("\nDONE!")