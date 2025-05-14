# Data Dictionary — SME Risk Tracker (Japan)

This document explains the columns used in the synthetic SME loan dataset (`sme_loans_japan_dummy.csv`). 
It reflects patterns and concerns identified in BOJ and FSA reports regarding zombie firms and credit guarantee overuse.

| Column               | Description                                                    | Source Inspiration               |
|----------------------|----------------------------------------------------------------|----------------------------------|
| SME_ID               | Unique identifier for each small-to-medium enterprise          | Synthetic                        |
| Revenue_JPY          | Annual revenue in Japanese yen                                 | Modeled after TSR reports        |
| Net_Income_JPY       | Net income in yen (negative = unprofitable)                    | BOJ 2025, FSA 2024               |
| Has_Guarantee        | TRUE if loan is covered by public credit guarantee             | BOJ 2025                         |
| Interest_Rate_Pct    | Interest rate assigned to SME loan                             | FSA lending guidelines           |
| Default_Flag         | TRUE if the firm defaulted (binary placeholder)                | Modeled risk rate (3–5%)         |
| Industry             | SME’s sector (e.g., Construction, Retail, IT)                  | SME White Paper, METI            |
| Prefecture           | Firm’s location within Japan                                   | BOJ Regional Reports             |
| Years_In_Business    | Age of the company in years                                    | SME Finance Surveys              |
| Num_Employees        | Number of full-time employees                                  | Small Enterprise Agency (JSMEA)  |
| Is_Zombie            | TRUE if net income is negative for a mature firm (≥3 years)    | BOJ 2025                         |
| Reason               | Explanation of zombie classification status                    | Heuristic logic for validation   |
