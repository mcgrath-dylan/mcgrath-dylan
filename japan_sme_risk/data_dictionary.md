# Data Dictionary — SME Risk Tracker (Japan)

This document explains the columns used in the synthetic SME loan dataset (`sme_loans_japan_dummy.csv`). 
It reflects patterns and concerns identified in BOJ and FSA reports regarding zombie firms and credit guarantee overuse.

| Column               | Description                                                    | Example             | Source Inspiration               |
|----------------------|----------------------------------------------------------------|---------------------|----------------------------------|
| SME_ID               | Unique identifier for each small-to-medium enterprise          | SME_001             | Synthetic                        |
| Revenue_JPY          | Annual revenue in Japanese yen                                 | 80,000,000          | Modeled after TSR reports        |
| Net_Income_JPY       | Net income in yen (negative = unprofitable)                    | -5,000,000          | BOJ 2025, FSA 2024               |
| Has_Guarantee        | TRUE if loan is covered by public credit guarantee             | TRUE                | BOJ 2025                         |
| Interest_Rate_Pct    | Interest rate assigned to SME loan                             | 1.2                 | FSA lending guidelines           |
| Default_Flag         | TRUE if the firm defaulted (binary placeholder)                | FALSE               | Modeled risk rate (3–5%)         |
| Industry             | SME’s sector (e.g., Construction, Retail, IT)                  | Retail              | SME White Paper, METI            |
| Prefecture           | Firm’s location within Japan                                   | Osaka               | BOJ Regional Reports             |
| Years_In_Business    | Age of the company in years                                    | 7                   | SME Finance Surveys              |
| Num_Employees        | Number of full-time employees                                  | 9                   | Small Enterprise Agency (JSMEA)  |
| Is_Zombie            | TRUE if net income is negative for a mature firm (≥3 years)    | TRUE                | BOJ 2025                         |
| Reason               | Explanation of zombie classification status                    | "Negative income 3+ yrs, guaranteed loan" | Heuristic logic for validation   |
