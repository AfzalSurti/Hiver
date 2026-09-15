# Data Exploration Summary

- Total rows: 2,811,774
- Unique authors: 702,777
- Customer (inbound) messages: 1,537,843
- Brand (outbound) messages: 1,273,931
- Duplicate text rows: 29,156
- Date range: 2008-05-08 20:13:59+00:00 to 2017-12-03 23:14:01+00:00

## Top brands by agent-side (outbound) message volume

- AmazonHelp: 169,840
- AppleSupport: 106,860
- Uber_Support: 56,270
- SpotifyCares: 43,265
- Delta: 42,253
- Tesco: 38,573
- AmericanAir: 36,764
- TMobileHelp: 34,317
- comcastcares: 33,031
- British_Airways: 29,361
- SouthwestAir: 28,977
- VirginTrains: 27,817
- Ask_Spectrum: 25,860
- XboxSupport: 24,557
- sprintcare: 22,381

## Candidate brands: conversation structure (sampled up to 20k root threads)

- **AmazonHelp**: n_sampled=20000, pct_with_brand_reply=0.996, avg_chain_len=3.87, max_chain_len=21
- **AmericanAir**: n_sampled=20000, pct_with_brand_reply=0.983, avg_chain_len=2.95, max_chain_len=21
- **AppleSupport**: n_sampled=20000, pct_with_brand_reply=0.993, avg_chain_len=3.04, max_chain_len=21
- **British_Airways**: n_sampled=16222, pct_with_brand_reply=0.99, avg_chain_len=2.9, max_chain_len=21
- **Delta**: n_sampled=20000, pct_with_brand_reply=0.973, avg_chain_len=2.76, max_chain_len=21
- **SouthwestAir**: n_sampled=20000, pct_with_brand_reply=0.98, avg_chain_len=2.64, max_chain_len=21
- **SpotifyCares**: n_sampled=13476, pct_with_brand_reply=0.998, avg_chain_len=2.93, max_chain_len=19
- **TMobileHelp**: n_sampled=7823, pct_with_brand_reply=0.995, avg_chain_len=2.53, max_chain_len=19