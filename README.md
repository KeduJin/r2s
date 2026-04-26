## Notes

DomainRecomb:
- 01: /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/2025Y_12M_08D_22h-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter/IntervalCheckpoints/step=150000_date-12-09/pytorch_model.bin

## 0108 generation

data: 
- 1022: the earliest data
- 1103: Remove the domain which only match with one domain in 1022 version
- 0108: Train the constrative learning model with only 2 domain protein
- 0113: Train the constrative learning model with only 2 domain protein, do not apply seq id filter, unique query
- 0113_2: do not apply seq id filter, unique query

### Summary 
| data | model | Type | Size | Sample Temperature | weighting | Notes | Trained steps | plddt | 
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1022 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 57.73 |
| 1103 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 56.81 |
| 0108 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 61.09 |
| 0113 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 73.79 |
| 0113_2 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 75.37 |
| 0129 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1 | ED | 100M | 1.0 | weighting5_1| | 200K | 73.76 |


### 0108
| data | model | Type | Size | Sample Temperature | weighting | Notes | Trained steps | plddt | 
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0108 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1                                      | ED | 100M | 1.0 | weighting5_1| | 200K | 61.09 |
| 0108 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter                         | ED | 100M | 1.0 | weighting1_1| plddt_filter| 200K | 57.45 |
| 0108 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter (t=0.8)                 | ED | 100M | 0.8 | weighting1_1| plddt_filter| 200K | 61.18 |
| 0108 | [2025Y_12M_19D_17h] qwen-esm2_650M-qwen3ca_600M-weighting1_1-perturb_domain-plddt_filter (t=0.8) | ED | 1.2B | 0.8 | weighting1_1| plddt_filter| 100K | 62.45 |

### 0113
| data | model | Type | Size | Sample Temperature | weighting | Notes | trained steps | plddt | 
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0113 | [2026Y_01M_14D_11h] qwen-esm2_35M-qwen3ca_100M-weighting1_1                                      | ED | 100M | 1.0 | weighting1_1| | 200K | 68.40 |
| 0113 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1                                      | ED | 100M | 1.0 | weighting5_1| | 200K | 73.79 |
| 0113 | [2025Y_11M_10D_16h] qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain                      | ED | 1.2B | 1.0 | weighting5_1| | 100K | 74.66 |
| 0113 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter                         | ED | 100M | 1.0 | weighting1_1| plddt_filter| 200K | 71.47 |
| 0113 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter (t=0.8)                 | ED | 100M | 0.8 | weighting1_1| plddt_filter| 200K | 73.08 |
| 0113 | [2025Y_12M_19D_17h] qwen-esm2_650M-qwen3ca_600M-weighting1_1-perturb_domain-plddt_filter (t=0.8) | ED | 1.2B | 0.8 | weighting1_1| plddt_filter| 100K | 74.54 |

### 0113_2
| data | model | Type | Size | Sample Temperature | weighting | Notes | Trained steps | plddt | 
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0113_2 | [2026Y_01M_14D_11h] qwen-esm2_35M-qwen3ca_100M-weighting1_1                                      | ED | 100M | 1.0 | weighting1_1| | 200K | 70.41 |
| 0113_2 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1                                      | ED | 100M | 1.0 | weighting5_1| | 200K | 75.37 |
| 0113_2 | [2025Y_11M_10D_16h] qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain                      | ED | 1.2B | 1.0 | weighting5_1| | 100K | 76.70 |
| 0113_2 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter                         | ED | 100M | 1.0 | weighting1_1| plddt_filter| 200K | 73.50 |
| 0113_2 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter (t=0.8)                 | ED | 100M | 0.8 | weighting1_1| plddt_filter| 200K | 75.56 |
| 0113_2 | [2025Y_12M_19D_17h] qwen-esm2_650M-qwen3ca_600M-weighting1_1-perturb_domain-plddt_filter (t=0.8) | ED | 1.2B | 0.8 | weighting1_1| plddt_filter| 100K | 76.49 |

### 0129
| data | model | Type | Size | Sample Temperature | weighting | Notes | Trained steps | plddt | 
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0129 | [2026Y_01M_14D_11h] qwen-esm2_35M-qwen3ca_100M-weighting1_1                                      | ED | 100M | 1.0 | weighting1_1| | 200K | 69.13 |
| 0129 | [2025Y_10M_16D_23h] qwen-esm2_35M-qwen3ca_100M-weighting5_1                                      | ED | 100M | 1.0 | weighting5_1| | 200K | 73.76 |
| 0129 | [2025Y_11M_10D_16h] qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain                      | ED | 1.2B | 1.0 | weighting5_1| | 100K |  74.55 |
| 0129 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter                         | ED | 100M | 1.0 | weighting1_1| plddt_filter| 200K | 71.18 |
| 0129 | [2025Y_12M_28D_17h] qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter (t=0.8)                 | ED | 100M | 0.8 | weighting1_1| plddt_filter| 200K | 73.50 |
| 0129 | [2025Y_12M_19D_17h] qwen-esm2_650M-qwen3ca_600M-weighting1_1-perturb_domain-plddt_filter         | ED | 1.2B | 1.0 | weighting1_1| plddt_filter| 100K | 73.45  |
| 0129 | [2025Y_12M_19D_17h] qwen-esm2_650M-qwen3ca_600M-weighting1_1-perturb_domain-plddt_filter (t=0.8) | ED | 1.2B | 0.8 | weighting1_1| plddt_filter| 100K | 75.40  |
| 0129 | [2026Y_01M_25D_15h] rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 100M | 1 | weighting1_1| plddt_filter| 200K | 73.83  |
| 0129 | [2026Y_01M_25D_15h] rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 100M | 0.8 | weighting1_1| plddt_filter| 200K | 74.15  |
| 0129 | [2026Y_01M_28D_03h] rag-qwen-qwen3_600M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 1.2B | 1 | weighting1_1| plddt_filter| 50K | 73.67  |
| 0129 | [2026Y_01M_28D_03h] rag-qwen-qwen3_600M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 1.2B | 0.8 | weighting1_1| plddt_filter| 50K | 74.32  |
| 0129 | [2026Y_01M_28D_03h] rag-qwen-qwen3_600M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 1.2B | 1 | weighting1_1| plddt_filter| 100K | 73.81  |
| 0129 | [2026Y_01M_28D_03h] rag-qwen-qwen3_600M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 1.2B | 0.8 | weighting1_1| plddt_filter| 100K | 74.23  |
<!-- | 0129 | [2026Y_01M_25D_15h] rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting | DecoderOnly | 100M | 1 | weighting1_1| plddt_filter| 50K | 73.67  | -->
