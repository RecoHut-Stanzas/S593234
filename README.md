# Basic Retail Recommender

## Project structure
```
.
├── code
│   ├── __init__.py
│   ├── metrics.py                                      # eval metrics
│   ├── recommenders.py                                 # recommender models
│   └── utils.py                                        # item pre-filtering rules
├── data
│   └── bronze
│       ├── data.zip                                    # raw csv data
│       ├── demographics.parquet.gzip                   # raw ingested data of customer demographics
│       ├── products.parquet.gzip                       # raw ingested data of product details
│       ├── test.parquet.gzip                           # raw ingested test interactions
│       ├── train.parquet.gzip                          # raw ingested train interactions
│       └── transactions.parquet.gzip                   # raw ingested transaction data
├── docs
├── LICENSE
├── model
├── notebooks
│   ├── reco-tut-brr-01-data-ingestion.ipynb            # notebook for data ingestion 
│   ├── reco-tut-brr-02-simple-recommender-models.ipynb # simple recommender model notebook
│   ├── reco-tut-brr-03-als-model.ipynb                 # ALS model notebook
│   ├── reco-tut-brr-04-lightfm-hybrid-model.ipynb      # LightFM Hybrid model notebook
│   ├── reco-tut-brr-05-two-stage-recommender.ipynb     # ALS-like model for candidate selection, LightFM-like model for ranking
│   └── reco-tut-brr-99-01-basics.ipynb                 # some basics
└── README.md
```