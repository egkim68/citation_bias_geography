# LLM Geographic Bias in Citation Generation

## Overview

This repository contains data and analysis code examining geographic bias in citation generation across four major LLMs tested on 10 countries with different income levels.

## Dataset

**Models**: GPT-4o-mini, Claude-3-haiku, Gemini-2.0-flash-lite, DeepSeek V3  
**Countries**: US, UK, Germany, South Korea, Australia (High Income); China, Brazil (Upper-Middle); India, Kenya, Bangladesh (Lower-Middle)  
**Categories**: 5 information behavior prompt types (Behavior, Needs, Seeking, Sharing, Use)

## Files

### Raw Data
- `raw_citations_[model].csv` - Unvalidated citations from each LLM
- `raw_citations_[model]_validated.csv` - DOI-validated citations from each LLM

### Analysis
- `table1_statistical_analysis.csv` - Statistical test results and effect sizes
- `complete_analysis.R` - Full R analysis script

## Key Variables

- `Country`, `Income_Level`, `LLM`, `Prompt_Type` - Experimental conditions
- `Author`, `Title`, `Journal`, `Year`, `DOI` - Citation metadata  
- `DOI_Valid`, `Geographic_Relevant` - Validation results

## Key Findings

- **DOI hallucination rates**: Lower-middle-income countries (70-85%) vs High-income (45-65%)
- **Geographic bias**: Systematic bias favoring Western/high-income countries
- **Model differences**: Significant variation between LLM performance

## Usage

```r
# Required packages
library(dplyr, ggplot2, readr, tidyr, stringr, broom, purrr)

# Run analysis
source("complete_analysis.R")
```

## Methods

Chi-square tests, Fisher's exact tests, Cramer's V effect sizes, Kruskal-Wallis tests, Spearman correlations.

## Citation

[To be added upon publication]

---

This dataset examines AI bias in academic citation generation for research and improvement purposes.
