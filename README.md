# LLM Geographic Bias in Citation Generation

## Overview
This repository contains data, analysis code, and API scripts examining geographic bias in citation generation across four major LLMs tested on 10 countries with different income levels. It includes code for generating citations with LLM APIs (including DOIs) and for validating DOIs using the CrossRef REST API.

## Dataset
**Models**: GPT-4o-mini, Claude-3-haiku, Gemini-2.0-flash-lite, DeepSeek V3  
**Countries**: US, UK, Germany, South Korea, Australia (High Income); China, Brazil (Upper-Middle); India, Kenya, Bangladesh (Lower-Middle)  
**Categories**: 5 information behavior prompt types (Behavior, Needs, Seeking, Sharing, Use)

## Files
- `raw_citations_[model].csv` – Unvalidated citations from each LLM  
- `raw_citations_[model]_validated.csv` – DOI-validated citations from each LLM  
- `table2_statistical_analysis.csv` – Statistical test results and effect sizes  
- `table3_citation_yield_summary.csv` – Citation yield summary (LLM Model, Conditions, Avg Citations per Condition, Response Rate (%), Conditions with Full 20 Citations)   
- `[model]_api.py` – API scripts for LLM citation generation (with DOIs)  
- `crossref_api.py` – CrossRef API script for DOI normalization and validation  

## Key Variables
- `Country`, `Income_Level`, `LLM`, `Prompt_Type` – Experimental conditions  
- `Author`, `Title`, `Journal`, `Year`, `DOI` – Citation metadata  
- `DOI_Valid`, `Geographic_Relevant` – Validation results  

## Key Findings
- **DOI hallucination rates**: Lower-middle-income countries (70–85%) vs High-income (45–65%)  
- **Geographic bias**: Systematic bias favoring Western/high-income countries  
- **Model differences**: Significant variation between LLM performance  

## Methods
Chi-square tests, Fisher’s exact tests, Cramer’s V effect sizes, Kruskal–Wallis tests, and Spearman correlations.
