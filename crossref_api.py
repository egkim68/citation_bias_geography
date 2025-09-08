import pandas as pd
import requests
import time
import sys
import os
from urllib.parse import quote

# ------------------- CONFIGURATION -------------------
# Add your email for CrossRef API good citizenship
USER_EMAIL = "your_email"  # Replace with your actual email

# Request headers for CrossRef API
HEADERS = {
    'User-Agent': f'GeographicBiasStudy/1.0 (mailto:{USER_EMAIL})',
    'Accept': 'application/json'
}

# ------------------- DOI VALIDATION FUNCTIONS ---------------------
def clean_doi(doi):
    """Clean and normalize DOI format"""
    if not doi or pd.isna(doi) or doi == '':
        return None
    
    doi_str = str(doi).strip()
    
    # Remove common prefixes
    if doi_str.startswith('https://doi.org/'):
        doi_str = doi_str.replace('https://doi.org/', '')
    elif doi_str.startswith('http://dx.doi.org/'):
        doi_str = doi_str.replace('http://dx.doi.org/', '')
    elif doi_str.startswith('doi:'):
        doi_str = doi_str.replace('doi:', '')
    
    # Basic validation - should have format like "10.xxxx/yyyy"
    if not doi_str.startswith('10.'):
        return None
    
    return doi_str

def get_country_variants(country):
    """Get search variants for country names"""
    variants = {
        'United States': ['united states', 'usa', 'us', 'america', 'american'],
        'United Kingdom': ['united kingdom', 'uk', 'britain', 'british', 'england', 'scotland', 'wales'],
        'Germany': ['germany', 'german', 'deutschland'],
        'South Korea': ['south korea', 'korea', 'korean', 'republic of korea'],
        'Australia': ['australia', 'australian'],
        'China': ['china', 'chinese', 'prc'],
        'Brazil': ['brazil', 'brazilian', 'brasil'],
        'India': ['india', 'indian'],
        'Kenya': ['kenya', 'kenyan'],
        'Bangladesh': ['bangladesh', 'bangladeshi']
    }
    
    if country in variants:
        return variants[country]
    else:
        return [country.lower()]

def check_geographic_relevance(crossref_data, target_country):
    """Check if paper is geographically relevant using text matching"""
    
    if not crossref_data:
        return False
    
    # Get country search terms
    search_terms = get_country_variants(target_country)
    
    # Combine all searchable text
    search_text = ""
    
    # Title
    if 'title' in crossref_data and crossref_data['title']:
        title = crossref_data['title'][0] if isinstance(crossref_data['title'], list) else str(crossref_data['title'])
        search_text += title.lower() + " "
    
    # Abstract
    if 'abstract' in crossref_data and crossref_data['abstract']:
        search_text += crossref_data['abstract'].lower() + " "
    
    # Subject/keywords
    if 'subject' in crossref_data and crossref_data['subject']:
        for subject in crossref_data['subject']:
            search_text += subject.lower() + " "
    
    # Check if any country variant is mentioned
    for term in search_terms:
        if term in search_text:
            return True
    
    return False

def validate_doi_crossref(doi, target_country, max_retries=2):
    """Validate DOI using CrossRef API and check geographic relevance"""
    clean_doi_str = clean_doi(doi)
    
    if not clean_doi_str:
        return {
            'doi_valid': False,
            'geographic_relevant': False,
            'crossref_title': None,
            'crossref_authors': None,
            'crossref_year': None,
            'crossref_journal': None,
            'error_message': 'Invalid DOI format'
        }
    
    for attempt in range(max_retries):
        try:
            # URL encode the DOI
            encoded_doi = quote(clean_doi_str, safe='')
            url = f"https://api.crossref.org/works/{encoded_doi}"
            
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})
                
                # Extract metadata
                title = None
                if 'title' in message and message['title']:
                    title = message['title'][0] if isinstance(message['title'], list) else str(message['title'])
                
                authors = None
                if 'author' in message and message['author']:
                    author_names = []
                    for author in message['author'][:3]:  # First 3 authors only
                        if 'given' in author and 'family' in author:
                            author_names.append(f"{author['given']} {author['family']}")
                        elif 'family' in author:
                            author_names.append(author['family'])
                    authors = '; '.join(author_names) if author_names else None
                
                year = None
                if 'published-print' in message and 'date-parts' in message['published-print']:
                    year = message['published-print']['date-parts'][0][0]
                elif 'published-online' in message and 'date-parts' in message['published-online']:
                    year = message['published-online']['date-parts'][0][0]
                elif 'created' in message and 'date-parts' in message['created']:
                    year = message['created']['date-parts'][0][0]
                
                journal = None
                if 'container-title' in message and message['container-title']:
                    journal = message['container-title'][0] if isinstance(message['container-title'], list) else str(message['container-title'])
                
                # Check geographic relevance
                geographic_relevant = check_geographic_relevance(message, target_country)
                
                return {
                    'doi_valid': True,
                    'geographic_relevant': geographic_relevant,
                    'crossref_title': title,
                    'crossref_authors': authors,
                    'crossref_year': year,
                    'crossref_journal': journal,
                    'error_message': None
                }
                
            elif response.status_code == 404:
                return {
                    'doi_valid': False,
                    'geographic_relevant': False,
                    'crossref_title': None,
                    'crossref_authors': None,
                    'crossref_year': None,
                    'crossref_journal': None,
                    'error_message': 'DOI not found in CrossRef'
                }
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {
                        'doi_valid': False,
                        'geographic_relevant': False,
                        'crossref_title': None,
                        'crossref_authors': None,
                        'crossref_year': None,
                        'crossref_journal': None,
                        'error_message': f'HTTP {response.status_code}'
                    }
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'doi_valid': False,
                    'geographic_relevant': False,
                    'crossref_title': None,
                    'crossref_authors': None,
                    'crossref_year': None,
                    'crossref_journal': None,
                    'error_message': 'Request timeout'
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'doi_valid': False,
                    'geographic_relevant': False,
                    'crossref_title': None,
                    'crossref_authors': None,
                    'crossref_year': None,
                    'crossref_journal': None,
                    'error_message': str(e)
                }
    
    return {
        'doi_valid': False,
        'geographic_relevant': False,
        'crossref_title': None,
        'crossref_authors': None,
        'crossref_year': None,
        'crossref_journal': None,
        'error_message': 'Max retries exceeded'
    }

# ------------------- MAIN VALIDATION FUNCTION ---------------------
def validate_citations_file(input_file, output_file=None, resume=True):
    """Validate DOIs in a citations CSV file"""
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return None
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = input_file.replace('.csv', '')
        output_file = f"{base_name}_validated.csv"
    
    print(f"Validating DOIs in: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Load citations
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} citations")
    
    # Add validation columns if they don't exist
    validation_columns = ['DOI_Valid', 'Geographic_Relevant', 'CrossRef_Title', 'CrossRef_Authors', 'CrossRef_Year', 'CrossRef_Journal', 'Validation_Error']
    for col in validation_columns:
        if col not in df.columns:
            df[col] = None
    
    # Resume from where we left off if requested
    start_index = 0
    if resume and os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) > 0 and 'DOI_Valid' in existing_df.columns:
                completed_rows = existing_df['DOI_Valid'].notna().sum()
                if completed_rows > 0:
                    print(f"Found existing progress: {completed_rows} DOIs already validated")
                    start_index = completed_rows
                    # Copy over existing validation results
                    for col in validation_columns:
                        if col in existing_df.columns:
                            df[col] = existing_df[col]
        except Exception as e:
            print(f"Could not read existing output file: {e}")
            print("Starting validation from the beginning")
    
    # Count DOIs to validate
    dois_to_validate = df[start_index:][df[start_index:]['DOI'].notna() & (df[start_index:]['DOI'] != '')]
    total_dois = len(dois_to_validate)
    
    print(f"Starting validation from row {start_index}")
    print(f"DOIs to validate: {total_dois}")
    
    if total_dois == 0:
        print("No DOIs to validate!")
        return df
    
    # Validate DOIs
    for i, (idx, row) in enumerate(df[start_index:].iterrows(), 1):
        if pd.isna(row['DOI']) or row['DOI'] == '':
            # Mark as invalid if no DOI provided
            df.at[idx, 'DOI_Valid'] = False
            df.at[idx, 'Geographic_Relevant'] = False
            df.at[idx, 'Validation_Error'] = 'No DOI provided'
            continue
        
        print(f"[{i}/{len(df[start_index:])}] Validating: {row['Country']} - {row['DOI'][:50]}...")
        
        # Validate DOI and check geographic relevance
        validation_result = validate_doi_crossref(row['DOI'], row['Country'])
        
        # Store results
        df.at[idx, 'DOI_Valid'] = validation_result['doi_valid']
        df.at[idx, 'Geographic_Relevant'] = validation_result['geographic_relevant']
        df.at[idx, 'CrossRef_Title'] = validation_result['crossref_title']
        df.at[idx, 'CrossRef_Authors'] = validation_result['crossref_authors']
        df.at[idx, 'CrossRef_Year'] = validation_result['crossref_year']
        df.at[idx, 'CrossRef_Journal'] = validation_result['crossref_journal']
        df.at[idx, 'Validation_Error'] = validation_result['error_message']
        
        # Save progress every 10 validations
        if i % 10 == 0 or i == len(df[start_index:]):
            df.to_csv(output_file, index=False)
            print(f"  Progress saved: {i}/{len(df[start_index:])} rows processed")
        
        # Rate limiting - be nice to CrossRef
        time.sleep(0.5)
    
    # Final save
    df.to_csv(output_file, index=False)
    print(f"Validation completed for {input_file}")
    
    return df

def validate_all_citation_files():
    """Validate DOIs in all citation CSV files from the 4 LLMs"""
    
    print("=== CrossRef DOI Validation for All LLM Models ===")
    
    # List of input files from Python citation generation
    input_files = [
        'data/raw_citations_openai.csv',
        'data/raw_citations_claude.csv', 
        'data/raw_citations_gemini.csv',
        'data/raw_citations_deepseek.csv'
    ]
    
    all_results = []
    
    # Process each file
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"\n{'='*60}")
            output_file = input_file.replace('.csv', '_validated.csv')
            result_df = validate_citations_file(input_file, output_file)
            if result_df is not None:
                all_results.append(result_df)
        else:
            print(f"Warning: File not found - {input_file}")
    
    # Combined summary across all models
    if all_results:
        print(f"\n{'='*60}")
        print("=== COMBINED SUMMARY ACROSS ALL MODELS ===")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        citations_with_dois = combined_df[combined_df['DOI'].notna() & (combined_df['DOI'] != '')]
        valid_dois = combined_df[combined_df['DOI_Valid'] == True]
        invalid_dois = citations_with_dois[citations_with_dois['DOI_Valid'] == False]
        geographically_relevant = combined_df[combined_df['Geographic_Relevant'] == True]
        
        print(f"Total citations across all models: {len(combined_df)}")
        print(f"Citations with DOIs: {len(citations_with_dois)}")
        print(f"Valid DOIs: {len(valid_dois)}")
        print(f"Invalid/Hallucinated DOIs: {len(invalid_dois)}")
        print(f"Geographically relevant papers: {len(geographically_relevant)}")
        
        if len(citations_with_dois) > 0:
            overall_hallucination = len(invalid_dois) / len(citations_with_dois) * 100
            print(f"Overall hallucination rate: {overall_hallucination:.1f}%")
            
            if len(valid_dois) > 0:
                overall_bias = (len(valid_dois) - len(geographically_relevant)) / len(valid_dois) * 100
                print(f"Overall geographic bias rate: {overall_bias:.1f}%")
        
        # By LLM model
        print(f"\n=== COMPARISON BY LLM MODEL ===")
        for llm in combined_df['LLM'].unique():
            if pd.notna(llm):
                subset = citations_with_dois[citations_with_dois['LLM'] == llm]
                if len(subset) > 0:
                    subset_invalid = subset[subset['DOI_Valid'] == False]
                    subset_valid = subset[subset['DOI_Valid'] == True]
                    subset_relevant = subset[subset['Geographic_Relevant'] == True]
                    
                    halluc_rate = len(subset_invalid) / len(subset) * 100
                    if len(subset_valid) > 0:
                        bias_rate = (len(subset_valid) - len(subset_relevant)) / len(subset_valid) * 100
                        print(f"  {llm}:")
                        print(f"    Total citations: {len(subset)}")
                        print(f"    Hallucination: {len(subset_invalid)}/{len(subset)} ({halluc_rate:.1f}%)")
                        print(f"    Geographic bias: {len(subset_valid) - len(subset_relevant)}/{len(subset_valid)} ({bias_rate:.1f}%)")
                    else:
                        print(f"  {llm}: {len(subset_invalid)}/{len(subset)} invalid ({halluc_rate:.1f}%)")
        
        # By income level  
        print(f"\n=== COMPARISON BY INCOME LEVEL ===")
        for income_level in ['High Income', 'Upper-Middle Income', 'Lower-Middle Income']:
            subset = citations_with_dois[citations_with_dois['Income_Level'] == income_level]
            if len(subset) > 0:
                subset_invalid = subset[subset['DOI_Valid'] == False]
                subset_valid = subset[subset['DOI_Valid'] == True]  
                subset_relevant = subset[subset['Geographic_Relevant'] == True]
                
                halluc_rate = len(subset_invalid) / len(subset) * 100
                if len(subset_valid) > 0:
                    bias_rate = (len(subset_valid) - len(subset_relevant)) / len(subset_valid) * 100
                    print(f"  {income_level}:")
                    print(f"    Total citations: {len(subset)}")
                    print(f"    Hallucination: {len(subset_invalid)}/{len(subset)} ({halluc_rate:.1f}%)")
                    print(f"    Geographic bias: {len(subset_valid) - len(subset_relevant)}/{len(subset_valid)} ({bias_rate:.1f}%)")
                else:
                    print(f"  {income_level}: {len(subset_invalid)}/{len(subset)} invalid ({halluc_rate:.1f}%)")
    
    print(f"\n=== ALL VALIDATIONS COMPLETED ===")
    print("Validated files ready for R analysis:")
    for input_file in input_files:
        output_file = input_file.replace('.csv', '_validated.csv')
        if os.path.exists(output_file):
            print(f" - {output_file}")

# ------------------- COMMAND LINE INTERFACE ---------------------
def main():
    if len(sys.argv) > 1:
        # If file provided, validate just that file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        validate_citations_file(input_file, output_file)
    else:
        # If no arguments, validate all files
        validate_all_citation_files()

if __name__ == "__main__":
    main()
