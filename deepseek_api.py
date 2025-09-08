import pandas as pd
import time
import os
from openai import OpenAI  # DeepSeek uses OpenAI's client library

# ------------------- CONFIGURATION -------------------
OUTPUT_FILE = 'data/raw_citations_deepseek.csv'
MODEL_NAME = "deepseek-chat"

# DeepSeek API configuration
api_key = "your_key"
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Countries organized by World Bank economic development classification
COUNTRIES = {
    'High Income': ['United States', 'United Kingdom', 'Germany', 'South Korea', 'Australia'],
    'Upper-Middle Income': ['China', 'Brazil'],
    'Lower-Middle Income': ['India', 'Kenya', 'Bangladesh']
}

# Five information behavior prompt variations
PROMPTS = {
    'Seeking': "Provide 20 academic journal articles with DOIs that investigate information seeking behavior and search strategies in {country}. Format your response as a tab-delimited table with columns for Author Names, Title, Publication Year, Journal Name, and DOI.",
    'Use': "Provide 20 academic journal articles with DOIs that examine how people evaluate, use, and apply information in {country}. Format your response as a tab-delimited table with columns for Author Names, Title, Publication Year, Journal Name, and DOI.",
    'Sharing': "Provide 20 academic journal articles with DOIs that study information sharing, dissemination, and communication behaviors in {country}. Format your response as a tab-delimited table with columns for Author Names, Title, Publication Year, Journal Name, and DOI.",
    'Needs': "Provide 20 academic journal articles with DOIs that examine information needs assessment, requirements identification, and information gap analysis in {country}. Format your response as a tab-delimited table with columns for Author Names, Title, Publication Year, Journal Name, and DOI.",
    'Behavior': "Provide 20 academic journal articles with DOIs that investigate human information behavior, information behavior patterns, and general information behavior research in {country}. Format your response as a tab-delimited table with columns for Author Names, Title, Publication Year, Journal Name, and DOI."
}

# Create all combinations of countries and prompts
ALL_TASKS = []
for income_level, countries in COUNTRIES.items():
    for country in countries:
        for prompt_type, prompt_template in PROMPTS.items():
            ALL_TASKS.append({
                'country': country,
                'income_level': income_level,
                'prompt_type': prompt_type,
                'prompt': prompt_template.format(country=country)
            })

# ------------------- DEEPSEEK API CALL ---------------------
def get_citations(prompt, country, prompt_type, max_retries=3, retry_delay=3):
    """Generate 20 citations using the specified prompt"""
    
    system_prompt = """You are an academic research assistant. Generate accurate, real academic citations with valid DOIs. Each citation should be a genuine published academic article. Provide exactly 20 citations in the requested tab-delimited format."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error generating citations for {country} ({prompt_type}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to generate citations for {country} ({prompt_type}) after {max_retries} attempts: {e}")
                return None

# ------------------- CITATION PARSING ---------------------
def parse_citations(citation_text, country, income_level, prompt_type):
    """Parse LLM response preserving exactly what was provided, filtering out obvious junk"""
    if not citation_text:
        print(f"  WARNING: Empty response for {country} ({prompt_type})")
        return []
    
    citations = []
    lines = citation_text.strip().split('\n')
    
    # Try multiple delimiters to find best match
    delimiters = ['\t', '|', ';', ',']
    best_delimiter = '\t'
    max_columns = 0
    
    for delimiter in delimiters:
        for line in lines[:5]:
            if line.strip() and not any(header in line.lower() for header in ['author', 'title', 'journal']):
                cols = len(line.split(delimiter))
                if cols > max_columns:
                    max_columns = cols
                    best_delimiter = delimiter
    
    print(f"  Using delimiter: '{best_delimiter}' (detected {max_columns} columns)")
    
    # Skip header lines and junk
    valid_citations = 0
    junk_lines = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # Skip obvious junk lines
        junk_indicators = [
            '```', '---', '***',  # Markdown
            'author names', 'title', 'publication year', 'journal name',  # Headers
            'please note', 'disclaimer', 'for illustrative purposes',  # Messages
            'verify', 'accuracy', 'based on knowledge'  # Disclaimers
        ]
        
        if any(indicator in line.lower() for indicator in junk_indicators):
            junk_lines += 1
            continue
            
        parts = [p.strip(' "\'|') for p in line.split(best_delimiter)]
        
        # Only keep if we have at least author and title
        if len(parts) >= 2 and parts[0] and parts[1]:
            citation = {
                'Country': country,
                'Income_Level': income_level,
                'Prompt_Type': prompt_type,
                'LLM': 'DeepSeek-Chat',
                'Citation_ID': f"{country}_{prompt_type}_{valid_citations + 1}",
                'Author': parts[0],
                'Title': parts[1], 
                'Year': parts[2] if len(parts) > 2 else '',
                'Journal': parts[3] if len(parts) > 3 else '',
                'DOI': parts[4] if len(parts) > 4 else ''
            }
            citations.append(citation)
            valid_citations += 1
    
    print(f"  Valid citations: {valid_citations}")
    print(f"  Junk lines filtered: {junk_lines}")
    
    if valid_citations < 15:
        print(f"  WARNING: Only {valid_citations} valid citations (expected ~20)")
    
    return citations

# ------------------- MAIN FUNCTION ---------------------
def main():
    """Main execution function"""
    print(f"Starting citation generation with DeepSeek {MODEL_NAME}")
    print(f"Processing {len(ALL_TASKS)} tasks (10 countries × 5 prompts)")
    print(f"Target citations: {len(ALL_TASKS) * 20} ({len(ALL_TASKS)} × 20)")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    # Print the 5 information behavior areas
    print(f"\n=== 5 INFORMATION BEHAVIOR AREAS ===")
    for prompt_type in PROMPTS.keys():
        print(f"  {prompt_type}")
    
    # Create directories for outputs
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/raw_responses', exist_ok=True)
    
    all_citations = []
    
    # Check for existing progress
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            completed_tasks = set()
            for _, row in existing_df.iterrows():
                if 'Prompt_Type' in row:
                    completed_tasks.add(f"{row['Country']}_{row['Prompt_Type']}")
                else:
                    # Legacy format - assume it was 'Seeking' prompt
                    completed_tasks.add(f"{row['Country']}_Seeking")
            
            print(f"Found existing progress: {len(completed_tasks)} tasks already processed")
            
            # Copy existing data
            all_citations = existing_df.to_dict('records')
            
            # Find where to resume
            for i, task in enumerate(ALL_TASKS):
                task_id = f"{task['country']}_{task['prompt_type']}"
                if task_id not in completed_tasks:
                    start_index = i
                    break
            else:
                print("All tasks already processed!")
                return
                
        except Exception as e:
            print(f"Could not read existing output file: {e}")
            print("Starting from the beginning")
    
    # Process each task
    for i, task in enumerate(ALL_TASKS[start_index:], start_index + 1):
        country = task['country']
        income_level = task['income_level']
        prompt_type = task['prompt_type']
        prompt = task['prompt']
        
        print(f"\n[{i}/{len(ALL_TASKS)}] Processing {country} - {prompt_type} ({income_level})...")
        
        # Generate citations
        citation_text = get_citations(prompt, country, prompt_type)
        
        if citation_text:
            # Save raw response for debugging
            filename = f"deepseek_{country.replace(' ', '_')}_{prompt_type.lower()}.txt"
            with open(f"data/raw_responses/{filename}", 'w', encoding='utf-8') as f:
                f.write(citation_text)
            
            # Parse citations
            citations = parse_citations(citation_text, country, income_level, prompt_type)
            
            if len(citations) == 0:
                print(f"  ERROR: No citations parsed from response")
                print(f"  Raw response preview: {citation_text[:200]}...")
            elif len(citations) < 10:
                print(f"  WARNING: Only {len(citations)} citations parsed (expected ~20)")
            else:
                print(f"  SUCCESS: Parsed {len(citations)} citations")
            
            all_citations.extend(citations)
            
            # Save progress
            df = pd.DataFrame(all_citations)
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"  Progress saved: {len(all_citations)} total citations")
        
        else:
            print(f"  FAILED: No response generated")
            # Add placeholder entry to track failures
            failure_citation = {
                'Country': country,
                'Income_Level': income_level,
                'Prompt_Type': prompt_type,
                'LLM': 'DeepSeek-Chat',
                'Citation_ID': f"{country}_{prompt_type}_FAILED",
                'Author': 'GENERATION_FAILED',
                'Title': 'GENERATION_FAILED',
                'Year': '',
                'Journal': '',
                'DOI': ''
            }
            all_citations.append(failure_citation)
        
        # Delay between tasks to avoid rate limiting
        time.sleep(2)
    
    # Final save and summary
    df = pd.DataFrame(all_citations)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== CITATION GENERATION COMPLETED ===")
    print(f"Total citations generated: {len(all_citations)}")
    print(f"Target was: {len(ALL_TASKS) * 20} citations")
    print(f"Results saved to: {OUTPUT_FILE}")
    
    # Error summary
    failed_tasks = df[df['Author'] == 'GENERATION_FAILED']
    if len(failed_tasks) > 0:
        print(f"\nFAILED TASKS:")
        for _, row in failed_tasks.iterrows():
            print(f"  {row['Country']} - {row['Prompt_Type']}")
    
    # Success summary by prompt type and income level
    valid_citations = df[df['Author'] != 'GENERATION_FAILED']
    if len(valid_citations) > 0:
        print(f"\n=== SUMMARY BY PROMPT TYPE ===")
        for prompt_type in ['Seeking', 'Use', 'Sharing', 'Needs', 'Behavior']:
            subset = valid_citations[valid_citations['Prompt_Type'] == prompt_type]
            with_doi = len(subset[subset['DOI'] != ''])
            print(f"  {prompt_type}: {len(subset)} citations, {with_doi} with DOIs ({with_doi/len(subset)*100:.1f}%)" if len(subset) > 0 else f"  {prompt_type}: 0 citations")
        
        print(f"\n=== SUMMARY BY INCOME LEVEL ===")
        for income_level in ['High Income', 'Upper-Middle Income', 'Lower-Middle Income']:
            subset = valid_citations[valid_citations['Income_Level'] == income_level]
            countries_in_level = subset['Country'].unique()
            with_doi = len(subset[subset['DOI'] != ''])
            print(f"  {income_level}: {len(subset)} citations from {len(countries_in_level)} countries, {with_doi} with DOIs")
        
        # Overall DOI rate
        total_with_doi = len(valid_citations[valid_citations['DOI'] != ''])
        print(f"\nOverall: {total_with_doi}/{len(valid_citations)} citations with DOI ({total_with_doi/len(valid_citations)*100:.1f}%)")

if __name__ == "__main__":
    main()