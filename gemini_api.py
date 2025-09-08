import pandas as pd
import time
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel

# ------------------- CONFIGURATION -------------------
OUTPUT_FILE = 'data/raw_citations_gemini.csv'
MODEL_NAME = "gemini-2.0-flash-lite"

# Google Gemini API configuration
api_key = "your_key"
genai.configure(api_key=api_key)
model = GenerativeModel(MODEL_NAME)

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

# ------------------- VALIDATION FUNCTIONS ---------------------
def validate_response(citation_text, country, prompt_type):
    """Validate response quality before parsing"""
    if not citation_text:
        return False, "Empty response"
    
    lines = [line.strip() for line in citation_text.split('\n') if line.strip()]
    
    # Check if we have reasonable number of lines
    if len(lines) < 15:
        return False, f"Too few lines: {len(lines)}"
    
    # Check tab-delimited format
    tab_lines = [line for line in lines if '\t' in line and len(line.split('\t')) >= 4]
    if len(tab_lines) < 10:
        return False, f"Too few properly formatted lines: {len(tab_lines)}"
    
    # Check for common junk patterns
    junk_patterns = ['sorry', 'cannot provide', 'please note', 'disclaimer', 'i apologize', 'i cannot', 'based on my knowledge']
    if any(pattern in citation_text.lower() for pattern in junk_patterns):
        return False, "Response contains disclaimers/apologies"
    
    return True, "Valid"

def create_gemini_prompt(base_prompt):
    """Create Gemini-specific prompt with strict formatting instructions"""
    return f"""{base_prompt}

STRICT OUTPUT FORMAT REQUIREMENTS:
- Return ONLY the table data, no headers, no explanations, no markdown
- Each row must have exactly 5 columns separated by tabs
- Format: [Author Names][TAB][Title][TAB][Year][TAB][Journal][TAB][DOI]
- Do NOT include column headers
- Do NOT add explanatory text before or after the table
- Do NOT use markdown formatting (no |, *, -, etc.)
- Start your response immediately with the first citation
- Provide exactly 20 citations

Example format:
Smith, J. & Jones, M.	Information seeking in digital libraries	2019	Journal of Information Science	10.1177/0165551234567890"""

# ------------------- GEMINI API CALL ---------------------
def get_citations(prompt, country, prompt_type, max_retries=3, retry_delay=3):
    """Generate 20 citations using the specified prompt with enhanced validation"""
    
    system_prompt = """You are an academic research assistant. Generate REAL academic citations with valid DOIs. 

CRITICAL REQUIREMENTS:
- Provide exactly 20 citations
- Use actual published academic articles only
- Format as tab-delimited table: Author Names[TAB]Title[TAB]Publication Year[TAB]Journal Name[TAB]DOI
- Do NOT include headers or markdown formatting
- Do NOT add explanatory text before or after the table
- Each line should contain exactly 5 fields separated by tabs
- DOIs should be in format: 10.xxxx/xxxx
- Each citation should be a genuine published academic article"""
    
    # Create Gemini-specific prompt
    enhanced_prompt = create_gemini_prompt(prompt)
    full_prompt = f"{system_prompt}\n\n{enhanced_prompt}"
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.05,  # Very low temperature for consistency
                    "max_output_tokens": 4000,  # More tokens for complete responses
                    "top_p": 0.7,  # Better control
                    "top_k": 20,
                    "candidate_count": 1,
                }
            )
            
            # Validate response quality
            is_valid, reason = validate_response(response.text, country, prompt_type)
            if is_valid:
                print(f"  Valid response received")
                return response.text
            else:
                print(f"  Invalid response ({reason}), retrying...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"  All validation attempts failed for {country} ({prompt_type})")
                    return response.text  # Return anyway for debugging
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error generating citations for {country} ({prompt_type}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to generate citations for {country} ({prompt_type}) after {max_retries} attempts: {e}")
                return None

# ------------------- ENHANCED CITATION PARSING ---------------------
def parse_citations_improved(citation_text, country, income_level, prompt_type):
    """Enhanced parsing for Gemini responses with better validation and cleaning"""
    if not citation_text:
        print(f"  WARNING: Empty response for {country} ({prompt_type})")
        return []
    
    # Clean the response
    lines = citation_text.strip().split('\n')
    
    # Remove markdown formatting, headers, and junk
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        
        # Remove markdown table formatting
        line = line.replace('|', '\t').replace('```', '').replace('***', '').replace('---', '')
        
        # Skip obvious headers and junk
        skip_indicators = [
            'author', 'title', 'journal', 'doi', 'year', 'publication',  # Headers
            'note:', 'disclaimer', 'please', 'verify', 'based on', 'i cannot',  # Disclaimers
            'sorry', 'apologize', 'unable to provide',  # Apologies
            'here are', 'here is', 'below are', 'following are'  # Introductions
        ]
        
        if (line and 
            not any(indicator in line.lower() for indicator in skip_indicators) and
            len(line) > 20):  # Minimum length check
            cleaned_lines.append(line)
    
    print(f"  Cleaned lines: {len(cleaned_lines)} (from {len(lines)} original)")
    
    # Better delimiter detection with scoring
    delimiter_scores = {}
    for delimiter in ['\t', '|', ';', ',']:
        score = 0
        for line in cleaned_lines[:10]:  # Check first 10 lines
            parts = line.split(delimiter)
            if 4 <= len(parts) <= 6:  # Expect 4-6 columns
                score += 1
                # Bonus for proper formatting
                if len(parts) == 5:
                    score += 0.5
        delimiter_scores[delimiter] = score
    
    best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
    best_score = delimiter_scores[best_delimiter]
    
    print(f"  Using delimiter: '{repr(best_delimiter)}' (score: {best_score})")
    
    # If no good delimiter found, try fallback
    if best_score == 0:
        print(f"  WARNING: No good delimiter found, trying tab as fallback")
        best_delimiter = '\t'
    
    citations = []
    valid_citations = 0
    skipped_lines = 0
    
    for i, line in enumerate(cleaned_lines):
        parts = [p.strip(' "\'|*') for p in line.split(best_delimiter)]
        
        # Enhanced validation
        if (len(parts) >= 4 and 
            parts[0] and len(parts[0]) > 2 and  # Author should be meaningful
            parts[1] and len(parts[1]) > 10 and  # Title should be substantial
            not any(skip in parts[0].lower() for skip in ['author', 'name']) and  # Not a header
            not any(skip in parts[1].lower() for skip in ['title', 'article'])):  # Not a header
            
            # Clean and limit field lengths
            author = parts[0][:200].strip()
            title = parts[1][:300].strip()
            year = parts[2].strip() if len(parts) > 2 else ''
            journal = parts[3][:200].strip() if len(parts) > 3 else ''
            doi = parts[4].strip() if len(parts) > 4 else ''
            
            # Additional validation for year (should be 4 digits or empty)
            if year and not (year.isdigit() and 1900 <= int(year) <= 2025):
                year = ''
            
            citation = {
                'Country': country,
                'Income_Level': income_level,
                'Prompt_Type': prompt_type,
                'LLM': 'Gemini-Flash-Lite',
                'Citation_ID': f"{country}_{prompt_type}_{valid_citations + 1}",
                'Author': author,
                'Title': title,
                'Year': year,
                'Journal': journal,
                'DOI': doi
            }
            citations.append(citation)
            valid_citations += 1
            
            if valid_citations >= 25:  # Stop at reasonable limit
                break
        else:
            skipped_lines += 1
    
    print(f"  Valid citations: {valid_citations}")
    print(f"  Skipped lines: {skipped_lines}")
    
    if valid_citations < 10:
        print(f"  WARNING: Only {valid_citations} valid citations (expected ~20)")
        # Show some examples of what was skipped for debugging
        if len(cleaned_lines) > 0:
            print(f"  Sample raw lines:")
            for line in cleaned_lines[:3]:
                print(f"    '{line[:100]}...'")
    
    return citations

# ------------------- MAIN FUNCTION ---------------------
def main():
    """Main execution function"""
    print(f"Starting ENHANCED citation generation with Gemini {MODEL_NAME}")
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
            filename = f"gemini_{country.replace(' ', '_')}_{prompt_type.lower()}.txt"
            with open(f"data/raw_responses/{filename}", 'w', encoding='utf-8') as f:
                f.write(citation_text)
            
            # Parse citations with enhanced parser
            citations = parse_citations_improved(citation_text, country, income_level, prompt_type)
            
            if len(citations) == 0:
                print(f"  ERROR: No citations parsed from response")
                print(f"  Raw response preview: {citation_text[:200]}...")
                # Save debug info
                with open(f"data/raw_responses/DEBUG_{filename}", 'w', encoding='utf-8') as f:
                    f.write(f"COUNTRY: {country}\nPROMPT_TYPE: {prompt_type}\n\n")
                    f.write(f"RAW RESPONSE:\n{citation_text}\n")
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
                'LLM': 'Gemini-Flash-Lite',
                'Citation_ID': f"{country}_{prompt_type}_FAILED",
                'Author': 'GENERATION_FAILED',
                'Title': 'GENERATION_FAILED',
                'Year': '',
                'Journal': '',
                'DOI': ''
            }
            all_citations.append(failure_citation)
        
        # Delay between tasks to avoid rate limiting
        time.sleep(3)  # Slightly longer delay for stability
    
    # Final save and summary
    df = pd.DataFrame(all_citations)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== ENHANCED CITATION GENERATION COMPLETED ===")
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
        
        # Quality metrics
        avg_citations_per_task = len(valid_citations) / (len(ALL_TASKS) - len(failed_tasks)) if len(ALL_TASKS) > len(failed_tasks) else 0
        print(f"Average citations per successful task: {avg_citations_per_task:.1f}")

if __name__ == "__main__":
    main()