import logging
from funsearch import config
import re

def extract_system_prompt(spec_content, function_to_evolve):
    """Extract system prompt from spec file content if present."""
    system_prompt_custom = None
    specification = spec_content
    
    # Look for system prompt in the spec file, make patterns case-insensitive
    system_prompt_start_marker_pattern = r'(?i)#{1,3}\s*SYSTEM\s*PROMPT.*$'
    system_prompt_end_marker_pattern = r'(?i)#{1,3}\s*END\s*SYSTEM\s*PROMPT.*$'

    start_marker_match = re.search(system_prompt_start_marker_pattern, spec_content, re.MULTILINE)
    end_marker_match = re.search(system_prompt_end_marker_pattern, spec_content, re.MULTILINE)
    
    if start_marker_match and end_marker_match:
        try:
            # Extract content between markers
            start_idx = start_marker_match.start()
            start_line_end = spec_content.find('\n', start_marker_match.end())
            if start_line_end == -1:
                start_line_end = start_marker_match.end()
                
            end_idx = end_marker_match.start()
            
            # Extract the prompt content (everything between the start line and end line)
            prompt_parts = spec_content[start_line_end:end_idx].strip()
            if prompt_parts:
                # Remove triple quotes if present
                if prompt_parts.startswith('"""') and prompt_parts.endswith('"""'):
                    prompt_parts = prompt_parts[3:-3]
                else:
                    logging.warning("System prompt does not start and end with triple quotes. Proceeding anyway.")
                
                system_prompt_custom = prompt_parts
                logging.info("Custom system prompt extracted from spec file")
                
                # Find the end of the line containing the end marker
                end_marker_pos = end_marker_match.end()
                line_end = spec_content.find('\n', end_marker_pos)
                end_idx = line_end if line_end != -1 else len(spec_content)
                
                # Handle case where there might be nothing before the system prompt
                before_part = spec_content[:start_idx].strip()
                after_part = spec_content[end_idx:].strip()
                specification = (before_part + "\n" + after_part) if before_part else after_part
            logging.info("System prompt extracted from spec file:")
            logging.info(system_prompt_custom)
            
            # Check if system prompt contains priority_v but function_to_evolve is not priority
            if function_to_evolve != 'priority' and 'priority_v' in system_prompt_custom:
                import time
                logging.warning(f"WARNING: System prompt contains the string 'priority_v' but the function to evolve is '{function_to_evolve}'. Please update your system prompt, as you may confuse the LLM.")
                logging.warning("Continuing in 10 seconds...")
                time.sleep(10)
                
        except Exception as e:
            logging.warning(f"Error extracting system prompt from spec file: {e}")
            logging.info("Using default system prompt")
            
    
    if system_prompt_custom is None:
        system_prompt_custom = config.get_system_prompt(function_to_evolve)
        logging.info("No system prompt found in spec file, using default:")
        logging.info(system_prompt_custom)
        # No need to modify the specification as there's no system prompt to remove
        specification = spec_content
    
    # Check if user prompt contains priority_v but function_to_evolve is not priority
    if function_to_evolve != 'priority':
        # Try to find the first triple-quoted string in the user prompt
        user_prompt = specification
        triple_quote_match = re.search(r'"""(.*?)"""', user_prompt, re.DOTALL)
        if triple_quote_match and 'priority_v' in triple_quote_match.group(1):
            import time
            logging.warning(f"WARNING: User prompt contains the string 'priority_v' but the function to evolve is '{function_to_evolve}'. Please update your user prompt, as you may confuse the LLM.")
            logging.warning("Continuing in 10 seconds...")
            time.sleep(10)
        
    return system_prompt_custom, specification