def _simple_processor(desc: str) -> str:
    return desc.strip()

def _first_sentence_processor(desc: str) -> str:
    end_chars = ['.', '?', '!']
    min_index = float('inf')
    for char in end_chars:
        try:
            index = desc.index(char)
            if index < min_index:
                min_index = index
        except ValueError:
            continue
    
    if min_index != float('inf'):
        return desc[:min_index + 1].strip()
    else:
        return desc.strip()

def get_description_processor(name: str):
    if not name:
        return None
    
    if name == 'simple':
        return _simple_processor
    elif name == 'first_sentence':
        return _first_sentence_processor
    
    else:
        print(f"Warning: Unknown description processor name '{name}'. No processor will be applied.")
        return None 