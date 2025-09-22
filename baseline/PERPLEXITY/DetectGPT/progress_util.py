import os
import pickle
import concurrent.futures
import hashlib
from tqdm import tqdm

def _get_item_hash(item):
    if isinstance(item, list):
        content = "\n---\n".join(map(str, item))
    else:
        content = str(item)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def run_with_checkpointing(
        all_items,
        process_item_func,
        checkpoint_path,
        save_interval=20,
        desc="Processing with checkpoints"
):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            results_map = pickle.load(f)
    else:
        results_map = {}

    processed_hashes = set(results_map.keys())
    items_to_process = [item for item in all_items if _get_item_hash(item) not in processed_hashes]

    if not items_to_process:
        print("All items have been processed in previous runs.")
        return [_get_result_from_map(item, results_map) for item in all_items]

    print(f"Total {len(all_items)} items, {len(items_to_process)} of which need to be processed.")

    processed_count_since_save = 0
    pbar = tqdm(total=len(items_to_process), desc=desc)
    try:
        for item in items_to_process:
            result = process_item_func(item)
            item_hash = _get_item_hash(item)
            results_map[item_hash] = result
            
            pbar.update(1)
            processed_count_since_save += 1

            if processed_count_since_save >= save_interval:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(results_map, f)
                processed_count_since_save = 0

    finally:
        pbar.close()
        print("\nSaving final checkpoint...")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(results_map, f)
        print(f"Final checkpoint saved to '{checkpoint_path}'.")

    final_ordered_results = [_get_result_from_map(item, results_map) for item in all_items]
    return final_ordered_results

def _get_result_from_map(item, results_map):
    item_hash = _get_item_hash(item)
    return results_map.get(item_hash)


def run_with_checkpointing_parallel(
        all_items,
        process_item_func,
        checkpoint_path,
        max_workers=3,
        save_interval=20,
        desc="Processing in parallel with checkpoints"
):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            results_map = pickle.load(f)
    else:
        results_map = {}

    processed_hashes = set(results_map.keys())
    items_to_process = [item for item in all_items if _get_item_hash(item) not in processed_hashes]

    if not items_to_process:
        print("All items have been processed in previous runs.")
        return [_get_result_from_map(item, results_map) for item in all_items]

    print(f"Total {len(all_items)} items, {len(items_to_process)} of which need to be processed.")
    print(f"Will use {max_workers} parallel workers.")

    processed_count_since_save = 0
    pbar = tqdm(total=len(items_to_process), desc=desc)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_item_func, item): item for item in items_to_process}

            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                item_hash = _get_item_hash(item)
                try:
                    result = future.result()
                    results_map[item_hash] = result

                    pbar.update(1)
                    processed_count_since_save += 1

                    if processed_count_since_save >= save_interval:
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump(results_map, f)
                        processed_count_since_save = 0

                except Exception as exc:
                    print(f'\nItem (hash: {item_hash[:8]}...) encountered an exception during processing: {exc}')
    finally:
        pbar.close()
        print("\nSaving final checkpoint...")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(results_map, f)
        print(f"Final checkpoint saved to '{checkpoint_path}'.")

    final_ordered_results = [_get_result_from_map(item, results_map) for item in all_items]
    return final_ordered_results