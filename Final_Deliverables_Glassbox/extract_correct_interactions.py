import pickle
import os

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    model_path = os.path.join(base_dir, 'outputs/ebm_single_interaction/ebm_single_model.pkl')
    list_path = os.path.join(base_dir, 'outputs/ebm_single_interaction/interaction_list.txt')
    
    with open(model_path, 'rb') as f:
        ebm = pickle.load(f)
        
    names = ebm.term_names_
    importances = ebm.term_importances()
    
    pairs = []
    names = ebm.term_names_
    importances = ebm.term_importances()
    
    # Heuristic for Cross Domain: & exists, AND one part is TFIDF, one is not.
    # We rely on 'TFIDF_' prefix for text features ? 
    # Let's check names first.
    # Text features from extract_nursing_notes.py are named 'TFIDF_word'.
    
    for n, imp in zip(names, importances):
        if ' & ' in n:
             parts = n.split(' & ')
             # Check if Cross Domain (One text, one not)
             is_text_0 = 'TFIDF_' in parts[0]
             is_text_1 = 'TFIDF_' in parts[1]
             
             if is_text_0 != is_text_1:
                 pairs.append((n, imp))
            
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    with open(list_path, 'w') as f:
        f.write("All Cross-Interactions (Text x Tabular):\n")
        f.write("=======================================\n")
        for n, imp in pairs:
            # Format: 'Feature1 & Feature2'
            f.write(f"{n}\n")
            
    print(f"Extracted {len(pairs)} interactions to {list_path}")

if __name__ == "__main__":
    main()
