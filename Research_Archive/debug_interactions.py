import pickle
import sys

def main():
    path = 'outputs/ebm_single_interaction/ebm_single_model.pkl'
    try:
        with open(path, 'rb') as f:
            ebm = pickle.load(f)
    except Exception as e:
        print(e)
        return

    print(f"Total Terms: {len(ebm.term_names_)}")
    print("Sample Terms:")
    for n in ebm.term_names_[:20]:
        print(f" - {n}")
        
    interactions = [n for n in ebm.term_names_ if ' x ' in n or '&' in n]
    print(f"Total Interactions Found: {len(interactions)}")
    if interactions:
        print("Sample Interactions:")
        for n in interactions[:10]:
            print(f" - {n}")

if __name__ == "__main__":
    main()
