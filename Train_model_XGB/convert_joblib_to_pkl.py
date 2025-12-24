"""
Simple converter: load a joblib model and write it as a pickle (.pkl) file.
Usage: python Train_model/convert_joblib_to_pkl.py --joblib Train_model/outputs/readmission_tpe/best_readmission_xgb_tpe.joblib
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joblib', required=True, help='Path to joblib file')
    parser.add_argument('--pkl', help='Path for output pkl file (default: same dir, .pkl)', default=None)
    args = parser.parse_args()

    joblib_path = args.joblib
    pkl_path = args.pkl if args.pkl else os.path.splitext(joblib_path)[0] + '.pkl'

    try:
        import sys
        import importlib
        import importlib.util
        import site
        import joblib
        import pickle


        def ensure_installed_xgboost_loaded():
            """Ensure that the installed `xgboost` package (from site-packages) is available
            as `sys.modules['xgboost']` even if a local file named `xgboost.py` exists.
            """
            # collect site-packages paths
            site_dirs = []
            try:
                site_dirs.extend(site.getsitepackages())
            except Exception:
                pass
            try:
                site_dirs.append(site.getusersitepackages())
            except Exception:
                pass
            site_dirs = [p for p in site_dirs if p]

            if not site_dirs:
                return

            found_path = None
            for sd in site_dirs:
                candidate = os.path.join(sd, 'xgboost')
                init_py = os.path.join(candidate, '__init__.py')
                if os.path.exists(init_py):
                    found_path = init_py
                    break
            if found_path is None:
                return
            # Load module from the located file
            loader = importlib.machinery.SourceFileLoader('xgboost', found_path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            sys.modules['xgboost'] = module


        ensure_installed_xgboost_loaded()
        model = joblib.load(joblib_path)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(model, pf)
        print('Saved pkl at', pkl_path)
    except Exception as e:
        print('Failed to convert joblib to pkl:', e)

if __name__ == '__main__':
    main()
