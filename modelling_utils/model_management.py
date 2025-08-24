import pickle

def import_model_and_comparison(model_path,comparison_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(comparison_path,'rb') as f:
        comparison = pickle.load(f)
    return model,comparison

def save_model_and_comparison(model,comparison,model_path,comparison_path):
    with open(model_path,'wb') as f:
        pickle.dump(model,f)
    with open(comparison_path,'wb') as f:
        pickle.dump(comparison,f)
    return 