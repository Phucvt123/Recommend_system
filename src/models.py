import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    with open(path,'r') as file:
        col_names = file.readline().strip().split(',')
    data = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    return data,col_names

def split_data(X, y, test_size=0.2,  random_state=None,stratify=None,shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)
    m = len(X)
    
    if isinstance(test_size, int):
        test_len = test_size
    else:
        test_len = int(m * test_size)
    
    if random_state is not None:
        np.random.seed(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        if len(stratify) != m:
            raise ValueError("stratify phải có cùng độ dài với y!")
        
        classes = np.unique(stratify)
        train_idx = []
        test_idx = []
        
        for cls in classes:
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            
            test_cls_idx = cls_indices[: int(len(cls_indices) * test_size)]
            train_cls_idx = cls_indices[int(len(cls_indices) * test_size):]
            
            test_idx.extend(test_cls_idx)
            train_idx.extend(train_cls_idx)
        
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        

        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
    
    else:
        indices = np.arange(m)
        if shuffle:
            np.random.shuffle(indices)
        
        test_idx = indices[:test_len]
        train_idx = indices[test_len:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
def evaluate_models(y_true, y_pred, y_scores = None):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / len(y_true)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = None
    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        desc_idx = np.argsort(y_scores)[::-1]
        y_sorted = y_true[desc_idx]
        
        pos = np.sum(y_true == 1)
        neg = len(y_true) - pos
        correct_pairs = 0
        tp_count = 0
        
        for label in y_sorted:
            if label == 1:
                tp_count += 1
            else:  
                correct_pairs += tp_count
        
        auc = correct_pairs / (pos * neg + 1e-12) if pos > 0 and neg > 0 else 0.5
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "auc": auc,
    }


class LogisticRegressionCustom:
    
    def __init__(self, alpha=0.01, num_iters=1000, lambda_=0.0, random_state=None,verbose = True):
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = None
        self.w = None
        self.b = None
        self.cost_history = []
    
    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-np.clip(z,-250,250)))
    
    def compute_cost_reg(self,X, y, w, b, lambda_ = 0,sample_weight = None):
        m = len(y)
        f_wb = self.sigmoid(X @ w + b)
        f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
        cost = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
        if sample_weight is not None:
            cost = np.mean(cost * sample_weight)
        else:
            cost = np.mean(cost)
        reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
        return cost + reg_cost

    def fit(self, X, y, sample_weight=None):
        """
        Huấn luyện mô hình
        sample_weight: nếu None → tự động dùng class_weight='balanced'
        """
        X = np.array(X, copy = True)
        y = np.array(y).ravel()
        m, n = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []
        
 
        if sample_weight is None:
            sample_weight = self._compute_sample_weights(y)
        else:
            sample_weight = np.array([sample_weight[label] for label in y])
        
        alpha = self.alpha 
        
        for i in range(self.num_iters):

            z = X @ self.w + self.b
            f_wb = self.sigmoid(z)
            error = f_wb - y
            
            dj_dw = (X.T @ (error * sample_weight)) / m + (self.lambda_ / m) * self.w
            dj_db = np.sum(error * sample_weight) / m
            

            self.w -= alpha * dj_dw
            self.b -= alpha * dj_db
            

            cost = self.compute_cost_reg(X, y, self.w, self.b, self.lambda_,sample_weight)
            self.cost_history.append(cost)
            
            if self.verbose and i % max(1, self.num_iters//10) == 0 or i == self.num_iters-1:
                print(f"Iteration {i:5d} | Cost: {cost:.6f} |")
  
        return self

    def y_predict(self, X):
        X = np.array(X)
        prob = self.sigmoid(X @ self.w + self.b)
        return prob
    
    def predict(self, X, threshold=None):
        if threshold is None and self.threshold is not None:
            threshold = self.threshold
        elif threshold is None:
            threshold = 0.5
        proba = self.y_predict(X)
        return (proba >= threshold).astype(float)

    def score(self, X, y, threshold=None, method='f1'):
        """
        Tính F1 (hoặc metric khác) trên tập validation
        Nếu threshold=None → tự tìm threshold tốt nhất
        """
        y_scores = self.y_predict(X)
        if threshold is None:
            if self.threshold is None:
                self.threshold = self.find_best_threshold(y, y_scores, method=method)
            threshold = self.threshold

        y_pred = (y_scores >= threshold).astype(int)
        return evaluate_models(y,y_pred,y_scores)
    
    def find_best_threshold(self,y_true, y_scores, method='f1'):
        """Tìm threshold tốt nhất (copy từ hàm bạn đã có)"""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        thresholds = np.linspace(0.01, 0.99, 300)
        best_thr = 0.5
        best_score = -1
        
        for thr in thresholds:
            y_pred = (y_scores >= thr).astype(int)
            metrics = evaluate_models(y_true, y_pred)
            score = metrics.get(method, metrics['f1'])
            if score > best_score:
                best_score = score
                best_thr = thr
        return best_thr

    def _compute_sample_weights(self,y):
        """class_weight='balanced'"""
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts)
        return weights[np.searchsorted(classes, y)]
    def plot(self):
        plt.figure(figsize=(12,4))
        plt.plot(self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost over Iterations")
        plt.grid(True)
        plt.show()
    
def cross_val_score_custom(model, X, y, cv=5, scoring='f1', sample_weight=None):
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    
    scores = []
    print(f"Bắt đầu {cv}-Fold Cross Validation...")
    
    for i in range(cv):
        start, stop = current, current + fold_sizes[i]
        val_indices = indices[start:stop]
        
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_indices] = False
        train_indices = indices[train_mask]
        
        X_train_fold, y_train_fold = X[train_indices], y[train_indices]
        X_val_fold, y_val_fold = X[val_indices], y[val_indices]
        
        if sample_weight is not None:
            if isinstance(sample_weight, dict):
                fold_weights = sample_weight 
            else:
                fold_weights = np.array(sample_weight)[train_indices]
        else:
            fold_weights = None
            
        model.fit(X_train_fold, y_train_fold, sample_weight=fold_weights)
        
        val_metrics = model.score(X_val_fold, y_val_fold, method=scoring)
        score = val_metrics[scoring]
        scores.append(score)
        print(f"Fold {i+1}/{cv}: {scoring} = {score:.4f}")
        current = stop
        
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print("-" * 30)
    print(f"Average {scoring}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return scores, mean_score
def print_table(data, title="KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH"):
    header = f"| {'Model':<24}|{'Accuracy':>10}|{'Precision':>10}|{'Recall':>10}|{'F1-Score':>10}|{'AUC':>10}|"
    separator = "-" * len(header)
    
    print(f"\n{title}")
    print(separator)
    print(header)
    print(separator)
    
    for row in data:
        print(f"| {row['model']:<24}|{row['accuracy']:>10.4f}|{row['precision']:>10.4f}|{row['recall']:>10.4f}|{row['f1']:>10.4f}|{row['auc']:>10.4f}|")
    
    print(separator + "\n")