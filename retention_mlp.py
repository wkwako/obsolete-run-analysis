import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class RetentionDataset(Dataset):
    def __init__(self, X, y):
        #X.values, y.values gets the np array from a pandas df
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RetentionMLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_test, y_test, feature_cols=None):
    if feature_cols is not None:
        X_train = X_train[feature_cols]
        X_test  = X_test[feature_cols]

    input_dim = X_train.shape[1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = RetentionDataset(X_train, y_train)
    test_dataset = RetentionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = RetentionMLP(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #training
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb in train_loader:

            #reset gradients
            optimizer.zero_grad()

            #forward pass
            logits = model(xb)

            #compute loss
            loss = criterion(logits, yb)
            total_loss += loss.item()

            #backward pass
            loss.backward()

            #update weights
            optimizer.step()

        #check progress
        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss {total_loss/len(train_loader):.4f}")

    #evaluation
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            all_logits.append(logits)
            all_labels.append(yb)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits)
    auc = roc_auc_score(all_labels.numpy(), probs.numpy())
    print(f"MLP test AUC: {auc:.4f}")