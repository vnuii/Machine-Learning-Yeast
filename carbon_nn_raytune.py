import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import optuna
from optuna.trial import TrialState
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
data = pd.read_csv(
	"/home/von/yeast_metabolism_project/data4torch/OGcount_carbon/Cellobiose_OGcount.tsv",
	sep="\t",
)
print(data.head())

# 提取特征和标签
X = torch.tensor(data.iloc[:, 1:-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

# 分割数据集
X_train, X_temp, y_train, y_temp = train_test_split(
	X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
	X_temp, y_temp, test_size=0.5, random_state=42
)

# 将数据移动到设备
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# 创建数据加载器
def create_dataloaders(batch_size):
	train_loader = DataLoader(
		TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
	)
	val_loader = DataLoader(
		TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
	)
	test_loader = DataLoader(
		TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
	)
	return train_loader, val_loader, test_loader


# 定义模型
class BinaryClassifier(nn.Module):
	def __init__(self, input_size, layer1_size=1024, layer2_size=512, layer3_size=512):
		super(BinaryClassifier, self).__init__()
		self.layer1 = nn.Linear(input_size, layer1_size)
		self.dropout = nn.Dropout(0.5)
		self.layer2 = nn.Linear(layer1_size, layer2_size)
		self.dropout = nn.Dropout(0.5)
		self.layer3 = nn.Linear(layer2_size, layer3_size)
		self.relu = nn.ReLU()
		self.output_layer = nn.Linear(layer3_size, 1)

	def forward(self, x):
		x = self.relu(self.layer1(x))
		x = self.relu(self.layer2(x))
		x = self.relu(self.layer3(x))
		x = torch.sigmoid(self.output_layer(x))
		return x


# 训练和验证模型
def train_model(trial):
	"""
	Train a binary classification model using the given Optuna trial for hyperparameter optimization.

	Args:
		trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.

	Returns:
		float: The best validation accuracy achieved during training.

	Hyperparameters suggested by the trial:
		- layer1_size (int): Size of the first hidden layer.
		- layer2_size (int): Size of the second hidden layer.
		- layer3_size (int): Size of the third hidden layer.
		- lr (float): Learning rate for the optimizer.
		- batch_size (int): Batch size for training (currently fixed at 128).

	Training process:
		- Uses Adam optimizer with weight decay.
		- Binary cross-entropy loss function.
		- Early stopping if validation accuracy does not improve for a specified number of epochs.
		- Logs training and validation loss, and validation accuracy to TensorBoard.
		- Saves the best model based on validation accuracy.

	Raises:
		optuna.exceptions.TrialPruned: If the trial should be pruned based on the reported accuracy.
	"""
	layer1_size = trial.suggest_int("layer1_size", 128, 512)
	layer2_size = trial.suggest_int("layer2_size", 128, 512)
	layer3_size = trial.suggest_int("layer3_size", 128, 512)
	lr = trial.suggest_float("lr", 1e-4, 1e-3)
	# batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
	batch_size = 128

	model = BinaryClassifier(
		input_size=X_train.shape[1],
		layer1_size=layer1_size,
		layer2_size=layer2_size,
		layer3_size=layer3_size,
	).to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
	# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

	train_loader, val_loader, _ = create_dataloaders(batch_size)

	log_dir = "/home/von/yeast_metabolism_project/logs/optuna/trial_{}".format(
		trial.number
	)
	writer = SummaryWriter(log_dir=log_dir)

	best_val_accuracy = 0
	trials_without_improvement = 0
	early_stopping_threshold = 20  # 提前终止阈值：在若干个 epoch 内性能不再提升

	for epoch in range(100):
		model.train()
		train_loss = 0
		for data, target in train_loader:
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output.squeeze(), target.squeeze())
			train_loss += loss.item()
			loss.backward()
			optimizer.step()
		average_train_loss = train_loss / len(train_loader)

		model.eval()
		val_loss = 0
		all_outputs, all_targets = [], []
		with torch.no_grad():
			for data, target in val_loader:
				output = model(data)
				val_loss += criterion(output.squeeze(), target.squeeze()).item()
				all_outputs.extend(output.squeeze().cpu().numpy())
				all_targets.extend(target.cpu().numpy())
		average_val_loss = val_loss / len(val_loader)
		accuracy = accuracy_score(
			all_targets, (torch.tensor(all_outputs) > 0.5).numpy()
		)

		writer.add_scalar("Loss/train", average_train_loss, epoch)
		writer.add_scalar("Loss/val", average_val_loss, epoch)
		writer.add_scalar("Accuracy/val", accuracy, epoch)

		trial.report(accuracy, epoch)

		if trial.should_prune():
			writer.close()
			raise optuna.exceptions.TrialPruned()

		# 检查早停条件
		if accuracy > best_val_accuracy:
			best_val_accuracy = accuracy
			trials_without_improvement = 0
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"layer1_size": layer1_size,
					"layer2_size": layer2_size,
					"layer3_size": layer3_size,
					"lr": lr,
					"batch_size": batch_size,
				},
				"/home/von/yeast_metabolism_project/models/best_model_train.pth",
			)
		else:
			trials_without_improvement += 1

		if trials_without_improvement >= early_stopping_threshold:
			print(f"Early stopping at epoch {epoch}")
			break

	writer.close()
	return best_val_accuracy


# 使用Optuna进行超参数调优和最终评估
def main():
	"""
	Main function to perform hyperparameter optimization using Optuna, train a neural network model with the best 
	hyperparameters, and evaluate its performance on a test set.

	The function performs the following steps:
	1. Creates an Optuna study to maximize the objective function.
	2. Optimizes the study by running multiple trials.
	3. Prints statistics about the study, including the number of finished, pruned, and complete trials.
	4. Retrieves and prints the best trial's value and parameters.
	5. Re-trains the model using the best hyperparameters found during the study.
	6. Implements early stopping based on validation accuracy to avoid overfitting.
	7. Saves the best model during training.
	8. Loads the best model and evaluates its performance on the test set.
	9. Prints the test set performance metrics including accuracy, precision, recall, and F1 score.
	"""
	study = optuna.create_study(direction="maximize")
	study.optimize(train_model, n_trials=100, timeout=6000)

	pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	# 使用最佳超参数重新训练模型
	best_params = trial.params
	model = BinaryClassifier(
		input_size=X_train.shape[1],
		layer1_size=best_params["layer1_size"],
		layer2_size=best_params["layer2_size"],
		layer3_size=best_params["layer3_size"],
	).to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-3)
	# optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)

	train_loader, val_loader, test_loader = create_dataloaders(32)

	# 训练模型
	best_val_accuracy = 0
	early_stopping_threshold = 5
	trials_without_improvement = 0

	for epoch in range(100):
		model.train()
		train_loss = 0
		for data, target in train_loader:
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output.squeeze(), target.squeeze())
			train_loss += loss.item()
			loss.backward()
			optimizer.step()
		average_train_loss = train_loss / len(train_loader)

		model.eval()
		val_loss = 0
		all_outputs, all_targets = [], []
		with torch.no_grad():
			for data, target in val_loader:
				output = model(data)
				val_loss += criterion(output.squeeze(), target.squeeze()).item()
				all_outputs.extend(output.squeeze().cpu().numpy())
				all_targets.extend(target.cpu().numpy())
		average_val_loss = val_loss / len(val_loader)
		accuracy = accuracy_score(
			all_targets, (torch.tensor(all_outputs) > 0.5).numpy()
		)

		if accuracy > best_val_accuracy:
			best_val_accuracy = accuracy
			trials_without_improvement = 0
			torch.save(
				model.state_dict(),
				"/home/von/yeast_metabolism_project/models/best_model.pth",
			)
		else:
			trials_without_improvement += 1

		if trials_without_improvement >= early_stopping_threshold:
			print(f"Early stopping at epoch {epoch}")
			break

	# 加载最佳模型并在测试集上评估
	model.load_state_dict(
		torch.load("/home/von/yeast_metabolism_project/models/best_model.pth")
	)
	model.eval()
	all_outputs, all_targets = [], []
	with torch.no_grad():
		for data, target in test_loader:
			output = model(data)
			all_outputs.extend(output.squeeze().cpu().numpy())
			all_targets.extend(target.cpu().numpy())

	y_pred = (torch.tensor(all_outputs) > 0.5).numpy()
	accuracy = accuracy_score(all_targets, y_pred)
	precision = precision_score(all_targets, y_pred)
	recall = recall_score(all_targets, y_pred)
	f1 = f1_score(all_targets, y_pred)

	print("Best model test set performance:")
	print(f"  Accuracy: {accuracy}")
	print(f"  Precision: {precision}")
	print(f"  Recall: {recall}")
	print(f"  F1 Score: {f1}")


if __name__ == "__main__":
	main()
