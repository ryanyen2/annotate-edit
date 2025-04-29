export interface Task {
    id: string;
    title: string;
    description: string;
    starterCode: string;
    files?: { [key: string]: string };
}

export const userStudyTasks: Task[] = [
    // TASK 1
    {
        id: '1-1',
        title: 'Task 1-1',
        description: 'Sort Tasks by attribute "due_date"',
        starterCode: `class Task:
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.completed = False

    def mark_complete(self):
        self.completed = True

    def __str__(self):
        return f"Task('{self.title}', Completed: {self.completed})"


class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, description):
        task = Task(title, description)
        self.tasks.append(task)

    def list_tasks(self):
        for task in self.tasks:
            print(task)

    def list_completed_tasks(self):
        for task in self.tasks:
            if task.completed:
                print(task)


# Example Usage
if __name__ == "__main__":
    manager = TaskManager()
    manager.add_task("Buy groceries", "Milk, Bread, Eggs")
    manager.add_task("Read book", "Read 'Clean Code' book")
    manager.list_tasks()
    manager.tasks[0].mark_complete()
    manager.list_completed_tasks()
`,
    },
    {
        id: '1-2',
        title: 'Task 1-2',
        description: 'Allow User to Update Task Details',
        starterCode: `from datetime import datetime

class Task:
    def __init__(self, title, description, due_date=None):
        self.title = title
        self.description = description
        self.completed = False
        self.due_date = datetime.strptime(due_date, "%Y-%m-%d") if due_date else None

    def mark_complete(self):
        self.completed = True

    def __str__(self):
        due_date_str = self.due_date.strftime("%Y-%m-%d") if self.due_date else "No due date"
        return f"Task('{self.title}', Due: {due_date_str}, Completed: {self.completed})"

class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, description, due_date=None):
        task = Task(title, description, due_date)
        self.tasks.append(task)

    def list_tasks(self):
        for task in self.tasks:
            print(task)

    def list_completed_tasks(self):
        for task in self.tasks:
            if task.completed:
                print(task)

    def list_with_priority(self):
        sorted_tasks = sorted(self.tasks, key=lambda task: task.due_date or datetime.max)
        for task in sorted_tasks:
            print(task)


if __name__ == "__main__":
    manager = TaskManager()
    manager.add_task("Buy groceries", "Milk, Bread, Eggs", "2023-05-01")
    manager.add_task("Read book", "Read 'Clean Code' book", "2023-04-25")
    manager.add_task("Pay bills", "Electricity and Internet", "2023-04-20")
    manager.add_task("Exercise", "Go for a run", None)
    print("All Tasks:")
    manager.list_tasks()
    print("Tasks with Priority:")
    manager.list_with_priority()`,
    },
    // TASK 2 
    {
        id: '2-1',
        title: 'Task 2-1',
        description: 'Implement the Manhattan Distance Metric',
        starterCode: `import numpy as np
from typing import List, Tuple

class NearestNeighborRetriever:
    def __init__(self, data: List[Tuple[float]]):
        self.data = np.array(data)
    
    def euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def find_nearest_neighbors(self, query_point: Tuple[float], k: int) -> List[Tuple[float]]:
        distances = [self.euclidean_distance(np.array(query_point), point) for point in self.data]
        nearest_indices = np.argsort(distances)[:k]
        return [self.data[i] for i in nearest_indices]

# Example Usage
data_points = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (6.0, 7.0)]
nn_retriever = NearestNeighborRetriever(data_points)
query_point = (2.5, 3.5)
k = 2
nearest_neighbors = nn_retriever.find_nearest_neighbors(query_point, k)
print(f"Nearest neighbors to {query_point}: {nearest_neighbors}")
`,
    },
    {
        id: '2-2',
        title: 'Task 2-2',
        description: 'Support Datatype with Categorical Features',
        starterCode: `import numpy as np
from typing import List, Tuple

class NearestNeighborRetriever:
    def __init__(self, data: List[Tuple[float]]):
        self.data = np.array(data)
    
    def euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        difference = point1 - point2
        squared_difference = difference ** 2
        sum_squared_difference = np.sum(squared_difference)
        return np.sqrt(sum_squared_difference)
    
    def manhattan_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        absolute_difference = np.abs(point1 - point2)
        return np.sum(absolute_difference)
    
    def calculate_distances(self, query_point: np.ndarray, distance_metric: str) -> List[float]:
        if distance_metric == "euclidean":
            distances = [self.euclidean_distance(query_point, point) for point in self.data]
        elif distance_metric == "manhattan":
            distances = [self.manhattan_distance(query_point, point) for point in self.data]
        else:
            raise ValueError("Unsupported distance metric")
        return distances
    
    def find_nearest_neighbors(self, query_point: Tuple[float], k: int, distance_metric="euclidean") -> List[Tuple[float]]:
        query_array = np.array(query_point)
        distances = self.calculate_distances(query_array, distance_metric)
        nearest_indices = np.argsort(distances)[:k]
        return [self.data[i] for i in nearest_indices]

# Example Usage
data_points = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (6.0, 7.0)]
nn_retriever = NearestNeighborRetriever(data_points)
query_point = (2.5, 3.5)
k = 2
nearest_neighbors = nn_retriever.find_nearest_neighbors(query_point, k, distance_metric="manhattan")
print(f"Nearest neighbors to {query_point} using Manhattan distance: {nearest_neighbors}")
`,
    },
    // TASK 3
    {
        id: '3-1',
        title: 'Task 3-1',
        description: 'Impute Missing Data & Feature Engineering',
        starterCode: `import numpy as np
import pandas as pd
from typing import List, Dict

# Sample data
data = {
    'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'feature2': [2.0, np.nan, 3.0, 4.0, 5.0],
    'feature3': [0.1, 0.3, 1.0, 3.0, 12.0],
    'label': [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def preprocess(self) -> pd.DataFrame:
        return self.dataframe

# Example usage
processor = DataProcessor(df)
processed_df = processor.preprocess()
print(processed_df)`,
    },
    {
        id: '3-2',
        title: 'Task 3-2',
        description: 'Visualize Data Distribution',
        starterCode: `import numpy as np
import pandas as pd
from typing import List, Dict

# Sample data
data = {
    'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'feature2': [2.0, np.nan, 3.0, 4.0, 5.0],
    'feature3': [0.1, 0.3, 1.0, 3.0, 12.0],
    'label': [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def impute_missing_values(self) -> pd.DataFrame:
        for column in self.dataframe.columns:
            mean_value = self.dataframe[column].mean()
            self.dataframe[column].fillna(mean_value, inplace=True)
        return self.dataframe

    def create_quadratic_terms(self) -> pd.DataFrame:
        for column in self.dataframe.columns:
            self.dataframe[f'{column}_squared'] = self.dataframe[column] ** 2
        return self.dataframe

    def preprocess(self) -> pd.DataFrame:
        self.impute_missing_values()
        self.create_quadratic_terms()
        return self.dataframe

# Example usage
processor = DataProcessor(df)
processed_df = processor.preprocess()
print(processed_df)`,
    },
    // TASK 4 - CHI Conference Data Analysis
    {
        id: '4-1',
        title: 'CHI Paper Abstract Analysis',
        description: 'Create a visualization of CHI 2025 papers data',
        starterCode: `import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# df is CHI_2025_program.json (pre-loaded)
print(df.head())

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='award', palette='viridis')
plt.title('Papers by Award Status')
plt.xlabel('Award')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()`,
    },
    {
        id: '4-2',
        title: 'Advanced CHI Paper Analysis Dashboard',
        description: 'Enhance the visualization to analyze trends and patterns in CHI 2025 papers',
        starterCode: `import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# df is CHI_2025_program.json (pre-loaded)
# print(df.head())


def process_text(text):
    stopwords = ['and', 'the', 'to', 'of', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'as', 'by', 'this', 'we', 'are', 'from']
    words = [word for word in text.split(" ") if word not in stopwords and len(word) > 1]
    
    return words

countries = []
for authors_list in df['authors']:
    for author in authors_list:
        if 'affiliations' in author and len(author['affiliations']) > 0:
            if 'country' in author['affiliations'][0]:
                countries.append(author['affiliations'][0]['country'])


track_counts = df['trackId'].value_counts().reset_index()
track_counts.columns = ['Track ID', 'Count']
plt.figure(figsize=(16, 12))


all_words = []
for abstract in df['abstract']:
    all_words.extend(process_text(abstract))
print(all_words[:10])

word_counts = Counter(all_words).most_common(10)
word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
sns.barplot(data=word_df, x='Word', y='Count', palette='mako')
plt.title('Top 10 Words in Abstracts')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()`,
    },
//     {
//         id: '5-1',
//         title: 'main.py',
//         description: 'main financial data handling',
//         starterCode: `class Database:
//     async def execute_query(self, query: str):
//         # Simulated database query
//         await asyncio.sleep(0.1)  # Simulate query execution time
//         if "SELECT balance" in query:
//             return [{"balance": 1000}]
//         elif "SELECT id" in query:
//             return [{"id": 1}]
//         else:
//             return [{"status": "success"}]

// class Transaction:
//     def __init__(self, amount: float, sender: str, recipient: str):
//         self.amount = amount
//         self.sender = sender
//         self.recipient = recipient

// def validate_transaction(transaction: Transaction) -> bool:
//     # Simulated validation logic
//     await asyncio.sleep(0.1)
//     return transaction.amount > 0

// def process_transaction(transaction: Transaction, db: Database):
//     if transaction.amount > 0:
//         if transaction.sender != transaction.recipient:
//             if await validate_transaction(transaction):
//                 # Multiple separate database queries
//                 sender_balance = await db.execute_query(f"SELECT balance FROM accounts WHERE user='{transaction.sender}'")
//                 recipient_exists = await db.execute_query(f"SELECT id FROM accounts WHERE user='{transaction.recipient}'")
                
//                 if sender_balance[0]['balance'] >= transaction.amount and recipient_exists:
//                     # Perform the transaction
//                     await db.execute_query(f"UPDATE accounts SET balance = balance - {transaction.amount} WHERE user='{transaction.sender}'")
//                     await db.execute_query(f"UPDATE accounts SET balance = balance + {transaction.amount} WHERE user='{transaction.recipient}'")
//                     return {"status": "success", "message": "Transaction processed successfully"}
//                 else:
//                     return {"status": "error", "message": "Insufficient funds or recipient not found"}
//             else:
//                 return {"status": "error", "message": "Transaction validation failed"}
//         else:
//             return {"status": "error", "message": "Sender and recipient cannot be the same"}
//     else:
//         return {"status": "error", "message": "Invalid transaction amount"}


// # Example usage
// db = Database()
// transaction = Transaction(100.0, "sender@example.com", "recipient@example.com")
// result = process_transaction(transaction, db)
// print(result)`,
//     },
//     {
//         id: '6-1',
//         title: 'main.py',
//         description: 'main driver code for following tasks',
//         starterCode: `# main.py
// # from pipeline import preprocess_pipeline

// def main():
//     # Load data
//     train_data = [
//     {"name": "age", "type": "numerical", "value": [34, 50, 29], "scaling": "standardize"},
//     {"name": "city", "type": "categorical", "value": ["Paris", "London", "Paris"]}
//     ]

//     test_data = [
//         {"name": "age", "type": "numerical", "value": [40, 60, 30], "scaling": "standardize"},
//         {"name": "city", "type": "categorical", "value": ["London", "Paris", "London"]}
//     ]
    
//     # Preprocess data
//     train_data = preprocess_pipeline(train_data, "train")
//     test_data = preprocess_pipeline(test_data, "test")

//     print("Train data:", train_data)
//     print("Test data:", test_data)

// if __name__ == "__main__":
//     main()`,
//     },
//     {
//         id: '6-2',
//         title: 'data_utils.py',
//         description: 'Utility functions for data handling',
//         starterCode: `# data_utils.py
// def min_max_scale(values):
//     min_val = min(values)
//     max_val = max(values)
//     if max_val - min_val == 0:
//         return [0 for _ in values]
//     return [(v - min_val) / (max_val - min_val) for v in values]

// # The following functions are placeholders and unused initially,
// # to be utilized after Alicia refines her sketches.
// def standardize(values):
//     mean = sum(values) / len(values) if len(values) > 0 else 0
//     variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 0 else 0
//     std_dev = variance ** 0.5
//     if std_dev == 0:
//         return [0 for _ in values]
//     return [(x - mean) / std_dev for x in values]

// def one_hot_encode(values):
//     unique_vals = sorted(set(values))
//     encoded = []
//     for val in values:
//         vec = {uv: (1 if uv == val else 0) for uv in unique_vals}
//         encoded.append(vec)
//     return encoded
// `,
//     },
//     {
//         id: '6-3',
//         title: 'pipeline.py',
//         description: 'The pipeline for data processing',
//         starterCode: `# pipeline.py

// def preprocess_pipeline(data, data_type):
//     # data: a list of feature dicts with structure:

//     processed_data = []
//     for feature in data:
//         if feature["type"] == "numerical":
//             scaled_values = min_max_scale(feature["value"])
//             processed_data.append({"type": "numerical", "name": feature["name"], "value": scaled_values})
//         elif feature["type"] == "categorical":
//             # Currently unchanged
//             processed_data.append(feature)

//     return processed_data
// `,
//     },
];
