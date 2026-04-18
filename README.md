# DynamoDB Boto3 Toolkit

A robust, production-lean Python module for AWS DynamoDB. This project provides a high-level wrapper around the boto3 SDK, implementing professional database patterns such as composite key design, Global Secondary Indexes (GSI), ACID transactions, and conditional updates.

---

## 👤 Author Details

- **Name:** Darshan Ganesh Bhabad  
- **PRN:** 202301040169  
- **Subject:** Cloud Native Applications  
- **Practical No:** 4  

---

## 📂 Project Structure

The project follows a modular structure to ensure clean separation of concerns:

```
Practical_4_dynamodb/
├── dynamodb_module/           
│   ├── config.py              
│   ├── table_manager.py       
│   ├── crud_operations.py     
│   ├── batch_operations.py    
│   ├── transaction_operations.py 
│   ├── error_handler.py       
│   └── __init__.py            
└── main.py                    
```

### Description:
- **config.py** → Boto3 resource & client initialization  
- **table_manager.py** → Table creation with Composite Keys & GSI  
- **crud_operations.py** → CRUD logic, querying & pagination  
- **batch_operations.py** → High-throughput batch writing  
- **transaction_operations.py** → ACID transactions using `TransactWriteItems`  
- **error_handler.py** → Centralized AWS exception handling  
- **main.py** → Demonstration script  

---

## 📐 Database Schema & Design

### Table: `StudentCourses`

| Key Type                | Attribute Name | Data Type   | Description                         |
|------------------------|---------------|-------------|-------------------------------------|
| Partition Key          | student_id    | String (S)  | Unique Student Identifier           |
| Sort Key               | course_id     | String (S)  | Unique Course Identifier            |
| Global Secondary Index | status        | String (S)  | Used for filtering enrollment status |

### GSI Details:
- **Index Name:** `StatusIndex`  
- **Key Schema:** `status (HASH)`  
- **Projection:** `ALL`  

### Use Case:
Enables efficient filtering of students based on their enrollment status (active/inactive) without requiring the Student ID.

---

## 🚀 Features & Implementation

### 1. Production-Lean CRUD

- **Pagination:**
  - Uses `LastEvaluatedKey` to fetch large datasets in chunks  
  - Prevents memory overflow  

- **Conditional Updates:**
  - Uses `ConditionExpression`  
  - Example: Update marks only if new value is higher  

---

### 2. High-Performance Writes

- **Batch Operations:**
  - Uses `batch_writer()`  
  - Groups multiple `PutItem` requests  
  - Reduces network round-trips to AWS  

---

### 3. ACID Transactions

- Uses `transact_write_items`  
- Ensures:
  - Atomicity  
  - Consistency  
  - Isolation  
  - Durability  

- All operations either:
  - ✅ Succeed together  
  - ❌ Fail together  

---

### 4. Resilient Error Handling

- Centralized error handler for:
  - `ProvisionedThroughputExceededException` (with retry logic)  
  - `ConditionalCheckFailedException`  

- Improves system reliability and debugging  

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.8+  
- AWS CLI configured:
  ```
  aws configure
  ```
- Install dependencies:
  ```
  pip install boto3
  ```

---

### ▶️ Running the Project

1. Clone the repository  
2. Navigate to the project directory  
3. Run:

```
python main.py
```

---

## 📜 License

This project is part of the academic curriculum for the **Cloud Native Applications** course.
