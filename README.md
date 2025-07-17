# AI-Powered Search Agent using LangChain, Arxiv, Wikipedia, and LangGraph

This project is a simple AI-based search agent that answers questions by retrieving information from **Wikipedia** and **Arxiv**. It uses **LangGraph** to build the flow of nodes and **OpenAI GPT-4o** for generating answers. LangSmith is used for tracing the workflow.

---

## 🔧 Technologies Used

* **LangChain OpenAI**
* **LangGraph**
* **LangSmith Tracing**
* **Arxiv API (LangChain Tool)**
* **Wikipedia (LangChain Loader)**
* **OpenAI GPT-4o**

---

## 📄 What It Does

1️⃣ Takes your question (Example: *"What are the recent advancements in AI chatbots?"*)
2️⃣ Searches Wikipedia and Arxiv for relevant information
3️⃣ Uses GPT-4o to generate an answer based on the retrieved data
4️⃣ Displays the answer in the terminal

---

## 🚀 How to Run

1️⃣ Clone the repository:

```bash
git clone <your-repo-link>
cd <your-project-folder>
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Set up API keys:

* OpenAI Key
* LangSmith Key

You can add them via `.env` file or inside the script as environment variables.

4️⃣ Run the code:

```bash
python searchAgent.py
```

---

## 📌 Example Output

**Question:**
What are the most recent advancements in AI chatbots and agents?

**Answer:**
(AI will answer based on Wikipedia and Arxiv content.)

---

## 🏁 Why This Project?

This project is built for learning how to:
✅ Use **LangGraph** for AI workflows
✅ Perform **web search integration**
✅ Generate AI answers using **LLMs**
✅ Trace pipelines with **LangSmith**

