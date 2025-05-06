# **DocuMind – A Beginner’s Guide to Chatbots**

download documentation.docx(present in source code repo) for documentation in word format
## 1 **Table of Contents**

[2 What is this project about?](#_toc197242091)  
- [2.1 Why are chatbots important / cool?](#_toc197242092)  
- [2.2 Who is this guide for?](#_toc197242093)  
- [2.3 What you’ll learn by the end?](#_toc197242094)  

[3 Project Workflow (High-Level Overview)](#_toc197242095)  
- [3.1.1 PDF Input](#_toc197242096)  
- [3.1.2 Text Extraction and Chunking](#_toc197242097)  
- [3.1.3 Embedding Generation](#_toc197242098)  
- [3.1.4 Storage and Caching](#_toc197242099)  
- [3.1.5 Question Answering Setup](#_toc197242100)  
- [3.1.6 User Interaction](#_toc197242101)  
- [3.1.7 Feedback Collection](#_toc197242102)  

[4 Technologies Used (Summary)](#_toc197242103)  
- [4.1.1 PyMuPDF](#_toc197242104)  
- [4.1.2 NLTK (Natural Language Toolkit)](#_toc197242105)  
- [4.1.3 Sentence Transformers](#_toc197242106)  
- [4.1.4 FAISS (Facebook AI Similarity Search)](#_toc197242107)  
- [4.1.5 SQLite](#_toc197242108)  
- [4.1.6 Transformers (Hugging Face)](#_toc197242109)  
- [4.1.7 Torch (PyTorch)](#_toc197242110)  
- [4.1.8 Regex (re module)](#_toc197242111)  
- [4.1.9 Logging](#_toc197242112)  
- [4.1.10 NumPy](#_toc197242113)  
- [4.1.11 Datetime](#_toc197242114)  

[5 Deep Dive into Each Technology](#_toc197242115)  
- [5.1 Tkinter](#_toc197242116)  
  - [5.1.1 What is Tkinter?](#_toc197242117)  
  - [5.1.2 How Tkinter works behind the scenes?](#_toc197242118)  
  - [5.1.3 How is Tkinter used in the project?](#_toc197242119)  
- [5.2 PyMuPDF](#_toc197242120)  
  - [5.2.1 What is PyMuPDF?](#_toc197242121)  
  - [5.2.2 How does PyMuPDF work behind the scenes?](#_toc197242122)  
  - [5.2.3 How is PyMuPDF used in the project?](#_toc197242123)  
- [5.3 NLTK](#_toc197242124)  
  - [5.3.1 What is NLTK?](#_toc197242125)  
  - [5.3.2 How does NLTK work behind the scenes?](#_toc197242126)  
  - [5.3.3 How is NLTK used in the project?](#_toc197242127)  
- [5.4 Sentence Transformers](#_toc197242128)  
  - [What are Sentence Transformers?](#_toc197242129)  
  - [5.4.1 How do Sentence Transformers work behind the scenes?](#_toc197242130)  
  - [How are Sentence Transformers used in the project?](#_toc197242131)  
- [5.5 FAISS](#_toc197242132)  
  - [5.5.1 What is FAISS?](#_toc197242133)  
  - [5.5.2 How does FAISS work behind the scenes?](#_toc197242134)  
  - [5.5.3 How is FAISS used in the project?](#_toc197242135)  
- [5.6 SQLite](#_toc197242136)  
  - [5.6.1 What is SQLite?](#_toc197242137)  
  - [5.6.2 How does SQLite work behind the scenes?](#_toc197242138)  
  - [5.6.3 How is SQLite used in the project?](#_toc197242139)  
- [5.7 Transformers (Hugging Face)](#_toc197242140)  
  - [5.7.1 What are Transformers (Hugging Face)?](#_toc197242141)  
  - [5.7.2 How do Transformers work behind the scenes?](#_toc197242142)  
  - [5.7.3 How are Transformers used in the project?](#_toc197242143)  
- [5.8 Torch (PyTorch)](#_toc197242144)  
  - [5.8.1 What is Torch (PyTorch)?](#_toc197242145)  
  - [5.8.2 How does Torch (PyTorch) work behind the scenes?](#_toc197242146)  
  - [5.8.3 How is Torch (PyTorch) used in the project?](#_toc197242147)  
- [5.9 Regex (re module)](#_toc197242148)  
  - [5.9.1 What is Regex (re module)?](#_toc197242149)  
  - [5.9.2 How does Regex work behind the scenes?](#_toc197242150)  
  - [5.9.3 How is Regex (re module) used in the project?](#_toc197242151)  
- [5.10 Logging](#_toc197242152)  
  - [5.10.1 What is Logging?](#_toc197242153)  
  - [5.10.2 How does Logging work behind the scenes?](#_toc197242154)  
  - [5.10.3 How is Logging used in the project?](#_toc197242155)  
- [5.11 NumPy](#_toc197242156)  
  - [5.11.1 What is NumPy?](#_toc197242157)  
  - [5.11.2 How NumPy works behind the scenes?](#_toc197242158)  
  - [5.11.3 How is NumPy used in the project?](#_toc197242159)  
- [5.12 Datetime](#_toc197242160)  
  - [5.12.1 What is Datetime?](#_toc197242161)  
  - [5.12.2 How Datetime works behind the scenes?](#_toc197242162)  
  - [5.12.3 How is Datetime used in the project?](#_toc197242163)  

[6 Model Selection and Usage](#_toc197242164)  
- [6.1 Sentence Transformer: all-MiniLM-L6-v2](#_toc197242165)  
  - [6.1.1 What is all-MiniLM-L6-v2?](#_toc197242166)  
  - [6.1.2 Architecture and Explanation](#_toc197242167)  
  - [6.1.3 How does all-MiniLM-L6-v2 work?](#_toc197242168)  
  - [6.1.4 Why is it used in this project?](#_toc197242169)  
- [6.2 Text to Text Model: Google Flan T5](#_toc197242170)  
  - [6.2.1 What is FLAN-T5?](#_toc197242171)  
  - [6.2.2 Architecture and Explanation](#_toc197242172)  
  - [6.2.3 How does FLAN-T5 work?](#_toc197242173)  
  - [6.2.4 Why is FLAN-T5 used in this project?](#_toc197242174)  

[7 Project Workflow (Detailed)](#_toc197242175)  
- [7.1 Stage 1: Reading the PDF](#_toc197242176)  
- [7.2 Chunking and Storing](#_toc197242177)  
- [7.3 Creating Embeddings](#_toc197242178)  
- [7.4 Searching with FAISS](#_toc197242179)  
- [7.5 Answering Questions](#_toc197242180)  
- [7.6 Feedback Saving (Optional)](#_toc197242181)  

[8 Project structure and Code documentation](#_toc197242182)  
- [8.1 Project Structure](#_toc197242183)  
- [8.2 Code Documentation](#_toc197242184)  

[9 How to Run the Project](#_toc197242186)  
- [9.1 Clone the Repository](#_toc197242187)  
- [9.2 Install Dependencies](#_toc197242188)  
- [9.3 Run the Script](#_toc197242189)  

[10 Learning Resources (Extra Links)](#_toc197242190)  
- [10.1 Learning Resources: Transformers](#_toc197242191)  
- [10.2 Learning Resources: Sentence Transformers](#_toc197242192)  
- [10.3 Learning Resources: FAISS](#_toc197242193)  
- [10.4 Learning Resources: NumPy](#_toc197242194)  
- [10.5 Learning Resources: Pytorch](#_toc197242195)  
- [10.6 Some more Links](#_toc197242196)  
  - [10.6.1 Regular expressions](#_toc197242197)  
  - [10.6.2 Logging](#_toc197242198)  
  - [10.6.3 Datetime](#_toc197242199)  
  - [10.6.4 PyMuPDF](#_toc197242200)  
  - [10.6.5 SQLite](#_toc197242201)  

[11 Future Improvements / Fun Ideas](#_toc197242202)



# <a name="_toc196993523"></a><a name="_toc197242091"></a>**What is this project about?**
DocuMind🤖💬 is an open-source chatbot that can read, understand, and answer questions about your documents! 📄📚 It’s designed to help teams get instant support ⏱️ and clarify doubts from their existing knowledge base, saving time and boosting employee efficiency 🚀 — all without worrying about costly API bills 💸 or subscriptions 🔒.

## <a name="_toc196993524"></a><a name="_toc197242092"></a>**Why are chatbots important / cool?**
Chatbots are important because they provide 24/7 support 🕒 for teams without the need for manual assistance. This enables users to get instant answers 💡 and understand complex information with ease 📘➡️✨.

They help break down difficult documents 🧠📑, speed up productivity ⚡, and offer personalized support 🎯, making the user experience much more enjoyable and efficient. Plus, they’re always improving! 🔄📈

## <a name="_toc196993525"></a><a name="_toc197242093"></a>**Who is this guide for?**
This guide is for anyone new to Artificial Intelligence 🤔🧠 but who has a basic understanding of computer science 💻. You don’t need to be familiar with all the technologies used — just a willingness to learn and explore! 🌱

## <a name="_toc196993526"></a><a name="_toc197242094"></a>**What you’ll learn by the end?**
By the end of this guide, you’ll gain a clear understanding 🕵️‍♀️ of how the Chatty Chat Bot works under the hood 🔧🤖.

You’ll explore not just AI, but also the fundamental theories 📘 and practical code 💡👨‍💻 that power the whole project. It’s your first step into the world of chatbots 🚪➡️🤖, and a solid foundation for building your own! 🏗️✨

# <a name="_toc196993527"></a><a name="_toc197242095"></a>**Project Workflow (High-Level Overview)** 

### <a name="_toc196993528"></a><a name="_toc197242096"></a>**PDF Input**
The user uploads a PDF file with a file uploader GUI.

### <a name="_toc196993529"></a><a name="_toc197242097"></a>**Text Extraction and Chunking**
PDF content is extracted and split into smaller pieces for easier processing.

### <a name="_toc196993530"></a><a name="_toc197242098"></a>**Embedding Generation**
Each piece of content is converted into a numeric representation called an embedding. This is done using a pretrained Sentence Transformer model and is used for similarity search.

### <a name="_toc196993531"></a><a name="_toc197242099"></a>**Storage and Caching**
The embeddings and text chunks are saved for fast retrieval without reprocessing the same PDF again.

### <a name="_toc196993532"></a><a name="_toc197242100"></a>**Question Answering Setup:**
A language model is loaded to generate answers based on a given context.

### <a name="_toc196993533"></a><a name="_toc197242101"></a>**User Interaction:**
User input is converted into embedding and the chatbot finds the most relevant chunk embedding from the PDF and generates an answer using relevant chunks.

### <a name="_toc196993534"></a><a name="_toc197242102"></a>**Feedback Collection:**
Users can optionally provide feedback on the answers, which is stored in databases and log files for review.

# <a name="_toc196993535"></a><a name="_toc197242103"></a>**Technologies Used (Summary)** 

### <a name="_toc196993536"></a><a name="_toc197242104"></a>**PyMuPDF**
Library for data extraction, analysis, and manipulation of PDF documents. PDF given by user is processed using PyMuPDF.

### <a name="_toc196993537"></a><a name="_toc197242105"></a>**NLTK (Natural Language Toolkit)**
Natural Language Library. NLTK tokenizes (splits) sentences into smaller chunks.

### <a name="_toc196993538"></a><a name="_toc197242106"></a>**Sentence Transformers**
Sentence Transformers is a library used for creating embeddings (Numerical representation) of text chunks and queries.

### <a name="_toc196993539"></a><a name="_toc197242107"></a>**FAISS (Facebook AI Similarity Search)**
Library created by Meta for clustering (grouping of similar chunks) and searching (searching the most relevant chunk based on user query) embeddings.

### <a name="_toc196993540"></a><a name="_toc197242108"></a>**SQLite**
SQLite is a free and open-source relational database. Here SQLite stores PDF embeddings and chunks along with user feedback.

### <a name="_toc196993541"></a><a name="_toc197242109"></a>**Transformers (Hugging Face)**
Library which contains various open-source AI models. AI models used in these projects are imported from Hugging Face.

### <a name="_toc196993542"></a><a name="_toc197242110"></a>**Torch (PyTorch)**
PyTorch is an open-source deep learning framework developed by Meta. In this project, Hugging Face models are implemented using PyTorch.

### <a name="_toc196993543"></a><a name="_toc197242111"></a>**Regex (re module)**
Library for regular expression matching operations. Regex is used for finding certain patterns which need to be removed from input text before it is processed and tokenized.

### <a name="_toc196993544"></a><a name="_toc197242112"></a>**Logging**
Library for event logging. Logging helps in recording errors and feedback in log files for monitoring and debugging purposes.

### <a name="_toc196993545"></a> **<a name="_toc197242113"></a>NumPy:** 
Library for efficient computation on large arrays and matrices. Here it is used for operations related to storage and retrieval of embedding vectors.

### <a name="_toc196993546"></a> **<a name="_toc197242114"></a>Datetime:** 
The `datetime` library supports manipulating dates and times. It is used here for timestamping data such as logs.

# <a name="_toc196993547"></a><a name="_toc197242115"></a>**Deep Dive into Each Technology**

## <a name="_toc196993548"></a><a name="_toc197242116"></a>**Tkinter**

### <a name="_toc196993549"></a><a name="_toc197242117"></a>**What is Tkinter?**
Tkinter is the standard GUI (Graphical User Interface) library of Python. It gives a simple way to create windows, dialogs, and other common GUI elements. Tkinter acts as a Python wrapper around the Tcl (Tool Command Language)/Tk GUI toolkit (A high-level language made for building GUI applications), making it easier for Python developers to build GUI elements and applications without external dependencies.

### <a name="_toc196993550"></a><a name="_toc197242118"></a>**How Tkinter works behind the scenes?**
Tkinter internally bridges Python and the Tcl/Tk interpreter. When a Python script uses Tkinter to create a window or a widget, Tkinter generates the corresponding Tcl commands under the hood. These commands are executed by the Tcl interpreter to render the GUI elements on the screen. Tkinter maintains its own event loop (called `mainloop`), listening for user actions like clicks or keystrokes and dispatching the appropriate callback functions when events occur. This event-driven model allows dynamic interaction between the user and the application.

### <a name="_toc196993551"></a><a name="_toc197242119"></a>**How is Tkinter used in the project?**
In the project, Tkinter is used for:
- Creating a GUI for users to upload PDF files.
- Restricting users to upload only PDF files to avoid errors in further processing.

## <a name="_toc196993552"></a><a name="_toc197242120"></a>**PyMuPDF**

### <a name="_toc196993553"></a><a name="_toc197242121"></a>**What is PyMuPDF?**
PyMuPDF is a lightweight, high-performance Python library for working with PDF documents and other file formats. It provides simple functions for extracting text, images, and metadata as well as for modifying documents.

### <a name="_toc196993554"></a><a name="_toc197242122"></a>**How does PyMuPDF work behind the scenes?**
PyMuPDF is a Python binding for the MuPDF C library, which is designed for fast and memory-efficient PDF rendering. A Python binding refers to functions that allow calling of C/C++ functions with Python code. When a PDF is loaded, PyMuPDF parses the document structure and identifies pages, text blocks, and other data. For text extraction, it reads the internal structure of each page (page dictionary) and reconstructs the visible text by analyzing layout, fonts, positions, and characters without relying on OCR (Optical Character Recognition). This makes extraction very fast and accurate for digital PDFs.

**How is PyMuPDF used in the project?**  
In this project, PyMuPDF is used to:

- Open the PDF uploaded by the user.
- Extract raw text from each page of the PDF.
- Provides extracted text for further processing like cleaning, tokenization, and embedding generation.

## <a name="_toc196993556"></a><a name="_toc197242124"></a>**NLTK**

### <a name="_toc196993557"></a><a name="_toc197242125"></a>**What is NLTK?**  
NLTK (Natural Language Toolkit) is a Python library for working with human language data (Natural Language). It provides easy-to-use interfaces for Natural Language Processing (NLP) tasks like tokenization (splitting text into smaller words or sentences). NLTK is commonly used in research and education due to its ability to make text processing easier.

### <a name="_toc196993558"></a><a name="_toc197242126"></a>**How does NLTK work behind the scenes?**  
NLTK provides pre-built datasets, models, and algorithms that can work with textual data. For tokenization, NLTK uses a mix of rule-based methods and trained models to split text. Some rules like punctuation, spaces, and language rules are used to recognize where sentences end or words break. NLTK stores data like stop words, corpora (collections of text), and grammar structures that can be used directly without training models from scratch.

### <a name="_toc196993559"></a><a name="_toc197242127"></a>**How is NLTK used in the project?**  
In this project, NLTK is used to:

- Break (tokenize) the extracted text from the PDF into smaller chunks.
- Tokenized chunks are then used to generate numerical representations (embeddings).
- Processes text into manageable segments for better analysis and performance.

## <a name="_toc196993560"></a><a name="_toc197242128"></a>**Sentence Transformers**  
(For more explanation on transformers refer 4.6 Transformers (Hugging Face))

### <a name="_toc196993561"></a><a name="_toc197242129"></a>**What are Sentence Transformers?**  
Sentence Transformers is a Python library that creates vector representations (embeddings) of sentences or text chunks. Embeddings are dense numeric arrays that represent and capture the semantic meaning of input text. Sentence Transformers are built on top of pre-trained models like BERT for tasks like sentence similarity, clustering, and paraphrasing mining.

### <a name="_toc196993562"></a><a name="_toc197242130"></a>**How do Sentence Transformers work behind the scenes?**  
Sentence Transformers use attention-based transformer models like BERT. Normally BERT produces token-level (word or letter level) and not sentence-level embeddings. Sentence transformers modify the architecture to help create a single embedding for a full sentence. The numeric representation of a sentence (sentence vector) is done by pooling strategies like mean pooling (average of all token embeddings).

### <a name="_toc196993563"></a><a name="_toc197242131"></a>**How are Sentence Transformers used in the project?**  
In this project, Sentence Transformers are used to:

- Conversion of text chunks into dense vectors (embeddings).
- Embeddings capture the true semantic meaning of text for searching.
- These embeddings are later stored in FAISS for fast similarity search.

## <a name="_toc196993564"></a><a name="_toc197242132"></a>**FAISS**

### <a name="_toc196993565"></a><a name="_toc197242133"></a>**What is FAISS?**  
FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta for efficient similarity search and clustering of dense vectors. A dense vector is an array with mostly non-zero real numbers. It is mainly used for finding nearest neighbors or points that are most similar to a given query point in large datasets.

### <a name="_toc196993566"></a><a name="_toc197242134"></a>**How does FAISS work behind the scenes?**  
FAISS works by representing data points as numerical vectors called embeddings. To find similar vectors, FAISS compares the distance between vectors using metrics like Euclidean distance or cosine similarity. Cosine similarity is a measure of similarity between two vectors that determines the cosine of the angle between them. It quantifies how similar two vectors are in terms of their direction, regardless of their magnitude. For large datasets, FAISS does Indexing by creating special data structures (indexes) like flat indexes, inverted files, or graph-based structures to organize vectors and speed up the similarity search process. Using indexes allows for more efficient storage and retrieval of text data compared to storing the raw text strings. FAISS uses Quantization, a technique to reduce the computational and memory costs by representing vectors with low-precision data types like 8-bit integers instead of the usual 32-bit floats. FAISS does searching using methods like Approximate Nearest Neighbor (ANN). Instead of scanning every vector one by one, FAISS searches only a small portion of the dataset, which provides a balance between speed and accuracy. This makes FAISS fast for real-time search tasks even with millions of vectors.

### <a name="_toc196993567"></a><a name="_toc197242135"></a>**How is FAISS used in the project?**  
In this project, FAISS is used to:

- Create an index from the embeddings generated from PDF text chunks.
- Search the most relevant chunk quickly when the user enters a query.
- Optimize the chunk matching process for large PDFs.

## <a name="_toc196993568"></a><a name="_toc197242136"></a>**SQLite**

### <a name="_toc196993569"></a><a name="_toc197242137"></a><a name="_toc196993570"></a>What is SQLite?  
SQLite (Structured Query Language Lite) is a lightweight and serverless relational database engine. Unlike traditional databases that require a server process (like MySQL or PostgreSQL), SQLite stores the database as a single file on disk. This allows for fast and simple setup, making it perfect for local storage and prototyping.

### <a name="_toc197242138"></a>**How does SQLite work behind the scenes?**  
When a program interacts with SQLite, it reads from and writes directly to the database file using optimized file I/O operations. SQLite directly links into the application and does not require a separate server process. It is a relational database where data is stored in tables as rows and columns. Relationships can be established between different tables and data operations are done by queries. Internally, SQLite uses a B-tree (optimized data structure for data access) to organize tables and indexes for fast lookup. All operations like inserts and updates are wrapped inside transactions to ensure atomicity (operations fully complete or fully fail).

### <a name="_toc197242139"></a>**How is SQLite used in the project?**  
In this project, SQLite is used to:

- Store the extracted chunks of PDF text and their corresponding embeddings.
- Save user feedback for further analysis and fine-tuning.
- Allow immediate retrieval of data when needed without requiring an external database server.

## <a name="_toc197242140"></a>**Transformers (Hugging Face):**

### <a name="_toc197242141"></a><a name="_toc196993571"></a>**What are Transformers (Hugging Face)?**  
Transformers are a type of deep learning model architecture that is effective for tasks with sequential data like text and audio. Entire sequences are processed at once compared to older models like RNNs using a mechanism called attention. Attention allows capturing long-range dependencies better.

Hugging Face provides an open-source platform and library that contains pre-trained models that can be easily fine-tuned or used directly for many tasks like text classification, translation, summarization, and more.

### <a name="_toc197242142"></a>**How do Transformers work behind the scenes?**  
Transformers mainly rely on the *self-attention* mechanism. Self-attention allows transformers to look at all parts of the input sequence simultaneously and decide which parts are important for making predictions. Instead of processing inputs one by one (like RNNs), transformers process the full input in parallel, using layers made up of attention blocks and feed-forward neural networks (Neural networks in which information flows in one direction without loops or feedback). The attention mechanism assigns attention scores that determine the importance given to different parts of the input sequence. Models like BERT, RoBERTa, and GPT are built based on transformer architecture. These models are trained on huge amounts of text data (corpora) so that the models can understand grammar, context, meaning, and even relationships between words.

### <a name="_toc197242143"></a>**How are Transformers used in the project?**  
In this project, the transformers module of HuggingFace is used for:

- Loading the corresponding tokenizer for the pretrained model using Auto.
- Loading the pretrained model using AutoModelForSeq2SeqLM.
- Creating a text-to-text generation pipeline (input of the model is query and output is answer generated based on relevant text chunk) with the model, tokenizer, and device (CPU/GPU).

## <a name="_toc197242144"></a>**Torch (PyTorch)**

### <a name="_toc196993572"></a><a name="_toc197242145"></a><a name="_toc196993575"></a>**What is Torch (PyTorch)?**  
Torch (PyTorch) is an open-source deep learning framework developed by Facebook’s AI Research lab. It is used for building and training neural networks. PyTorch gives easy and flexible methods to define, compute, and optimize operations on tensors (algebraic objects) which are the fundamental blocks of deep learning models.

### <a name="_toc196993573"></a>**How does Torch (PyTorch) work behind the scenes?**  
PyTorch works by creating a dynamic computation graph (also called define-by-run). Computation graphs are used to represent mathematical expressions. It provides a functional description of the required computation. Instead of predefining the entire computation graph before execution, PyTorch builds the graph dynamically along with operations performed. This makes it easier to debug and modify models on the fly. PyTorch uses Tensors (multi-dimensional arrays similar to NumPy arrays but with GPU acceleration support). When tensor operations are performed, PyTorch records these operations to later compute gradients (which measure how much the model's predictions should change to reduce errors) efficiently.

### <a name="_toc196993574"></a>**How is Torch (PyTorch) used in the project?**  
In this project, Torch (PyTorch) is used for:

- Computing and handling tensor operations in transformer models.
- Provides a backend for computations in embedding generation and text-to-text prediction in models imported from the HuggingFace library.

## <a name="_toc197242148"></a>**Regex (re module)**

### <a name="_toc196993576"></a><a name="_toc197242149"></a><a name="_toc196993579"></a>**What is Regex (re module)?**  
Regex (Regular Expression) is a sequence of characters that specifies a match pattern in text. Python’s re module gives methods to search, match, and manipulate strings based on these patterns. Regular expressions can be used for extracting text that follows a specific format (email), validation (checking input format), and splitting strings.

### <a name="_toc196993577"></a>**How does Regex work behind the scenes?**  
Regex works by parsing a regular expression written in a syntax where certain characters have specific meanings. For example, “\d” matches any digit, “\w” matches any alphanumeric character, and “.” matches any character. Under the hood when the pattern is compiled with re.compile(), the regex engine converts the pattern into a state machine (a computation model) that checks the input text character-by-character according to the pattern rules. Optimized internal algorithms allow faster execution of complex matches.

### <a name="_toc196993578"></a>**How is Regex (re module) used in the project?**  
In this project, the re module is used for:

- Sanitizing text given by the user in the form of questions and feedback by removing unprintable characters.
- The sanitized text can be further processed for creating embeddings of user questions or storing user feedback in SQLite.

## <a name="_toc197242152"></a>**Logging**

### <a name="_toc196993580"></a> **<a name="_toc197242153"></a>What is Logging?**  
Logging is the process of recording events, messages, and data generated by an application at execution. It helps developers track the flow of the application, monitoring, and debugging. In Python, the built-in logging module provides methods to create and store logs of different types (DEBUG, INFO, WARNING, ERROR, CRITICAL).

### <a name="_toc196993583"></a> **<a name="_toc196993581"></a><a name="_toc197242154"></a>**How does Logging work behind the scenes?**  
When a log message is generated using Python’s logging module, it is passed through a Logger object, which decides how to handle the message based on its severity level. The Logger object routes log messages to different log handlers like console output or log files. Behind the scenes, the module uses an internal tree-like hierarchy of loggers, filters, formatters, and handlers to control what is logged, how it is formatted, and where it is stored. This ensures that messages are recorded consistently across the application without handling file writes or console prints.

### <a name="_toc196993582"></a> **<a name="_toc197242155"></a>How is Logging used in the project?**  
In this project, logging is used for:

- Recording errors and exceptions during different stages like model loading, text generation, and database operations.
- Logging user feedback data for later analysis and model fine-tuning.

## <a name="_toc197242156"></a>**NumPy**

### <a name="_toc196993584"></a> **<a name="_toc197242157"></a>What is NumPy?**  
NumPy (Numerical Python) is a Python library used for numerical and scientific computing. It provides tools for working with large multi-dimensional arrays and matrices, and also allows a wide range of optimized mathematical functions to operate on these arrays efficiently. It is used in data science, machine learning, and scientific research.

### <a name="_toc196993585"></a> **<a name="_toc197242158"></a>How NumPy works behind the scenes?**  
NumPy arrays are stored in contiguous blocks of memory unlike Python lists, which are collections of pointers to list objects. List objects are slow since list elements do not have a fixed data type and need to be allocated dynamically. Since Numpy arrays contain the same type of elements, they can be stored in a contiguous manner in memory. This type of memory storage can be used for hardware-level optimization in Numpy operations (using C and Fortran internally). This results in much faster computations compared to standard Python operations. NumPy also uses a technique called broadcasting which allows operations on arrays of different shapes without writing for loops.

### <a name="_toc196993586"></a> **<a name="_toc197242159"></a>How is NumPy used in the project?**  
In the project, NumPy is used for:

- Internally, it is used for handling large sets of numerical data efficiently. It helps in performing mathematical operations like summations, averages, and matrix manipulations easily.
- Explicitly, the Numpy module is used in the project to load and store embedding vectors effectively.

## <a name="_toc196993587"></a><a name="_toc197242160"></a>**Datetime**

### <a name="_toc196993588"></a> **<a name="_toc197242161"></a>What is Datetime?**  
Datetime is a Python built-in module that provides classes for manipulating dates (year, month, day) and times (hour, minute, second, microsecond). It supports date arithmetic operations, comparison of dates, supports different date formats, and parses dates from strings.

### <a name="_toc196993589"></a> **<a name="_toc197242162"></a>How Datetime works behind the scenes?**  
In the datetime module, dates and times are internally represented as numbers (with methods like the number of days since a reference date, or the number of microseconds). This numerical representation makes it easier to optimize operations on dates. The module combines date, time, datetime, and timedelta classes to give a full toolkit for working with time-related data. It also handles calculations like leap years behind the scenes to abstract complexity from the user.

### <a name="_toc196993590"></a> **<a name="_toc197242163"></a>How is Datetime used in the project?**  
In the project, datetime is used for:

- Generating timestamps for error logs and feedback logs so that the user can observe when the error or feedback submission occurred.
- Formatting timestamps into human-readable formats when storing feedback data in the SQLite database.
- Keeping track of feedback timestamps to enforce a minimum gap between submissions.


# <a name="_toc196993591"></a><a name="_toc197242164"></a>**Model Selection and Usage**

## <a name="_toc196993592"></a><a name="_toc197242165"></a>**Sentence Transformer: all-MiniLM-L6-v2**

### <a name="_toc196993593"></a><a name="_toc197242166"></a>**What is all-MiniLM-L6-v2?**
all-MiniLM-L6-v2 is a pre-trained Sentence Transformer model designed to convert sentences or short paragraphs into 384-dimensional dense vector embeddings. These embeddings capture the semantic meaning of the text, enabling tasks like semantic search, clustering, and similarity comparison.​

### <a name="_toc196993594"></a><a name="_toc197242167"></a>**Architecture and Explanation**
The model is based on MiniLM, a distilled version of BERT (Bidirectional Encoder Representations from Transformers) developed by Google. MiniLM uses only 6 Transformer layers (L6) and fewer attention heads, making it significantly more lightweight. It is created through knowledge distillation, a process where a smaller student model learns to replicate the behaviour of a larger teacher model by mimicking its predictions and internal representations. Despite its reduced size, a distilled model like MiniLM often maintains comparable performance while requiring far fewer computational resources, making it ideal for deployment on low-power devices and scalable applications. 

MiniLM doesn’t just copy the output predictions of the teacher—it also learns to replicate the self-attention distributions, including attention maps (which show how tokens relate to each other) and value-layer outputs from the teacher’s attention blocks. This allows the student model to internalize the teacher’s reasoning and structural understanding of language, resulting in strong generalization even with limited capacity. 

Additionally, Sentence Transformers adapt the architecture for sentence-level embeddings by introducing pooling strategies. In the case of all-MiniLM-L6-v2, mean pooling is used, where token embeddings are averaged to produce a fixed-size vector representing the entire sentence. This enables the model to efficiently handle tasks like semantic similarity, clustering, and search.

### <a name="_toc196993595"></a><a name="_toc197242168"></a>**How does all-MiniLM-L6-v2 work?**
The input sentence is tokenized into smaller units called tokens.​ These tokens pass through 6 transformer layers, each performing self-attention (considers the entire input sequence to determine the importance of each word in relation to others) and feed-forward (Neural networks with no feedback or loops) operations to build contextual embeddings for each token.​ After the transformer layers, a pooling layer (mean pooling) aggregates token embeddings into a single vector that represents the entire sentence.​ The final result is a 384-dimensional dense vector that captures the semantic meaning of the sentence.

### <a name="_toc196993596"></a><a name="_toc197242169"></a>**Why is it used in this project?**
In the project, all-MiniLM-L6-v2 is used because:

- **Semantic Understanding**: It captures the real meaning behind user queries, not just the words used.​
- **Speed & Efficiency**: It runs fast, even on computers with limited processing power.​
- **High-quality Sentence Embeddings**: The 384-dimensional vectors it produces allow for accurate comparison between queries and document chunks.​
- **Smooth Integration with FAISS**: These embeddings are stored in a FAISS index, which instantly retrieves the most relevant text chunks based on user questions.​ 

This makes all-MiniLM-L6-v2 ideal for tasks like semantic search and document Q&A in this application.​

---

## <a name="_toc196993597"></a><a name="_toc197242170"></a>**Text to Text Model: Google Flan T5**

### <a name="_toc196993598"></a><a name="_toc197242171"></a>**What is FLAN-T5?**
**FLAN-T5** (Fine-tuned LAnguage Net T5) is an advanced version of Google's T5 (Text-to-Text Transfer Transformer) model. The original T5 model treats every NLP task as a text-to-text problem, FLAN-T5 enhances this approach by incorporating instruction fine-tuning. This means the model is trained not just on tasks but also on understanding and following specific instructions, making it more adept at handling a wide range of tasks with better generalization.

### <a name="_toc196993599"></a><a name="_toc197242172"></a>**Architecture and Explanation**
FLAN-T5 (Fine-tuned Language Net based on T5) builds upon the architecture of the original T5 (Text-to-Text Transfer Transformer) model, which uses a sequence-to-sequence (encoder-decoder) structure. In this architecture, the encoder is responsible for reading and understanding the input text, transforming it into a contextualized internal representation. This representation is then passed to the decoder, which generates the appropriate output text like translation, summary, answer to a question, or any other form of natural language output. This structure allows T5 and its derivatives to treat all NLP tasks uniformly as a text-to-text problem.

What sets FLAN-T5 apart is its instruction fine-tuning process. Unlike traditional fine-tuning, which typically adapts a model to perform well on a narrow set of tasks or datasets, instruction fine-tuning exposes the model to a diverse collection of NLP tasks, each presented in the form of explicit natural language instructions (e.g., "Translate this sentence into French," "Summarize the following paragraph"). These instructions act as prompts that guide the model in understanding the goal of the task, improving its ability to generalize to new or unseen tasks.

To implement this, existing datasets are reformatted using prompting templates that convert tasks into instructional examples. For instance, instead of training the model on raw data for sentiment classification, the data might be framed as: "Is the following review positive or negative? [review text]." By training on thousands of such examples spanning multiple NLP domains—including translation, summarization, question answering, and more—FLAN-T5 learns to follow instructions and adapt to new tasks without needing task-specific fine-tuning.

This instruction-tuned methodology results in a model that is not just capable of solving tasks it was explicitly trained on, but also performs strongly on zero-shot and few-shot evaluations—where the model is asked to perform a task with little to no prior exposure. As a result, FLAN-T5 demonstrates robust generalization, significantly improving performance across a wide range of benchmarks compared to models that were not trained with instructional data.

The FLAN-T5 family includes models of varying sizes to cater to different computational needs:

- **FLAN-T5 Small**: ~80 million parameters
- **FLAN-T5 Base**: ~250 million parameters
- **FLAN-T5 Large**: ~770 million parameters
- **FLAN-T5 XL**: ~3 billion parameters
- **FLAN-T5 XXL**: ~11 billion parameters

### <a name="_toc196993600"></a><a name="_toc197242173"></a>**How does FLAN-T5 work?**
The model receives an input in the form of an instruction and associated text, such as "Translate English to French: 'Hello'". The encoder transforms this input into a sequence of hidden representations, capturing the contextual meaning of the instruction and text. The decoder generates the output text based on the encoded representations, producing responses like "Bonjour" for the above example. FLAN-T5 is trained on a mixture of tasks with varied instructions, enabling it to understand and follow a wide array of prompts effectively. 

### <a name="_toc196993601"></a><a name="_toc197242174"></a>**Why is FLAN-T5 used in this project?**
FLAN-T5 is chosen for this project due to its:

- **Versatility**: Its instruction fine-tuning allows it to perform well across diverse NLP tasks without task-specific fine-tuning.
- **Efficiency**: Smaller models like FLAN-T5 Base or Large offer a good balance between performance and computational resource requirements.
- **Accessibility**: Being open-source and available on platforms like Hugging Face, it's easy to integrate and deploy in various applications.
- **Performance**: FLAN-T5 models have demonstrated strong performance on benchmarks, often rivalling larger models in effectiveness.

These attributes make FLAN-T5 a suitable choice for building applications that require understanding and generating human-like text based on instructions.

# <a name="_toc196993602"></a><a name="_toc197242175"></a>**Project Workflow (Detailed)**

## <a name="_toc197242176"></a>**Stage 1: Reading the PDF**

**🎯 Goal**

Open a PDF file and read its entire text content.

**🧠 What's happening?**

First, a small window pops up asking the user to choose a .pdf file.

The code then uses a Python library called PyMuPDF to go through every page in that PDF and collect all the visible text.

This full text is joined into a single long string.

We then use a special function (hashlib) to generate a unique fingerprint (called a hash) of the entire text. This is like giving your PDF a digital ID, useful for caching (so the app doesn’t redo the same work next time).

**✅ What we achieved**

We loaded all the text from the PDF into memory and gave it a unique ID for later use.

---

## <a name="_toc197242177"></a>**Chunking and Storing**

**🎯 Goal**

Break that long text into small parts (“chunks”) for better processing.

**🧠 Why this step?**

Large language models (LLMs), like ChatGPT, can't process very large text inputs all at once. They work best with small, meaningful portions of text. So we divide the long text into small overlapping segments.

**💡 How it's done**

The text is split into sentences using the nltk library.

We combine sentences into groups (chunks), with about 500 words per chunk.

To maintain continuity between chunks, we let each one slightly overlap (100 words by default) with the next.

These chunks are stored in a small database (SQLite) on your computer. This makes it easy to look up later when we need relevant pieces.

**✅ What we achieved**

Turned one long PDF into organized, smaller pieces stored locally.

---

## <a name="_toc197242178"></a>**Creating Embeddings**

**🎯 Goal**

Convert each chunk of text into a numerical format that a computer can understand and compare.

**🧠 Why embeddings?**

Computers can't understand raw human language. Instead, we translate each chunk of text into a vector (a list of numbers). These vectors represent the meaning of the text.

**💡 How it's done**

We use a tool called SentenceTransformer (a type of AI model) to turn each chunk into a vector.

Each vector captures the semantic meaning of the text — meaning similar chunks get similar vectors.

All these vectors are saved in a file (.npy) to avoid recalculating them later.

**✅ What we achieved**

We now have a computer-readable map of all PDF content, ready for quick comparisons.

---

## <a name="_toc197242179"></a>**Searching with FAISS**

**🎯 Goal**

Find the most relevant chunks of text based on a user’s question.

**🧠 What is FAISS?**

FAISS is a powerful library developed by Facebook that lets you search through vectors very fast. Think of it like a smart search engine that finds “similar meanings” instead of exact words.

**💡 How it works**

We load all our saved vectors into a FAISS index.

When a user types a question, we also turn it into a vector.

FAISS compares this question vector with all the chunk vectors and gives back the top matching chunks.

We look up these matching chunks in our SQLite database.

**✅ What we achieved**

We can now find relevant parts of the PDF that relate to the user's question.

---

## <a name="_toc197242180"></a>**Answering Questions**

**🎯 Goal**

Generate a natural, human-readable answer using the relevant chunks.

**🧠 How does the model generate an answer?**

The top-matching chunks (from Stage 4) are combined into a single context paragraph. This paragraph is passed along with the user's question to a QA model (a fine-tuned AI model that specializes in answering questions).

A prompt is created that looks like this:

Context: [relevant chunks]

Question: [user question]

The QA model then generates a sentence or paragraph as the answer.

**✅ What we achieved**

The system uses the original PDF content + AI to give you a smart answer.

---

## <a name="_toc197242181"></a>**Feedback Saving (Optional)**

**🎯 Goal**

Let users rate or comment on the answers they get, to improve future performance.

**🧠 Why feedback?**

Collecting user feedback helps developers understand if the answers are useful. It can also be used later to fine-tune the AI model or improve data quality.

**💡 How it works**

After every answer, the user can leave a comment or rate it.

The feedback is saved in two places:

- A simple log file (feedback.log) for manual reading.
- A database (feedback.db) for structured analysis.

There's also a rate limit (e.g. 30 seconds) to avoid spam feedback on the same question.

**✅ What we achieved**

A feedback loop that lets the system learn and improve over time.

---

# <a name="_toc196993603"></a><a name="_toc197242182"></a>**Project structure and Code documentation**

## <a name="_toc197242183"></a>**Project Structure**

├── cache/                     # Stores FAISS index, embeddings, and chunk DBs

│   └── <pdf\_hash>/            # Subfolder for each uploaded PDF

│       ├── chunks.db          #(created)Database for storing text chunks

│       ├── embeddings.npy      #(created)Store embeddings arrays

│       ├── faiss.index         #faiss index file for fast retrieval of data 

├── constants.py               # Configuration constants

├── chunk\_extraction.py        # Extracts and chunks PDF text

├── create\_db.py               # Creates SQLite DB to store chunks

├── download\_models.py         # (Optional) Script to download models locally

├── feedback\_logger.py         # Logs user feedback to text file and SQLite

├── file\_uploader.py           # Handles PDF file selection using GUI

├── get\_chunk.py               # Retrieves chunks from SQLite DB by ID

├── load\_qa\_model.py           # Loads local QA (T5) model pipeline

├── loading\_and\_caching.py     # Caches embeddings, FAISS index, and DB

├── main.py                    # Main CLI app for PDF Q&A

├── sanitizer.py               # Cleans and sanitizes input text

├── local\_models/              # Directory to store downloaded local models

│   ├── sentence\_model/        # Saved sentence transformer model

│   └── qa\_model/              # Saved QA (text2text) model

├── errors.log                 # Error logs for debugging

├── feedback.db                # SQLite DB for feedback entries

├── feedback.log               # Text log for feedback entries

├── requirements.txt           # Requirements needed for running chatbot

├── documentation.docx         # Detailed word document about project

└── README.md                  # Get a quick gist about project.



## <a name="_toc197242184"></a>**Code Documentation**

📝 Code documentation is done in the form of docstrings 📄 for modules 📦 as well as functions ⚙️. Refer to them for a better understanding of the code 🧠💡.

# <a name="_toc196993604"></a><a name="_toc197242186"></a>**How to Run the Project**

Follow these simple steps to get the project up and running on your machine:

## <a name="_toc196993605"></a><a name="_toc197242187"></a>**Clone the Repository**

**Download the project files to your local machine:**

```console
git clone https://github.com/KDSCRIPT/DocuMind.git
cd DocuMind
```

<a name="_toc196993606"></a><a name="_toc197242188"></a>Install Dependencies
Make sure you’re using Python 3.7+ and install all required libraries:

```console
pip install -r requirements.txt
```

💡 *Tip*: Consider using a virtual environment to avoid conflicts:

```console
python -m venv venv
```

**Linux:** 
```console
source venv/bin/activate
```
**Windows:** 
```console
venv\Scripts\activate
```

```console
pip install -r requirements.txt
```

<a name="_toc196993607"></a><a name="_toc197242189"></a>**Run the Script**

**Execute the main Python script to test or run the model:**

```console
python script.py
```

✅ If everything is set up correctly, you should see file upload GUI for PDF and chatbot output will be visible in terminal after uploading pdf.

# <a name="_toc196993608"></a> **<a name="_toc197242190"></a>Learning Resources (Extra Links)**

## <a name="_toc196993609"></a> **<a name="_toc197242191"></a>Learning Resources: Transformers**

**🔰 Easy Start**

- **📺 The Illustrated Transformer by Jay Alammar**  
  A visual and intuitive guide that demystifies the transformer model's inner workings.  
  **[Explore the guide](http://jalammar.github.io/illustrated-transformer/)**

**🛤️ Intermediate**

- **📘 Visual Guide to Transformer Neural Networks ([Hedu AI by Batool Haider](https://www.youtube.com/@HeduAI))**  
  This is a YouTube series explaining transformer architecture in a visual way.  
  [**Visual Guide to Transformer Neural Networks**](https://www.youtube.com/watch?v=dichIcUZfOw)

**🚀 Deep Dive**

- **📚 Attention is all you need**  
  This is the first research paper by Google that introduced transformer architecture.  
  **[Access the paper](https://arxiv.org/abs/2311.17633)**
- **💻 Transformers from Scratch - DL**  
  Kaggle notebook that implements the encoder of a transformer from scratch.  
  **[transformers-from-scratch](https://www.kaggle.com/code/auxeno/transformers-from-scratch-dl?scriptVersionId=137372275)**

## <a name="_toc196993610"></a> **<a name="_toc197242192"></a>Learning Resources: Sentence Transformers**

**🔰 Easy Start**

- **📖 GeeksforGeeks article on Sentence Transformers**  
  A post from GeeksforGeeks to get started with sentence transformers.  
  [**GeeksforGeeks - Sentence Transformers**](https://geeksforgeeks.org/sentence-transformer/)
- **♠️All all-MiniLM-L6-v2 Hugging Face model card**  
  Model card in Hugging Face for all-MiniLM-L6-v2 that gives a gist about the model and how to use it from Hugging Face model hub.  
  [**HuggingFace all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**🛤️ Intermediate**

- **📘 Official Sentence-Transformers Documentation**  
  Direct from the source. Includes setup, training, and use cases like semantic search and clustering.  
  **[Sentence Transformers Documentation](https://www.sbert.net/)**
- **🔍 Hugging Face space for semantic search using Sentence Transformers**  
  A Hugging Face Space showing how to use sentence transformers for similarity search.  
  **[Using Sentence Transformers for Semantic Search](https://huggingface.co/spaces/sentence-transformers/embeddings-semantic-search)**
- **💡 Blog: Using Sentence Transformers for Semantic Search**  
  [**Read Here**](https://sakilansari4.medium.com/unleashing-the-power-of-sentence-transformers-revolutionising-semantic-search-and-sentence-29405c13f2b0)

**🚀 Deep Dive**

- **📄 Sentence-BERT: Making BERT Efficient for Semantic Similarity**  
  The original research paper that introduced Sentence Transformers and their training strategy.  
  **[Read the paper](https://arxiv.org/abs/1908.10084)**
- **💻 GitHub: Sentence Transformers Codebase**  
  Explore the source code, models, pooling strategies, and training methods.  
  **[Check the repo](https://github.com/UKPLab/sentence-transformers)**

## <a name="_toc196993611"></a> **<a name="_toc197242193"></a>Learning Resources: FAISS**

**🔰 Easy Start**

- **📔Faiss: The Missing Manual**  
  Manual with blogs and videos explaining FAISS from a high-level overview to an in-depth level.  
  [**Faiss: The Missing Manual**](https://www.pinecone.io/learn/series/faiss/)
- **🔍 Simple Python Example Using FAISS**  
  A quick notebook-style tutorial showing how to set up FAISS, index embeddings, and retrieve similar vectors.  
  **[See this GitHub example](https://github.com/facebookresearch/faiss/wiki/Getting-started)**

**🛤️ Intermediate**

- **🧪 Official FAISS Wiki (Facebook Research)**  
  Explains indexing strategies, GPU/CPU usage, quantization, and more. Ideal if you want to optimize your searches or use FAISS at scale.  
  **[Explore the Wiki](https://github.com/facebookresearch/faiss/wiki)**

**🚀 Deep Dive**

- **📄 FAISS Research Paper: “FAISS: A Library for Efficient Similarity Search”**  
  The original paper from Facebook AI Research outlining the design, performance, and scalability of FAISS.  
  **[Read the paper](https://arxiv.org/abs/1702.08734)**
- **📘 FAISS Index Types and Trade-offs (Official Guide)**  
  Detailed descriptions of flat vs. quantized indexes, IVF, HNSW, PQ, and more — great for architecture-level understanding.  
  **[Study here](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)**
- **📊 Benchmarks: FAISS vs Other Vector Databases**  
  In-depth comparisons of FAISS vs alternatives like Annoy, HNSWLib, and ScaNN.  
  **[View benchmark results](https://github.com/erikbern/ann-benchmarks)**

## <a name="_toc196993612"></a> **<a name="_toc197242194"></a>Learning Resources: NumPy**

**🔰 Easy Start**

- **📺 YouTube: "Python NumPy Crash Course" by freeCodeCamp**  
  A beginner-friendly video that covers basics of Numpy.  
  **[Watch here](https://www.youtube.com/watch?v=QUT1VHiLmmI)**
- **📘 Official NumPy Beginner Tutorial**  
  Official documentation of numpy for Absolute Beginners.  
  **[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)**
- **📒 NumPy Illustrated (by Jay Alammar)**  
  Visual explanation of key NumPy concepts.  
  [**A Visual Intro to NumPy and Data Representation**](https://jalammar.github.io/visual-numpy/)

**🛤️ Intermediate**

- **📗 NumPy Documentation**  
  Get a deeper dive into the numpy library straight from the official documentation.  
  [**Numpy Documentation**](https://numpy.org/doc/stable/user/index.html#user)
- **📖 Numpy chapter of Python Data Science Handbook by Jake VanderPlas**  
  Explore basics of Machine learning with Numpy, pandas, Matplotlib, and Scikit Learn.  
  [**Introduction to NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html)**
- **🧰 Cheat Sheet: NumPy for Data Science**  
  A downloadable quick-reference for array operations, math functions, reshaping, and performance tips.  
  **[View and download](https://github.com/numpy/numpy/blob/main/doc/source/user/quickstart.rst)**

**🚀 Deep Dive**

- **📖 Broadcasting: Working with arrays of different shapes.**  
  A deeper look at how broadcasting actually works.  
  **[Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)**
- **🔍 Performance Optimization with NumPy**  
  Learn how to write fast, vectorized code with tips on avoiding loops, using memory efficiently, and benchmarking.  
  **[RealPython Numpy Array Programming](https://realpython.com/numpy-array-programming/)**
- **📄 NumPy Under the Hood**  
  A detailed look at how NumPy arrays work internally, with insights into memory layout, C-extensions, and dtype mechanics.  
  [**Read here**](https://scipy-lectures.org/advanced/advanced_numpy/)

## <a name="_toc196993613"></a> **<a name="_toc197242195"></a>Learning Resources: PyTorch**

**🔰 Easy Start**

- **📺 PyTorch for Deep Learning (Full Course) by freeCodeCamp**  
  A beginner-friendly 6-hour course that walks through the basics of tensors, models, training loops, and building your first neural networks.  
  [Watch here](https://www.youtube.com/watch?v=GIsg-ZUy0MY)
- **📘 PyTorch 60-Minute Blitz (Official)**  
  Hands-on introduction by PyTorch itself. Teaches tensors, autograd, and training a simple neural network.  
  [Start here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- **👩‍🏫 Learn PyTorch from Scratch (GeeksforGeeks)**  
  Simple and progressive explanation of PyTorch fundamentals.  
  [Read here](https://www.geeksforgeeks.org/introduction-to-pytorch/)

**🛤️ Intermediate**

- **📝 Learn Pytorch in a Day (Daniel Bourke)**  
  YouTube video that helps learn pytorch in 24 hours.  
  [Learn Pytorch in a day](https://www.youtube.com/watch?v=Z_ikDlimN6A)
- **🔧 PyTorch Tutorials: Pytorch official documentation**  
  Official documentation for PyTorch.  
  [Explore tutorials](https://pytorch.org/tutorials/)
- **💻 Hands-On with PyTorch Notebooks (by Yann LeCun's team)**  
  A practical collection of interactive notebooks for training neural networks, CNNs, and GANs using PyTorch.  
  [View the GitHub](https://github.com/yunjey/pytorch-tutorial)

**🚀 Deep Dive**

- **📖 PyTorch Internals (by Edward Raff)**  
  A deep dive into how PyTorch works under the hood — including autograd mechanics, tensor operations, and backpropagation internals.  
  [Read here](https://github.com/EdwardRaff/DeepLearningBook/blob/master/notes/pytorch.md)
- **⚙️ PyTorch Profiler Tutorial**  
  Benchmark and optimize your training loops, memory usage, and GPU utilization.  
  [Profiler docs](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- **🧠 From Scratch: Building PyTorch Autograd**  
  Recreate a mini version of PyTorch’s automatic differentiation engine to really understand how gradients work.  
  [Walkthrough](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **🧪 "The Annotated Transformer" (Harvard NLP)**  
  Builds a full Transformer model using PyTorch. It is explained with code and intuition.  
  [Read here](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## <a name="_toc196993614"></a> **<a name="_toc197242196"></a>Some more Links**

### <a name="_toc197242197"></a>**Regular expressions**  
🔬Learn Regex in Depth - [https://www.rexegg.com/](https://www.rexegg.com/)

**Website for trying regular expressions** - [Regex101.com](https://regex101.com)

### <a name="_toc197242198"></a>**Logging**  
🚩Geeks for Geeks post for logging in Python - [https://www.geeksforgeeks.org/logging-in-python/](https://www.geeksforgeeks.org/logging-in-python/)

📄Official Python documentation for Logging - [https://docs.python.org/3/howto/logging.html](https://docs.python.org/3/howto/logging.html)

### <a name="_toc197242199"></a>**Datetime**  
📝Get started with datetime - [https://www.w3schools.com/python/python_datetime.asp](https://www.w3schools.com/python/python_datetime.asp)

📄Official Python documentation for datetime - [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html)

### <a name="_toc197242200"></a>**PyMuPDF**  
📄Official documentation for PyMuPDF - [https://pymupdf.readthedocs.io/en/latest/index.html](https://pymupdf.readthedocs.io/en/latest/index.html)

### <a name="_toc197242201"></a>**SQLite**  
📽️FreeCodeCamp video to learn SQL - [https://www.youtube.com/watch?v=HXV3zeQKqGY](https://www.youtube.com/watch?v=HXV3zeQKqGY)

📄Documentation for using SQLite in Python - [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)

# <a name="_toc196993620"></a> **<a name="_toc197242202"></a>Future Improvements / Fun Ideas**

Try different Models

File Validation

Evaluation of model pending

Support for multiple Document types

Scale model for multiple PDFs

Fine tune based on Feedback

