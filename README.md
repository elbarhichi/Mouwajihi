# Student Orientation AI Chatbot for Morocco

## Overview

Our project aims to implement a Retrieval-Augmented Generation (RAG) system to assist high school and college students in Morocco with academic and career orientation. The solution consists of two main parts:

1. *Interview Phase*: An interactive session where the Language Learning Model (LLM) gathers information about the student's preferences, interests, and background.
2. *Guiding Phase*: Utilizing RAG, the chatbot processes documents to provide personalized academic and career guidance.

## Team

EL BARHICHI Mohammed
MEZIANY Imane
EL YOUBI Asmae

## Solution Description

We leverage RAG with documents extracted via web scraping from official Moroccan school and university websites. Our approach includes several optimizations:

- *Enhanced Chunking*: Each chunk contains comprehensive information about a single field of study, including access routes, instead of using traditional chunking methods.
- *Graph-Vector Database Integration*: We combine graph databases with vector databases to detect various possible pathways a student can take to reach a specific career goal.

## Architectures

- *Graph Database Architecture*:



- *Overall Project Architecture*:



## Repository Structure

- *Data Directory*: Contains the scraped data post-chunking and the graph structure showing different paths to a given career.
- *Scripts Directory*: Includes scripts for automating web scraping, graph creation, and path detection.
- *Models Directory*: Houses notebooks for comparing different LLM approaches.
- *Demos*: Contains the demo videos of the project
- *App1 and App2*: Streamlit applications for launching the two parts of the solution (interviewing and guiding).

## Getting Started

To get started with our project, follow these steps:

1. *Clone the Repository*:

```b
    sh
    git clone https://github.com/mohammed-el-barhichi/Mouwajihi.git
    cd mouwajihi
```

2. *Install Dependencies*:

```b
    sh
    pip install -r requirements.txt
```

3. *Run the Streamlit Applications*:

    - For the Interview Phase:

    ```b
      sh
      streamlit run app_Interviewing Section.py
    ```

    - For the Guiding Phase:
    ```b
      sh
      streamlit run app_RAG and recommend Section.py
    ```

## Contributing

We welcome contributions to improve our project. Please fork the repository, create a new branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.