# RAG Application 🚀

This application fetches data stored in MongoDB backend servers and stores it in Chroma DB in the form of embeddings for persistent RAG (Retrieval-Augmented Generation) queries across multiple chat sessions. Users can get answers to any query related to the documents. The web backend is written in another repository and communicates with this RAG Application.

## Features ✨

1. **Persistent RAG Queries**: The application enables persistent RAG queries across multiple chat sessions by storing data in Chroma DB.
2. **Optimized Training Time**: The application optimizes model training time by storing embeddings in the form of collections, eliminating the need to train the model on documents every time.
3. **Advanced Answer Generation**: The application uses Langchain, Chroma DB, and Gemini Pro for advanced answer generation.
4. **Web Backend**: The web backend for this application is located in another repository (https://github.com/HanzlaHassan123/FYP-BookWiz).

## Prerequisites 🛠️

- MongoDB
- Chroma DB
- Langchain
- Gemini Pro

## Steps to Run 🏃‍♂️

1. Clone this project using the command: `git clone git@github.com:umair-hassan2/Persistent_RAG-Gemini-Chroma-.git`
2. Run the virtual environment.
3. Switch to backend folder `cd Model_Backend`
4. Install dependencies using `pip install -r reqs.txt`.
5. Run the backend using `uvicorn main:app --reload`.
6. Test the backend using Thunder Client.

## Contributing 🤝

Contributions are welcome! Let's improve it together.

## License 📜

This project is licensed under the MIT License. See the `LICENSE` file for details.
