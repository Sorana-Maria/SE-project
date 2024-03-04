# SummarEase

SummarEase is an innovative software application designed to assist students by providing centralized, concise summaries of study materials. This includes long or short documents, book chapters, and articles, with a particular focus on scientific literature. SummarEase leverages advanced AI components to transform the way students interact with academic content, making study sessions more efficient and productive.

## Key Features

- **Document Summarization**: Utilizes a fine-tuned Llama 2 model to generate abstract-like summaries of complex academic papers and literature.
- **Multi-language Support**: Features a translation model to provide summaries in multiple languages, catering to a global student base.
- **Custom AI Training**: Employs a trimmed version of the arxiv-summarization dataset for fine-tuning on scientific literature, optimizing the model for STEM fields.

## Technology Stack

- **Llama 2-7b Model**: A powerful language model fine-tuned on a subset of the arxiv-summarization dataset from Hugging Face for generating summaries.
- **BART Translation Model**: Used for translating summaries into various languages, ensuring wide accessibility.
- **AWS for Scalability**: Hosted on AWS instances for reliable and scalable access. Utilizes Apache server on Ubuntu, along with a MySQL and PHP stack for the web interface.
- **Flask for Backend**: A Flask server runs the AI models, handling requests and processing summaries.

## Getting Started

To access SummarEase, visit our application at [ec2-52-91-198-116.compute-1.amazonaws.com](http://ec2-52-91-198-116.compute-1.amazonaws.com).(Might be temporarily down due to cost issues) 

### Installation

To set up a development environment for SummarEase, follow these steps:

1. **Clone the Repository**: `git clone https://github.com/SummarEase/SummarEase.git`
2. **Install Dependencies**: Navigate to the cloned directory and run `pip install -r requirements.txt` to install the required Python packages.
3. **Environment Setup**: Configure your local environment variables as needed for database and model access.
4. **Run the Flask Application**: Execute `python app.py` to start the Flask server locally.

### Usage

1. **Uploading Documents**: Users can upload documents through the web interface. Supported formats include PDF, DOCX, and plain text.
2. **Selecting Language**: Choose your preferred language for the summary from the available options.
3. **Generating Summaries**: Submit the document for processing. The summary will be displayed on the web interface.

## Contributing

We welcome contributions from the community. If you're interested in improving SummarEase or adding new features, please fork the repository and submit a pull request.

## License

SummarEase is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- **Dataset**: Our fine-tuning utilizes the arxiv-summarization dataset from [Hugging Face](https://huggingface.co/datasets/ccdv/arxiv-summarization).
- **AI Models**: Thanks to OpenAI for the Llama 2 model and to the creators of the BART translation model for enabling multi-language support.
