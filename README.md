Text Summary Generator
-----------------
This Flask application allows users to upload PDF, DOCX, and TXT files and generates summaries of the contained text. Utilizing PyMuPDF for PDFs, python-docx for DOCX files, and basic Python for TXT files, using a fine-tuned neural network and optional pegasus-xsum language model for summary generation, and newly added image generation using Stable Diffusion--the app provides an easy-to-use interface for extracting, summarizing, and visualizing text.

Features

    File upload support for PDF, DOCX, and TXT formats.
    Options for generating short, medium, and long summaries.
    Multiple model options for summary generation.
    Story visualization for generated summaries.

![image1](https://github.com/midniteclub/SumGen/assets/57697320/33d0c10d-9825-4195-b27f-9d765d25dcf0)

![image2](https://github.com/midniteclub/SumGen/assets/57697320/66264144-d6d6-4920-8c78-4e7395de4e52)


INSTALLATION:
-------------
To run this app, you need Python 3.6+ and pip installed on your system.

1. Clone the Repository (Git Bash):

`git clone https://github.com/midniteclub/SumGen.git`

`cd SumGen`

Or download directly via GitHub, extract SumGen-main, and locate on terminal/cmd (replace `path/to/` with the actual path):

`cd path/to/SumGen-main`


2. Set Up a Virtual Environment (Optional but recommended):

`python -m venv venv`

`source venv/bin/activate`

On Windows, use: `venv\Scripts\activate`


3. Install Required Packages:

`pip install -r requirements.txt`

or

`python -m pip install -r requirements.txt` (if necessary)

Then:

`pip install torch torchvision torchaudio` ([or find your installation type](https://pytorch.org/))

If all else fails (manually install each):
- `pip install transformers`
- `pip install pymupdf`
- `pip install flask`
- `pip install python-docx`
- `pip install --upgrade diffusers[torch]`
- `pip install --upgrade diffusers[flax]`
- `pip install accelerate` (if possible)
- `pip install torch torchvision torchaudio`

4. Run the Application:

`python app.py`


5. Access the web interface at:

`http://127.0.0.1:5000/`



USAGE:
-------------
Upload a File: Navigate to the main page and upload a PDF, DOCX, or TXT file using the file selection input.

Choose Model: Able to use the fine-tuned xsum bart model or Pegasus-xsum language model.

Choose Summary Length: Select the desired summary length (Short, Medium, or Long).

Generate Summary: Click the "Generate Summary" button. If the file is valid and text can be extracted, you will be redirected to a page displaying the generated summary.

[NEW] Generate Image: Option to visualize summary via Stable Diffusion (takes a few minutes)!

Error Handling: If the file type is unsupported or text extraction fails, you will be notified to try another file.



CUSTOMIZATION:
-------------
Model Tuning: The summarization model's performance can be tuned by adjusting parameters in the generate_summary function.

Adding File Types: Support for additional file types can be added by implementing new extraction functions.



TECHNICAL DETAILS:
------------------

Initial Model Training:
-----------------------
The application utilizes a neural network model trained on the PyTorch framework to generate text summaries. The initial training process involved the following:

Dataset: Utilized the XSum dataset for training, which consists of numerous articles paired with their summaries, providing a rich basis for learning summary generation. Due to training time constraints, a portion of the dataset was randomly sampled approximately 10% of the original dataset.

Model Architecture: Selected the BART (Bidirectional and Auto-Regressive Transformers) model for its effectiveness in sequence-to-sequence tasks, including summarization. BART combines bidirectional encoder representation from transformers with an autoregressive decoder, making it well-suited for this task.

Training Process: The model was trained using a masked language modeling (MLM) objective, where portions of the input text are masked, and the model learns to predict the masked words. This approach helps the model understand context and generate coherent summaries.

Evaluation: Employed ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores to evaluate the model's performance, ensuring the generated summaries' quality aligns closely with human-generated references.



Hyperparameter Optimization with Optuna:
---------------------------------------
After the initial training, sought to enhance the model's performance by optimizing its hyperparameters, utilizing Optuna, an open-source hyperparameter optimization framework, which streamlined the search for the best model configuration.

Search Space Definition: Defined a search space for various hyperparameters, including learning rate, batch size, and the number of encoder/decoder layers.

Optimization Objective: The objective function was set to maximize the ROUGE-L score on a validation subset of the XSum dataset, providing a direct measure of summary quality.

Optimization Process: Optuna's efficient search algorithms, such as TPE (Tree-structured Parzen Estimator), explored the defined search space. Through multiple trials, Optuna identified the hyperparameter set that yielded the best performance on the validation set.



Retraining with Optimized Hyperparameters:
------------------------------------------
With the optimized hyperparameters identified, proceeded to retrain the model:

Retraining Configuration: Configured the training process with the optimized hyperparameters, ensuring the model could leverage the improved settings for enhanced performance.

Training Dataset: The same XSum dataset was used, preserving the consistency of the training environment and allowing a direct comparison of performance improvements.

Performance Evaluation: Post-retraining, observed significant improvements in the model's ROUGE scores, indicating more accurate and coherent summaries.



ROUGE METRICS:
--------------
Rouge Scores (before tuning on small xsum subset):

- rouge1: 0.3532
- rouge2: 0.1463
- rougeL: 0.2693
- rougeLsum: 0.2693

Rouge scores (after tuning):

- rouge1: 0.3722
- rouge2: 0.1600
- rougeL: 0.2850
- rougeLsum: 0.2848


Pegasus Rouge scores on xsum test set (https://huggingface.co/google/pegasus-xsum)
(self-reported):
- rouge1: 46.862
- rouge2: 24.453
- rougeL: 39.055
- rougeLsum: 39.099



Additional Considerations:
--------------------------
Computational Resources: Training and hyperparameter optimization were computationally intensive processes, conducted on GPU-accelerated hardware to expedite the experiments.

Model Serving: The fine-tuned model was integrated into the Flask application, allowing users to generate summaries through a user-friendly web interface.

Future Work: Explore further model improvements (larger training data with further hyper parameter tuning, including experimenting with different datasets, model architectures, and training strategies.



CONTRIBUTING:
-------------
Contributions to improve the app or extend its functionality are welcome. Please follow the standard fork, branch, and pull request workflow.



LICENSE:
----------
Apache-2.0



REFERENCES:
-----------
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
