# Image Generation Server on Fast API

## Model Information
This server generates images using the Stable Diffusion 2.1 Base Model, which is a finetuned version of Stable Diffusion 2 Base. 
More information on this model [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
This model can run fine on a 16GB VRAM but for memory efficient attention, you can install the xformers library,

```bash
pip install xformers

```

This guide outlines the steps to set up and run the FastAPI project locally.

## Prerequisites

-   Python 3.9+
-   pip (Python package installer)
-   A CUDA-enabled GPU with CUDA installed. NVDIA GPUs come with CUDA by default. To install or 
    upgrade CUDA version, click [here](https://developer.nvidia.com/cuda-downloads).


## Setup Instructions

1.  **Create a virtual environment:**

    Navigate to the project directory in your terminal and create a virtual environment using the following command:

    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**

    Install the required packages from the `requirements.txt` file using pip:

    ```bash
    pip install -r requirements.txt

    ```

4.  **Installing Pytorch:**

    Based on your Operating System and Version of CUDA, you need to install PyTorch for the model to run. Based on your OS and CuDA version, 
    please visit [here](https://pytorch.org/get-started/locally/#start-locally) to get the install instructions. 
    

4.  **Run the FastAPI application:**

    Start the FastAPI server using uvicorn:

    ```bash
    python server.py
    ```

5.  **Access the application:**

    Congratulations, the app is now running on http://0.0.0.0:8000. 

## Additional Notes

-   Ensure that you have the necessary environment variables configured as required by the application.
-   Refer to the FastAPI documentation for more details and advanced usage: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)