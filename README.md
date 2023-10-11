# Telegram Story Bot

This Python script is designed to create interactive text-based stories in a Telegram chat using the Karpathy's tinyllamas model. Users can initiate a story by sending a message in the format "There once was a...". The bot generates and continues the story based on the user's input.

## Prerequisites

Before running this code, make sure you have the following prerequisites in place:

- Python installed on your system (Python 3.6 or later).
- The required Python packages can be installed using `pip`:

    ```bash
    pip install python-telegram-bot
    ```

- You need a Telegram Bot API key. You can obtain one by talking to [BotFather](https://core.telegram.org/bots#botfather).

## Installation

1. Clone the repository and navigate to the project folder:

    ```bash
    git clone https://github.com/karpathy/llama2.c.git
    cd llama2.c
    ```

2. Build and run the project using `make`:

    ```bash
    make runfast
    ```

3. Download the GPT-3.5 model for generating text. You can change the `download_url` variable to a different model if needed. By default, it points to a model provided by Hugging Face:

    ```bash
    download_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin"
    wget $download_url
    ```

## Configuration

Before running the script, make sure to replace `"xxxxxxxxxxxxxxxxxxxxxxxx"` in the `API_KEY` variable with your Telegram Bot API key that you obtained from BotFather.

## Usage

To start the bot, simply run the script:

```bash
python your_script_name.py
```

Once the bot is running, you can interact with it on Telegram. Start a conversation with your bot, and you can use the `/start` and `/help` commands to get instructions on how to initiate a story and some helpful guidance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This code is based on an example from [karpathy's repository](https://github.com/karpathy/llama2.c).
- It uses the [python-telegram-bot](https://python-telegram-bot.readthedocs.io/en/stable/) library to interact with Telegram.

Feel free to customize and modify this code to meet your specific requirements. Happy storytelling!
