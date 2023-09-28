!git clone https://github.com/karpathy/llama2.c.git
%cd llama2.c
!make runfast
download_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin"

!wget $download_url

import subprocess
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxx" #replace with your telegram api key you took from BotFather

def story_teller(prompt):
    cmd = f'./run {"stories42M.bin"} -t {0.8} -p {0.9} -n {256} -i "{prompt}"'
    try:
        # Run the command and capture the output
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during command execution
        return f"Error: {str(e)}"


def sample_responses(input_text):
    user_message = str(input_text).lower()
    keywords = ["there", "once", "was"]
    if any(keyword in user_message for keyword in keywords):
        return story_teller(user_message)
    else:
        return "You can use the command /help to get started"

def start_command(update,context):
    update.message.reply_text("type something to start")
    
def help_command(update,context):
    update.message.reply_text("Please write the beginning of your story in the following format: There once was a ...")

    
def handle_message(update, context):
    text = str(update.message.text).lower()
    response = sample_responses(text)
    
    update.message.reply_text(response)
    
def error(update, context):
    print("Update {} caused error {}").format(update, context.error)

    
def main():
    updater = Updater(API_KEY)
    dp = updater.dispatcher
    print("Bot started..")
    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))

    dp.add_handler(MessageHandler(Filters.text, handle_message))
    
    dp.add_error_handler(error)
    
    updater.start_polling()
    updater.idle()

    
main()
