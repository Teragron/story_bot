import subprocess
from gtts import gTTS
from googletrans import Translator
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

translator = Translator()

audio_path = "text.mp3"
API_KEY = "xxxxxxxxxxxxxxxxx"

def tlate(prompt):
  translation = translator.translate(prompt, dest='de')
  return translation.text


def story_teller(prompt):
    cmd = f'./run {"stories15M.bin"} -t {0.8} -p {0.9} -n {256} -i "{prompt}"'
    try:
        # Run the command and capture the output
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        translated = tlate(output)
        speech = gTTS(text = translated, lang = "de", slow = False)
        speech.save(audio_path)
        return str(translated)
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
    update.message.reply_text("(pythonanywhere.com) Please write the beginning of your story in the following format: There once was a ...")

def help_command(update,context):
    update.message.reply_text("Please write the beginning of your story in the following format: There once was a ...")


def handle_message(update, context):
    text = str(update.message.text).lower()
    response = sample_responses(text)
    update.message.reply_text(response)

    update.message.reply_audio(audio=open(audio_path, 'rb'))


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
