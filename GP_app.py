from fuzzywuzzy import fuzz
import torch
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize import word_tokenize
import os
import nltk
import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize import word_tokenize
import pyarabic.araby as araby
from nltk.tokenize import word_tokenize
import joblib
import nltk
import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.stem import ARLSTem
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
! pip install stop-words
from stop_words import get_stop_words
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_id = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

bert_tokenizer = AutoTokenizer.from_pretrained(model_id)
bert_model = AutoModelForMaskedLM.from_pretrained(model_id)


from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='macro') , accuracy_score(labels,preds)

dataset=pd.read_excel('/content/original_dataset.xlsx')

dataset=dataset[['review_description','rating']]
dataset.head()
dataset.rename(columns={'review_description':'text','rating': 'labels'}, inplace=True, errors='raise')

emojis = {
    "🙂":"يبتسم",
    "😂":"يضحك",
    "💔":"قلب حزين",
    "🙂":"يبتسم",
    "❤️":"حب",
    "❤":"حب",
    "😍":"حب",
    "😭":"يبكي",
    "😢":"حزن",
    "😔":"حزن",
    "♥":"حب",
    "💜":"حب",
    "😅":"يضحك",
    "🙁":"حزين",
    "💕":"حب",
    "💙":"حب",
    "😞":"حزين",
    "😊":"سعادة",
    "👏":"يصفق",
    "👌":"احسنت",
    "😴":"ينام",
    "😀":"يضحك",
    "😌":"حزين",
    "🌹":"وردة",
    "🙈":"حب",
    "😄":"يضحك",
    "😐":"محايد",
    "✌":"منتصر",
    "✨":"نجمه",
    "🤔":"تفكير",
    "😏":"يستهزء",
    "😒":"يستهزء",
    "🙄":"ملل",
    "😕":"عصبية",
    "😃":"يضحك",
    "🌸":"وردة",
    "😓":"حزن",
    "💞":"حب",
    "💗":"حب",
    "😑":"منزعج",
    "💭":"تفكير",
    "😎":"ثقة",
    "💛":"حب",
    "😩":"حزين",
    "💪":"عضلات",
    "👍":"موافق",
    "🙏🏻":"رجاء طلب",
    "😳":"مصدوم",
    "👏🏼":"تصفيق",
    "🎶":"موسيقي",
    "🌚":"صمت",
    "💚":"حب",
    "🙏":"رجاء طلب",
    "💘":"حب",
    "🍃":"سلام",
    "☺":"يضحك",
    "🐸":"ضفدع",
    "😶":"مصدوم",
    "✌️":"مرح",
    "✋🏻":"توقف",
    "😉":"غمزة",
    "🌷":"حب",
    "🙃":"مبتسم",
    "😫":"حزين",
    "😨":"مصدوم",
    "🎼 ":"موسيقي",
    "🍁":"مرح",
    "🍂":"مرح",
    "💟":"حب",
    "😪":"حزن",
    "😆":"يضحك",
    "😣":"استياء",
    "☺️":"حب",
    "😱":"كارثة",
    "😁":"يضحك",
    "😖":"استياء",
    "🏃🏼":"يجري",
    "😡":"غضب",
    "🚶":"يسير",
    "🤕":"مرض",
    "‼️":"تعجب",
    "🕊":"طائر",
    "👌🏻":"احسنت",
    "❣":"حب",
    "🙊":"مصدوم",
    "💃":"سعادة مرح",
    "💃🏼":"سعادة مرح",
    "😜":"مرح",
    "👊":"ضربة",
    "😟":"استياء",
    "💖":"حب",
    "😥":"حزن",
    "🎻":"موسيقي",
    "✒":"يكتب",
    "🚶🏻":"يسير",
    "💎":"الماظ",
    "😷":"وباء مرض",
    "☝":"واحد",
    "🚬":"تدخين",
    "💐" : "ورد",
    "🌞" : "شمس",
    "👆" : "الاول",
    "⚠️" :"تحذير",
    "🤗" : "احتواء",
    "✖️": "غلط",
    "📍"  : "مكان",
    "👸" : "ملكه",
    "👑" : "تاج",
    "✔️" : "صح",
    "💌": "قلب",
    "😲" : "مندهش",
    "💦": "ماء",
    "🚫" : "خطا",
    "👏🏻" : "برافو",
    "🏊" :"يسبح",
    "👍🏻": "تمام",
    "⭕️" :"دائره كبيره",
    "🎷" : "ساكسفون",
    "👋": "تلويح باليد",
    "✌🏼": "علامه النصر",
    "🌝":"مبتسم",
    "➿"  : "عقده مزدوجه",
    "💪🏼" : "قوي",
    "📩":  "تواصل معي",
    "☕️": "قهوه",
    "😧" : "قلق و صدمة",
    "🗨": "رسالة",
    "❗️" :"تعجب",
    "🙆🏻": "اشاره موافقه",
    "👯" :"اخوات",
    "©" :  "رمز",
    "👵🏽" :"سيده عجوزه",
    "🐣": "كتكوت",
    "🙌": "تشجيع",
    "🙇": "شخص ينحني",
    "👐🏽":"ايدي مفتوحه",
    "👌🏽": "بالظبط",
    "⁉️" : "استنكار",
    "⚽️": "كوره",
    "🕶" :"حب",
    "🎈" :"بالون",
    "🎀":    "ورده",
    "💵":  "فلوس",
    "😋":  "جائع",
    "😛":  "يغيظ",
    "😠":  "غاضب",
    "✍🏻":  "يكتب",
    "🌾":  "ارز",
    "👣":  "اثر قدمين",
    "❌":"رفض",
    "🍟":"طعام",
    "👬":"صداقة",
    "🐰":"ارنب",
    "☂":"مطر",
    "⚜":"مملكة فرنسا",
    "🐑":"خروف",
    "🗣":"صوت مرتفع",
    "👌🏼":"احسنت",
    "☘":"مرح",
    "😮":"صدمة",
    "😦":"قلق",
    "⭕":"الحق",
    "✏️":"قلم",
    "ℹ":"معلومات",
    "🙍🏻":"رفض",
    "⚪️":"نضارة نقاء",
    "🐤":"حزن",
    "💫":"مرح",
    "💝":"حب",
    "🍔":"طعام",
    "❤︎":"حب",
    "✈️":"سفر",
    "🏃🏻‍♀️":"يسير",
    "🍳":"ذكر",
    "🎤":"مايك غناء",
    "🎾":"كره",
    "🐔":"دجاجة",
    "🙋":"سؤال",
    "📮":"بحر",
    "💉":"دواء",
    "🙏🏼":"رجاء طلب",
    "💂🏿 ":"حارس",
    "🎬":"سينما",
    "♦️":"مرح",
    "💡":"قكرة",
    "‼":"تعجب",
    "👼":"طفل",
    "🔑":"مفتاح",
    "♥️":"حب",
    "🕋":"كعبة",
    "🐓":"دجاجة",
    "💩":"معترض",
    "👽":"فضائي",
    "☔️":"مطر",
    "🍷":"عصير",
    "🌟":"نجمة",
    "☁️":"سحب",
    "👃":"معترض",
    "🌺":"مرح",
    "🔪":"سكينة",
    "♨":"سخونية",
    "👊🏼":"ضرب",
    "✏":"قلم",
    "🚶🏾‍♀️":"يسير",
    "👊":"ضربة",
    "◾️":"وقف",
    "😚":"حب",
    "🔸":"مرح",
    "👎🏻":"لا يعجبني",
    "👊🏽":"ضربة",
    "😙":"حب",
    "🎥":"تصوير",
    "👉":"جذب انتباه",
    "👏🏽":"يصفق",
    "💪🏻":"عضلات",
    "🏴":"اسود",
    "🔥":"حريق",
    "😬":"عدم الراحة",
    "👊🏿":"يضرب",
    "🌿":"ورقه شجره",
    "✋🏼":"كف ايد",
    "👐":"ايدي مفتوحه",
    "☠️":"وجه مرعب",
    "🎉":"يهنئ",
    "🔕" :"صامت",
    "😿":"وجه حزين",
    "☹️":"وجه يائس",
    "😘" :"حب",
    "😰" :"خوف و حزن",
    "🌼":"ورده",
    "💋":  "بوسه",
    "👇":"لاسفل",
    "❣️":"حب",
    "🎧":"سماعات",
    "📝":"يكتب",
    "😇":"دايخ",
    "😈":"رعب",
    "🏃":"يجري",
    "✌🏻":"علامه النصر",
    "🔫":"يضرب",
    "❗️":"تعجب",
    "👎":"غير موافق",
    "🔐":"قفل",
    "👈":"لليمين",
    "™":"رمز",
    "🚶🏽":"يتمشي",
    "😯":"متفاجأ",
    "✊":"يد مغلقه",
    "😻":"اعجاب",
    "🙉" :"قرد",
    "👧":"طفله صغيره",
    "🔴":"دائره حمراء",
    "💪🏽":"قوه",
    "💤":"ينام",
    "👀":"ينظر",
    "✍🏻":"يكتب",
    "❄️":"تلج",
    "💀":"رعب",
    "😤":"وجه عابس",
    "🖋":"قلم",
    "🎩":"كاب",
    "☕️":"قهوه",
    "😹":"ضحك",
    "💓":"حب",
    "☄️ ":"نار",
    "👻":"رعب",
    "❎":"خطء",
    "🤮":"حزن",
    '🏻':"احمر"
    }

nltk.download('punkt')
arr = os.listdir('test_data')
all_datasets=[]
names=[]

import streamlit as st
st.session_state['max_value']=-10000
rec=" "



##################### hendle_emojis ###############################
def hendle_emojis(text,emojis):
  li=[]
  for i in word_tokenize(text):
    for x in i :
      if x not in emojis.keys():
        li.append(i)
        break
      else:
        li.append(emojis[x])
  return " ".join(li)
####################### data cleaning ##########################
def data_cleaning(x):
    new = re.sub(r'[^ء-ي]',' ',x)
    return new
################remove_diacritics##############################
def remove_diacritics(text):
    arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', str(text))
    return text
################################################################################    
dataset['text'] = dataset['text'].apply(handle_emojis, args=[emojis])
dataset['text']=dataset['text'].apply(data_cleaning)
dataset['text']=dataset['text'].apply(remove_diacritics)
dataset['text']=dataset['text'].apply(remove_duplicates)
################################################################################
label_counts = dataset["labels"].value_counts()
# print the label counts
print(label_counts)
################   decoding outputs   ##############################

def decode(x):
  if x == 0:
    return "Negative"
  elif x == 1:
    return "Neutral"
  else:
    return "Positive"
##############################################################

label_counts = dataset["labels"].value_counts()

# print the label counts
print(label_counts)
##############################################################

dataset['labels']=dataset['labels'].apply(encode)
num_rows=dataset.shape[0]
dataset = dataset.sample(frac = 1)
     

train=dataset[:int(num_rows*1)]
test=dataset[int(num_rows*.97)+1:]

     

model_id = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=2, overwrite_output_dir=True)

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    model_id,
    num_labels=3,
    args=model_args,
    use_cuda=True
)
model.train_model(train)
result, model_outputs, wrong_predictions = model.eval_model(test, f1_multiclass=f1_multiclass)

print(result)
result_train, model_outputs_train, wrong_predictions_train = model.eval_model(train, f1_multiclass=f1_multiclass)

print(result_train)

pred=[]
pred_train=[]
for i in model_outputs:
  pred.append(np.argmax(i))
for i in model_outputs_train:
  pred_train.append(np.argmax(i))
     

from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score ,classification_report
from sklearn import metrics
     

print("Test accuracy  :",metrics.accuracy_score(pred, test['labels']) *100 ,"%")
print("MSE [test]         :",mean_squared_error(test['labels'],pred ))
     

import matplotlib.pyplot as plt
import seaborn as sns
     

cf1 = confusion_matrix(test['labels'],pred)
sns.heatmap(cf1,annot=True,fmt = '.0f')
plt.xlabel('prediction')
plt.ylabel('Actual')
plt.title(' Confusion Matrix')
plt.show()
print("           classification_report for test")
print(classification_report(test['labels'], pred))
print("           classification_report for train")
print(classification_report(train['labels'],pred_train))
print("Train accuracy  :",metrics.accuracy_score(pred_train, train['labels']) *100 ,"%")
     

dataset.head(20)
     

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dataset[50000:67000], f1_multiclass=f1_multiclass)


     

wrong_predictions[:50]
     

import joblib

# train a scikit-learn model
model = model

# save the model to a file
joblib.dump(model, 'new_model(1).pkl')
#load model
model=joblib.load('new_model(1).pkl')
#################### Title ######################
st.title("WELCOME TO OUR RECOMMENDATION SYSTEM 😍😍😍")
st.write("WE ARE HERE TO HELP YOU...........!!!")




counter=0
num= st.number_input("Enter The number of inputs ")
for i in range (int(num)):

    file_name = st.text_input(" Input Number " +str(counter+1),"")
    counter+=1
    if file_name=="":
        break

    if file_name+".csv" not in arr:
      max_index =0
      max=fuzz.ratio(file_name,arr[0])
    for i in range(len(arr)):
      x=fuzz.ratio(file_name+".csv",arr[i])
      if x > max:
        max_index=i
        max=x
    file_name=arr[max_index]
    names.append(file_name)

for z in names:
    dataset = pd.read_csv(f'test_data/{z}')
    dataset.columns.values[0] = "text"
    dataset['label'] =0
    #st.dataframe(dataset)
    dataset=dataset.dropna()
    dataset=dataset.drop_duplicates()
    dataset['text'] = dataset['text'].apply(hendle_emojis, args=[emojis])
    dataset['text'] = dataset['text'].apply(lambda x: data_cleaning(x))
    dataset['text'] = dataset['text'].apply(lambda x: remove_diacritics(x))
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    result, model_outputs, wrong_predictions = model.eval_model(dataset)

    # Get the index of the maximum value for each row

    max_index_per_row = np.argmax(model_outputs, axis=1)
    decoded_labels = [decode(x) for x in max_index_per_row ]


# Print the decoded labels
    data = [[text, label] for text, label in zip(dataset['text'], decoded_labels)]
    new_df = pd.DataFrame(data, columns=['text', 'label'])
    all_datasets.append(new_df)

    #Count the number of positive, negative, and neutral labels
    pos_count = decoded_labels.count('Positive')
    neg_count = decoded_labels.count('Negative')
    neu_count = decoded_labels.count('Neutral')

    #Visualize the counts using Streamlit and Plotly
    import streamlit as st
    import plotly.graph_objects as go

    st.write("Info about:", re.sub(r'[^ء-ي ]', ' ', z))
    st.write('Number of Positive Labels:', pos_count)
    st.write('Number of Negative Labels:', neg_count)
    st.write('Number of Neutral Labels:', neu_count)

    labels = ['Positive', 'Negative', 'Neutral']
    counts = [pos_count, neg_count, neu_count]

    fig = go.Figure(data=[go.Pie(labels=labels, values=counts)])

# Adjust the height and width of the plot
    fig.update_layout(
    height=500,  # Adjust the height to your desired size
    width=500,  # Adjust the width to your desired size
    )

    st.plotly_chart(fig)

    if (pos_count /neg_count) > st.session_state['max_value']:
       rec = re.sub(r'[^ء-ي ]', ' ', z)
       st.session_state['max_value'] = (pos_count / neg_count)



st.write("HERE'S THE BEST THING FOR YOU 😍👉", f"{rec} \U0001F48E")


