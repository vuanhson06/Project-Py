{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23abbdcf",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mRunning cells with 'Python 3.13.5' requires the ipykernel package.\n",
      "\u001b[1;31m<a href='command:jupyter.createPythonEnvAndSelectController'>Create a Python Environment</a> with the required packages.\n",
      "\u001b[1;31mOr install 'ipykernel' using the command: 'c:/Users/Admin/AppData/Local/Programs/Python/Python313/python3.13t.exe -m pip install ipykernel -U --user --force-reinstall'"
     ]
    }
   ],
   "source": [
    "\n",
    "import os\n",
    "import csv\n",
    "import json\n",
    "import random\n",
    "import joblib\n",
    "import numpy as np\n",
    "from typing import List, Tuple, Dict\n",
    "\n",
    "# -----------------------------\n",
    "# 0) Cấu hình dự án\n",
    "# -----------------------------\n",
    "RANDOM_STATE = 42\n",
    "VOCAB_SIZE = 3000\n",
    "\n",
    "RAW_CSV = \"data/raw/sms.csv\"              # file gốc: 2 cột \"label\",\"text\"\n",
    "TRAIN_OUT = \"data/processed/train.csv\"    # output train\n",
    "TEST_OUT  = \"data/processed/test.csv\"     # output test\n",
    "VEC_PKL   = \"artifacts/vectorizer.pkl\"    # pickle vectorizer cho Bước 3/4\n",
    "VOCAB_TXT = \"artifacts/vocab.txt\"         # vocab (tham khảo)\n",
    "TEST_SIZE = 0.2                           # 80/20 split\n",
    "\n",
    "# -----------------------------\n",
    "# 1) Stopwords & Cleaning\n",
    "# -----------------------------\n",
    "STOPWORDS = {\n",
    "    \"a\",\"an\",\"the\",\"is\",\"are\",\"am\",\"was\",\"were\",\"be\",\"been\",\"being\",\"i\",\"you\",\"he\",\"she\",\"it\",\"we\",\"they\",\"me\",\"him\",\"her\",\"us\",\"them\",\n",
    "    \"this\",\"that\",\"these\",\"those\",\"there\",\"here\",\"of\",\"to\",\"in\",\"on\",\"for\",\"from\",\"with\",\"by\",\"at\",\"as\",\"about\",\"into\",\"over\",\"after\",\n",
    "    \"before\",\"between\",\"and\",\"or\",\"but\",\"if\",\"then\",\"so\",\"because\",\"while\",\"than\",\"though\",\"although\",\"not\",\"no\",\"do\",\"does\",\"did\",\"doing\",\n",
    "    \"done\",\"dont\",\"didnt\",\"doesnt\",\"isnt\",\"arent\",\"wasnt\",\"werent\",\"cant\",\"cannot\",\"my\",\"your\",\"his\",\"her\",\"its\",\"our\",\"their\",\n",
    "    \"have\",\"has\",\"had\",\"having\",\"will\",\"would\",\"shall\",\"should\",\"can\",\"could\",\"may\",\"might\",\"must\",\n",
    "    # thêm một số mảnh contraction phổ biến sau khi bỏ ký tự\n",
    "    \"im\",\"ive\",\"youre\",\"hes\",\"shes\",\"weve\",\"theyre\",\"ill\",\"youll\",\"dont\",\"cant\",\"wont\",\"didnt\",\"couldnt\",\"shouldnt\",\"wouldnt\",\"lets\"\n",
    "} \n",
    "def clean_text(text: str) -> str:\n",
    "    import re\n",
    "    text = text.lower()\n",
    "    #Bỏ số và ký tự đặc biệt\n",
    "    text = re.sub(r'[^a-z\\s]', ' ', text)\n",
    "    #Tách từ và bỏ stopwords\n",
    "    words = [w for w in text.split() if w not in STOPWORDS]\n",
    "    return ' '.join(words)\n",
    "\n",
    "\n",
    "def keep_letters_and_spaces(s: str) -> str:\n",
    "    # chỉ giữ a-z và space, hạ chữ thường (chúng ta sẽ lower trước khi gọi hàm này)\n",
    "    out = []\n",
    "    for ch in s:\n",
    "        if 'a' <= ch <= 'z' or ch == ' ':\n",
    "            out.append(ch)\n",
    "        else:\n",
    "            out.append(' ')\n",
    "    return ''.join(out)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
