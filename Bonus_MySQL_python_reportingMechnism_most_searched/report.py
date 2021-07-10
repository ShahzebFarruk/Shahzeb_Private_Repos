from datetime import datetime
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import ast
import json
import os
from azure.storage.blob import BlobClient
from os import path
import shutil
import zipfile
import sys
from os.path import basename

from app.models import Response as ChatbotResponse
from app.models import Messages, Evaluations

from response_learning import bert_embedding, fuzzy_ratios
import spacy 
from spacy_langdetect import LanguageDetector
from response_learning import lang_detect
from active_hours_run_file import active_hrs_class_main_call

from reporting import language_report, topic_content_time_report


class reporter:
    def __init__(self, company_id):
        self.datetime_info = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " INFO: "
        print(self.datetime_info + "Acquring reports.")
        # self.lang_report_dict, self.topic_dict, self.content_dict, self.content_list, self.average_time = None, None, None, None, None

        self.company_id = company_id  
        self.language_report = language_report.language_report()
        self.topic_content_time_report = topic_content_time_report.topic_content_time_report()

    def get_report(self):
        lang_report_dict = self.get_langauge_report()
        topic_dict, content_dict, content_list, average_time = self.get_topic_content_time_report()
        active_hours_list=self.get_active_hours()
        return lang_report_dict, topic_dict, content_dict, content_list, average_time, active_hours_list

    def get_langauge_report(self):
        lang_report_dict = self.language_report.get_language_report(self.company_id)
        return lang_report_dict

    def get_topic_content_time_report(self):
        topic_dict, content_dict, content_list, average_time = self.topic_content_time_report.get_topic_content_time_report(self.company_id)
        return topic_dict, content_dict, content_list, average_time
    def get_active_hours(self):
        active_hours_list=self.active_hrs_class_main_call.active_function_main(self)
        return active_hours_list
